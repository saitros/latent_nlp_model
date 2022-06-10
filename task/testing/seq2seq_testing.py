# Import modules
import os
import gc
import h5py
import time
import pickle
import logging
import sentencepiece as spm
from tqdm import tqdm
from collections import defaultdict
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
# Import Huggingface
from transformers import BartTokenizerFast
# Import custom modules
from model.dataset import Seq2SeqDataset
from model.custom_transformer.transformer import Transformer
from utils import TqdmLoggingHandler, write_log

def seq2seq_testing(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start testing!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
    if args.tokenizer == 'spm':
        save_name = f'processed_{args.task}_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed_{args.task}_{args.tokenizer}.hdf5'

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'r') as f:
        test_src_input_ids = f.get('test_src_input_ids')[:]
        test_src_attention_mask = f.get('test_src_attention_mask')[:]
        if args.task in ['translation', 'style_transfer', 'summarization']:
            test_trg_input_ids = f.get('test_trg_input_ids')[:]
            test_trg_attention_mask = f.get('test_trg_attention_mask')[:]
        elif args.task in ['reconstruction']:
            test_trg_input_ids = ftest_src_input_ids
            test_trg_attention_mask = test_src_attention_mask

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_word2id = data_['src_word2id']
        trg_word2id = data_['trg_word2id']
        trg_id2word = {v: k for k, v in trg_word2id.items()}
        src_vocab_num = len(src_word2id)
        trg_vocab_num = len(trg_word2id)
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    #===================================#
    #===========Test setting============#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Loading model...')
    if args.model_type == 'custom_transformer':
        model = Transformer(src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                            pad_idx=args.pad_id, bos_idx=args.bos_id, eos_idx=args.eos_id,
                            d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                            dim_feedforward=args.dim_feedforward,
                            num_common_layer=args.num_common_layer, num_encoder_layer=args.num_encoder_layer,
                            num_decoder_layer=args.num_decoder_layer,
                            src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                            dropout=args.dropout, embedding_dropout=args.embedding_dropout,
                            trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
                            emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing, 
                            variational_mode=args.variational_mode, z_var=args.z_var,
                            parallel=args.parallel)
        tgt_subsqeunt_mask = model.generate_square_subsequent_mask(args.trg_max_len - 1, device)
    elif args.model_type == 'T5':
        model = custom_T5(isPreTrain=args.isPreTrain, d_latent=args.d_latent, 
                        variational_mode=args.variational_mode, z_var=args.z_var,
                        decoder_full_model=True, device=device)
        tgt_subsqeunt_mask = None
    elif args.model_type == 'bart':
        model = custom_Bart(isPreTrain=args.isPreTrain, PreTrainMode='large',
                            variational_mode=args.variational_mode, z_var=args.z_var,
                            d_latent=args.d_latent, emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing)
        tgt_subsqeunt_mask = None
    # elif args.model_type == 'Bert':
    #     model = custom_T5(isPreTrain=args.isPreTrain, d_latent=args.d_latent, 
    #                       variational_mode=args.variational_mode, 
    #                       decoder_full_model=True, device=device)
    model = model.to(device)

    # loda model
    model = model.to(device)
    save_path = os.path.join(args.model_save_path, args.task, args.data_name, args.tokenizer)
    save_file_name = os.path.join(save_path, 
                                    f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
    model.load_state_dict(torch.load(save_file_name)['model'])
    model = model.eval()

    # 2) Dataloader setting
    test_dataset = Seq2SeqDataset(src_list=test_src_input_ids, src_att_list=test_src_attention_mask,
                                trg_list=test_trg_input_ids, trg_att_list=test_trg_attention_mask,
                                src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                                pad_idx=model.pad_idx, eos_idx=model.eos_idx)
    test_dataloader = DataLoader(test_dataset, drop_last=False, batch_size=args.test_batch_size, shuffle=False,
                                pin_memory=True, num_workers=args.num_workers)
    write_log(logger, f"Total number of trainingsets  iterations - {len(test_dataset)}, {len(test_dataloader)}")

    # 3)
    if args.tokenizer == 'bart':
        tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    else:
        preprocess_save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
        spm_model = spm.SentencePieceProcessor()
        spm_model.Load(f'{preprocess_save_path}/m_src_{args.sentencepiece_model}_{args.trg_vocab_size}.model')

    # Beam search
    with torch.no_grad():
        for i, batch_iter in enumerate(tqdm(test_dataloader, bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            # Input, output setting
            src_sequence = batch_iter[0]
            src_att = batch_iter[1]
            trg_sequence = batch_iter[2]
            trg_att = batch_iter[3]

            src_sequence = src_sequence.to(device, non_blocking=True)
            src_att = src_att.to(device, non_blocking=True)
            trg_sequence = trg_sequence.to(device, non_blocking=True)
            trg_att = trg_att.to(device, non_blocking=True)

            predicted = model.generate(src_sequence, src_att, beam_size=5, beam_alpha=0.7, repetition_penalty=0.7, device=device)

            for i, predicted_sequence in enumerate(predicted):
                print('Predicted')
                print(spm_model.DecodeIds(predicted_sequence))
                print('Label')
                print(spm_model.DecodeIds(trg_sequence.tolist()[i]))
                print()