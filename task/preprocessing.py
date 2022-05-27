import os
import time
import h5py
import pickle
import logging
import numpy as np
# Import custom modules
from tokenizer.spm_tokenize import spm_tokenizing
from tokenizer.plm_tokenize import plm_tokenizing
from tokenizer.spacy_tokenize import spacy_tokenizing
from task.data_load import total_data_load
from utils import TqdmLoggingHandler, write_log

def preprocessing(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, 'Start preprocessing!')

    src_list, trg_list = total_data_load(args)

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.data_name in ['WMT2016_Multimodal', 'WMT2014_de_en']:
        src_language = 'de'
        trg_language = 'en'
    elif args.data_name in ['korpora', 'aihub_en_kr']:
        src_language = 'en'
        trg_language = 'kr'
    elif args.data_name in ['GYAFC', 'WNC', 'cnn_dailymail']:
        src_language = 'en'
        trg_language = 'en'
    elif args.data_name in ['korean_hate_speech', 'NSMC']:
        src_language = 'kr'
    elif args.data_name in ['IMDB', 'ProsCons', 'MR']:
        src_language = 'en'

    if args.task in ['classification', 'reconstruction']:
        if args.tokenizer == 'spm':
            processed_src, word2id_src = spm_tokenizing(src_list, args, domain='src')
        # elif args.tokenizer == 'spacy':
        #     processed_src, processed_trg, word2id = spacy_tokenizing(src_list, trg_list, args)
        else:
            processed_src, word2id_src = plm_tokenizing(src_list, args, domain='src', language=src_language)

    elif args.task in ['translation', 'style_transfer', 'summarization']:
        if args.tokenizer == 'spm':
            processed_src, word2id_src = spm_tokenizing(src_list, args, domain='src')
            processed_trg, word2id_trg = spm_tokenizing(trg_list, args, domain='trg')
        # elif args.tokenizer == 'spacy':
        #     processed_src, processed_trg, word2id = spacy_tokenizing(src_list, trg_list, args)
        else:
            processed_src, word2id_src = plm_tokenizing(src_list, args, domain='src', language=src_language)
            processed_trg, word2id_trg = plm_tokenizing(trg_list, args, domain='trg', language=trg_language)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.tokenizer == 'spm':
        save_name = f'processed_{args.task}_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed_{args.task}_{args.tokenizer}.hdf5'

    with h5py.File(os.path.join(save_path, save_name), 'w') as f:
        f.create_dataset('train_src_input_ids', data=processed_src['train']['input_ids'])
        f.create_dataset('train_src_attention_mask', data=processed_src['train']['attention_mask'])
        f.create_dataset('valid_src_input_ids', data=processed_src['valid']['input_ids'])
        f.create_dataset('valid_src_attention_mask', data=processed_src['valid']['attention_mask'])
        if args.task in ['translation', 'style_transfer','summarization']:
            f.create_dataset('train_trg_input_ids', data=processed_trg['train']['input_ids'])
            f.create_dataset('train_trg_attention_mask', data=processed_trg['train']['attention_mask'])
            f.create_dataset('valid_trg_input_ids', data=processed_trg['valid']['input_ids'])
            f.create_dataset('valid_trg_attention_mask', data=processed_trg['valid']['attention_mask'])
        elif args.task in ['classification']:
            f.create_dataset('train_label', data=np.array(trg_list['train']).astype(int))
            f.create_dataset('valid_label', data=np.array(trg_list['valid']).astype(int))

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'w') as f:
        f.create_dataset('test_src_input_ids', data=processed_src['test']['input_ids'])
        f.create_dataset('test_src_attention_mask', data=processed_src['test']['attention_mask'])
        if args.task in ['translation', 'style_transfer','summarization']:
            f.create_dataset('test_trg_input_ids', data=processed_trg['test']['input_ids'])
            f.create_dataset('test_trg_attention_mask', data=processed_trg['test']['attention_mask'])
        elif args.task in ['classification']:
            f.create_dataset('test_label', data=np.array(trg_list['test']).astype(int))

    # Word2id pickle file save
    if args.task in ['classification', 'reconstruction']:
        word2id_dict = {'src_word2id' : word2id_src}
    else:
        word2id_dict = {
            'src_word2id': word2id_src,
            'trg_word2id': word2id_trg
        }
    
    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'wb') as f:
        pickle.dump(word2id_dict, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')