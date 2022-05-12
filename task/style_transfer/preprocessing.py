import os
import time
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
# Import custom modules
from tokenizer.spm_tokenize import spm_tokenizing
from tokenizer.plm_tokenize import plm_tokenizeing
from tokenizer.spacy_tokenize import spacy_tokenizing
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

    write_log(logger, 'Start preprocessing!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    src_sequences = dict()
    trg_sequences = dict()

    if args.data_name == 'GYAFC':
        args.data_path = os.path.join(args.data_path,'GYAFC_Corpus')

        # 1) Train data load
        with open(os.path.join(args.data_path, 'Entertainment_Music/train/informal_em_train.txt'), 'r') as f:
            music_src = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'Entertainment_Music/train/formal_em_train.txt'), 'r') as f:
            music_trg = [x.replace('\n', '') for x in f.readlines()]

        with open(os.path.join(args.data_path, 'Family_Relationships/train/informal_fr_train.txt'), 'r') as f:
            family_src = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'Family_Relationships/train/formal_fr_train.txt'), 'r') as f:
            family_trg = [x.replace('\n', '') for x in f.readlines()]

        assert len(music_src) == len(music_trg)
        assert len(family_src) == len(family_trg)

        record_list_src = music_src + family_src
        record_list_trg = music_trg + family_trg

        # 2) Data split
        paired_data_len = len(record_list_src)
        valid_num = int(paired_data_len * 0.15)
        test_num = int(paired_data_len * 0.1)

        valid_index = np.random.choice(paired_data_len, valid_num, replace=False)
        train_index = list(set(range(paired_data_len)) - set(valid_index))
        test_index = np.random.choice(train_index, test_num, replace=False)
        train_index = list(set(train_index) - set(test_index))

        src_sequences['train'] = [record_list_src[i] for i in train_index]
        trg_sequences['train'] = [record_list_trg[i] for i in train_index]

        src_sequences['valid'] = [record_list_src[i] for i in valid_index]
        trg_sequences['valid'] = [record_list_trg[i] for i in valid_index]

        src_sequences['test'] = [record_list_src[i] for i in test_index]
        trg_sequences['test'] = [record_list_trg[i] for i in test_index]

    if args.data_name == 'WNC':
        args.data_path = os.path.join(args.data_path,'bias_data')
        col_names = ['ID','src_tok','tgt_tok','src_raw','trg_raw','src_POS','trg_parse_tags']

        train_dat = pd.read_csv(os.path.join(args.data_path, 'WNC/biased.word.train'), 
                                sep='\t', names=col_names)
        valid_dat = pd.read_csv(os.path.join(args.data_path, 'WNC/biased.word.dev'),
                                sep='\t', names=col_names)
        test_dat = pd.read_csv(os.path.join(args.data_path, 'WNC/biased.word.test'),
                               sep='\t', names=col_names)

        src_sequences['train'] = train_dat['src_raw'].tolist()
        trg_sequences['train'] = train_dat['trg_raw'].tolist()
        src_sequences['valid'] = valid_dat['src_raw'].tolist()
        trg_sequences['valid'] = valid_dat['trg_raw'].tolist()
        src_sequences['test'] = test_dat['src_raw'].tolist()
        trg_sequences['test'] = test_dat['trg_raw'].tolist()

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.tokenizer == 'spm':
        processed_src, processed_trg, word2id = spm_tokenizing(src_sequences, trg_sequences, args)
    elif args.tokenizer == 'spacy':
        processed_src, processed_trg, word2id = spacy_tokenizing(src_sequences, trg_sequences, args)
    else:
        processed_src, processed_trg, word2id = plm_tokenizeing(src_sequences, trg_sequences, args)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.task, args.data_name, args.tokenizer)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.tokenizer == 'spm':
        save_name = f'processed_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed_{args.tokenizer}.hdf5'

    with h5py.File(os.path.join(save_path, save_name), 'w') as f:
        f.create_dataset('train_src_input_ids', data=processed_src['train'])
        f.create_dataset('train_trg_input_ids', data=processed_trg['train'])
        f.create_dataset('valid_src_input_ids', data=processed_src['valid'])
        f.create_dataset('valid_trg_input_ids', data=processed_trg['valid'])

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'w') as f:
        f.create_dataset('test_src_input_ids', data=processed_src['test'])
        f.create_dataset('test_trg_input_ids', data=processed_trg['test'])

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'wb') as f:
        pickle.dump({
            'src_word2id': word2id['src'],
            'trg_word2id': word2id['trg']
        }, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')