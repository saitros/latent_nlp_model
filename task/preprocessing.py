import os
import time
import h5py
import logging
import numpy as np
# Import custom modules
from model.tokenizer.spm_tokenize import spm_tokenizing
from model.tokenizer.plm_tokenize import plm_tokenizeing
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

    if args.data_name == 'WMT2016_Multimodal':
        args.data_path = os.path.join(args.data_path,'2016/multi_modal')
        
    elif args.data_name == 'WMT2014_de_en':
        args.data_path = os.path.join(args.data_path,'2014/de_en')

    # 1) Train data load
    with open(os.path.join(args.data_path, 'train.de'), 'r') as f:
        src_sequences['train'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'train.en'), 'r') as f:
        trg_sequences['train'] = [x.replace('\n', '') for x in f.readlines()]

    # 2) Valid data load
    with open(os.path.join(args.data_path, 'val.de'), 'r') as f:
        src_sequences['valid'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'val.en'), 'r') as f:
        trg_sequences['valid'] = [x.replace('\n', '') for x in f.readlines()]

    # 3) Test data load
    with open(os.path.join(args.data_path, 'test.de'), 'r') as f:
        src_sequences['test'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'test.en'), 'r') as f:
        trg_sequences['test'] = [x.replace('\n', '') for x in f.readlines()]

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.tokenizer == 'spm':
        processed_src, processed_trg, word2id = spm_tokenizing(src_sequences, trg_sequences, args)
    else:
        processed_src, processed_trg, word2id = plm_tokenizeing(src_sequences, trg_sequences, args)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.tokenizer)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.tokenizer == 'spm':
        save_name = f'processed_{args.data_name}_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed_{args.data_name}_{args.tokenizer}.hdf5'

    p_file = h5py.File(os.path.join(save_path, save_name), 'w')
    p_file.create_group('train')
    p_file.create_group('valid')
    # Input id write
    p_file['train'].create_dataset('src_input_ids', data=np.array(processed_src['train']['input_ids']))
    p_file['train'].create_dataset('trg_input_ids', data=np.array(processed_trg['train']['input_ids']))
    p_file['valid'].create_dataset('src_input_ids', data=np.array(processed_src['valid']['input_ids']))
    p_file['valid'].create_dataset('trg_input_ids', data=np.array(processed_trg['valid']['input_ids']))
    # word2id write
    p_file.create_dataset('src_word2id', data=word2id['src'])
    p_file.create_dataset('src_word2id', data=word2id['src'])
    p_file.close()

    t_file = h5py.File(os.path.join(save_path, 'test_' + save_name), 'w')
    t_file.create_group('test')
    # Input id write
    t_file['test'].create_dataset('src_input_ids', data=list(processed_src['test']['input_ids']))
    t_file['test'].create_dataset('trg_input_ids', data=list(processed_trg['test']['input_ids']))
    # word2id write
    t_file.create_dataset('src_word2id', data=word2id['src'])
    t_file.create_dataset('trg_word2id', data=word2id['trg'])
    t_file.close()

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')