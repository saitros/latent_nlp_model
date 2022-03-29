import os
import time
import pickle
import logging
# Import custom modules
from model.tokenizer.spm_tokenize import spm_tokenizing
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

    print('SentencePiece Training')
    start_time = time.time()

    if args.tokenizer == 'spm':
        processed_src, processed_trg, word2id = spm_tokenizing(src_sequences, trg_sequences, args)
    if args.tokenizer == 'bert':
        processed_src, processed_trg, word2id = spm_tokenizing(src_sequences, trg_sequences, args)
    if args.tokenizer == 'bart':
        processed_src, processed_trg, word2id = spm_tokenizing(src_sequences, trg_sequences, args)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence save setting...')
    start_time = time.time()

    save_name = f'processed_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.pkl'
    with open(os.path.join(args.preprocess_path, save_name), 'wb') as f:
        pickle.dump({
            'train_src_indices': processed_src['train'],
            'valid_src_indices': processed_src['valid'],
            'train_trg_indices': processed_trg['train'],
            'valid_trg_indices': processed_trg['valid'],
            'src_word2id': word2id['src'],
            'trg_word2id': word2id['trg']
        }, f)

    with open(os.path.join(args.preprocess_path, 'test_' + save_name), 'wb') as f:
        pickle.dump({
            'test_src_indices': processed_src['test'],
            'test_trg_indices': processed_src['test'],
            'src_word2id': word2id['src'],
            'trg_word2id': word2id['trg']
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')