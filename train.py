# Import modules
import os
import gc
import pickle
import logging
from apex import amp
from time import time
from tqdm import tqdm
# Import PyTorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
# Import custom modules

def training(args):
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

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()
    with open(os.path.join(args.preprocess_path, 'processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        train_src_indices = data_['train_src_indices']
        valid_src_indices = data_['valid_src_indices']
        train_trg_indices = data_['train_trg_indices']
        valid_trg_indices = data_['valid_trg_indices']
        src_word2id = data_['src_word2id']
        trg_word2id = data_['trg_word2id']
        src_vocab_num = len(src_word2id)
        trg_vocab_num = len(trg_word2id)
        del data_
    gc.enable()
    write_log(logger, "Finished loading data!")

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(train_src_indices, train_trg_indices, 
                            min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
        'valid': CustomDataset(valid_src_indices, valid_trg_indices,
                            min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

