# Import Modules
import os
import time
import pickle
import logging
import pandas as pd
from tqdm import tqdm
# Import PyTorch
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
# Import Custom Modules
# WAE
from model.wae.dataset import CustomDataset, PadCollate
from model.wae.model import TransformerWAE, Discirminator_model
from model.wae.loss import mmd, sample_z, log_density_igaussian
# VAE
from model.vae.model import TransformerVAE
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log

def augment_training(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger setting
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, "Augmentation Training Start")

    #===================================#
    #============Data Load==============#
    #===================================#