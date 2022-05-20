import os
import numpy as np
import pandas as pd

def data_split_index(seq):

    paired_data_len = len(seq)
    valid_num = int(paired_data_len * 0.15)
    test_num = int(paired_data_len * 0.1)

    valid_index = np.random.choice(paired_data_len, valid_num, replace=False)
    train_index = list(set(range(paired_data_len)) - set(valid_index))
    test_index = np.random.choice(train_index, test_num, replace=False)
    train_index = list(set(train_index) - set(test_index))

    return train_index, valid_index, test_index

def total_data_load(args):

    src_sequences = dict()
    trg_sequences = dict()

    #===================================#
    #============Translation============#
    #===================================#

    # WMT2016 Multimodal [DE -> EN]

    if args.data_name == 'WMT2016_Multimodal':
        args.data_path = os.path.join(args.data_path,'WMT/2016/multi_modal')

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

    # WMT2014 Translation [DE -> EN]
        
    elif args.data_name == 'WMT2014_de_en':
        args.data_path = os.path.join(args.data_path,'WMT/2014/de_en')

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

    # elif args.data_name == 'shift_challenge':
    #     args.data_path = os.path.join(args.data_path,'shift_challenge')

    # Korpora [EN -> KR]

    elif args.data_name == 'korpora':
        args.data_path = os.path.join(args.data_path,'korpora')

        en = pd.read_csv(os.path.join(args.data_path, 'pair_eng.csv'), names=['en'])['en']
        kr = pd.read_csv(os.path.join(args.data_path, 'pair_kor.csv'), names=['kr'])['kr']

        train_index, valid_index, test_index = data_split_index(en)

        src_sequences['train'] = [en[i] for i in train_index]
        trg_sequences['train'] = [kr[i] for i in train_index]

        src_sequences['valid'] = [en[i] for i in valid_index]
        trg_sequences['valid'] = [kr[i] for i in valid_index]

        src_sequences['test'] = [en[i] for i in test_index]
        trg_sequences['test'] = [kr[i] for i in test_index]

    # AIHUB [EN -> KR]

    elif args.data_name == 'aihub_en_kr':
        args.data_path = os.path.join(args.data_path,'AI_Hub_KR_EN')

        dat = pd.read_csv(os.path.join(args.data_path, '1_구어체(1).csv'))

        train_index, valid_index, test_index = data_split_index(dat)

        src_sequences['train'] = [dat['EN'][i] for i in train_index]
        trg_sequences['train'] = [dat['KR'][i] for i in train_index]

        src_sequences['valid'] = [dat['EN'][i] for i in valid_index]
        trg_sequences['valid'] = [dat['KR'][i] for i in valid_index]

        src_sequences['test'] = [dat['EN'][i] for i in test_index]
        trg_sequences['test'] = [dat['KR'][i] for i in test_index]

    #===================================#
    #========Text Style Transfer========#
    #===================================#

    # GYAFC [Informal -> Formal]

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

        train_index, valid_index, test_index = data_split_index(record_list_src)

        src_sequences['train'] = [record_list_src[i] for i in train_index]
        trg_sequences['train'] = [record_list_trg[i] for i in train_index]

        src_sequences['valid'] = [record_list_src[i] for i in valid_index]
        trg_sequences['valid'] = [record_list_trg[i] for i in valid_index]

        src_sequences['test'] = [record_list_src[i] for i in test_index]
        trg_sequences['test'] = [record_list_trg[i] for i in test_index]

    # WNC [Biased -> Neutral]

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

    return src_sequences, trg_sequences