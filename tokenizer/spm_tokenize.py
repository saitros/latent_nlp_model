import os
import numpy as np
import sentencepiece as spm

def pad_add(list_, max_len: int = 300):
    ind_list = list()
    for ind_ in list_:
        if len(ind_) <= max_len:
            ind = np.zeros(max_len, dtype=np.int32)
            ind[:len(ind_)] = np.array(ind_, dtype=np.int32)
            ind_list.append(ind)
    return np.array(ind_list, dtype=np.int32)

def spm_tokenizing(sequence_dict: dict, args, domain: str = 'src'):

    # 0) Path Setting
    if not os.path.exists(os.path.join(args.preprocess_path, args.task, args.data_name)):
        os.mkdir(os.path.join(args.preprocess_path, args.task, args.data_name))

    preprocess_save_path = os.path.join(args.preprocess_path, args.task, args.data_name, args.tokenizer)
    if not os.path.exists(preprocess_save_path):
        os.mkdir(preprocess_save_path)

    # 1) Pre-setting
    processed_sequences = dict()
    if domain == 'src':
        vocab_size = args.src_vocab_size
        character_coverage = args.src_character_coverage
    if domain == 'trg':
        vocab_size = args.trg_vocab_size
        character_coverage = args.trg_character_coverage

    # Make text to train vocab
    with open(f'{preprocess_save_path}/{domain}.txt', 'w') as f:
        for text in sequence_dict['train']:
            f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={preprocess_save_path}/src.txt --model_type={args.sentencepiece_model} '
        f'--model_prefix={preprocess_save_path}/m_{domain}_{args.sentencepiece_model}_{vocab_size} '
        f'--vocab_size={vocab_size} --character_coverage={character_coverage}'
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--split_by_whitespace=true')

    vocab_list = list()
    with open(f'{preprocess_save_path}/m_{domain}_{args.sentencepiece_model}_{vocab_size}.vocab') as f:
        for line in f:
            vocab_list.append(line[:-1].split('\t')[0])

    word2id = {w: i for i, w in enumerate(vocab_list)}
    spm_src = spm.SentencePieceProcessor()
    spm_src.Load(f'{preprocess_save_path}/m_{domain}_{args.sentencepiece_model}_{vocab_size}.model')

    # Encoding
    train_src_input_ids = tuple(
        [args.bos_id] + spm_src.encode(
                            text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [args.eos_id] for text in sequence_dict['train']
    )
    valid_src_input_ids = tuple(
        [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in sequence_dict['valid']
    )
    test_src_input_ids = tuple(
        [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in sequence_dict['test']
    )

    # Pad token add
    processed_sequences['train'] = pad_add(train_src_input_ids, args.src_max_len)
    processed_sequences['valid'] = pad_add(valid_src_input_ids, args.src_max_len)
    processed_sequences['test'] = pad_add(test_src_input_ids, args.src_max_len)

    return processed_sequences, word2id