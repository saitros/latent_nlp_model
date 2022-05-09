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

def spm_tokenizing(src_sequences, trg_sequences, args):

    # 1) Source lanugage
    processed_src, processed_trg, word2id = dict(), dict(), dict()

    # Make text to train vocab
    with open(f'{args.preprocess_path}/{args.task}/src.txt', 'w') as f:
        for text in src_sequences['train']:
            f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.preprocess_path}/{args.task}/src.txt --model_prefix={args.preprocess_path}/{args.task}/m_src_{args.src_vocab_size} '
        f'--vocab_size={args.src_vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--model_type={args.sentencepiece_model}')

    src_vocab = list()
    with open(f'{args.preprocess_path}/{args.task}/m_src_{args.src_vocab_size}.vocab') as f:
        for line in f:
            src_vocab.append(line[:-1].split('\t')[0])

    word2id['src'] = {w: i for i, w in enumerate(src_vocab)}
    spm_src = spm.SentencePieceProcessor()
    spm_src.Load(f'{args.preprocess_path}/{args.task}/m_src_{args.src_vocab_size}.model')

    # Encoding
    train_src_input_ids = tuple(
        [args.bos_id] + spm_src.encode(
                            text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [args.eos_id] for text in src_sequences['train']
    )
    valid_src_input_ids = tuple(
        [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in src_sequences['valid']
    )
    test_src_input_ids = tuple(
        [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in src_sequences['test']
    )

    # Pad token add
    processed_src['train'] = pad_add(train_src_input_ids, args.src_max_len)
    processed_src['valid'] = pad_add(valid_src_input_ids, args.src_max_len)
    processed_src['test'] = pad_add(test_src_input_ids, args.src_max_len)

    # 2) Target lanugage

    # Make text to train vocab
    with open(f'{args.preprocess_path}/{args.task}/trg.txt', 'w') as f:
        for text in trg_sequences['train']:
            f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.preprocess_path}/{args.task}/trg.txt --model_prefix={args.preprocess_path}/{args.task}/m_trg_{args.trg_vocab_size} '
        f'--vocab_size={args.trg_vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--model_type={args.sentencepiece_model}')

    trg_vocab = list()
    with open(f'{args.preprocess_path}/{args.task}/m_trg_{args.trg_vocab_size}.vocab') as f:
        for line in f:
            trg_vocab.append(line[:-1].split('\t')[0])

    word2id['trg'] = {w: i for i, w in enumerate(trg_vocab)}
    spm_trg = spm.SentencePieceProcessor()
    spm_trg.Load(f'{args.preprocess_path}/{args.task}/m_trg_{args.trg_vocab_size}.model')

    train_trg_input_ids = tuple(
        [args.bos_id] + spm_trg.encode(
                            text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [args.eos_id] for text in trg_sequences['train']
    )
    valid_trg_input_ids = tuple(
        [args.bos_id] + spm_trg.encode(text, out_type=int) + [args.eos_id] for text in trg_sequences['valid']
    )
    test_trg_input_ids = tuple(
        [args.bos_id] + spm_trg.encode(text, out_type=int) + [args.eos_id] for text in trg_sequences['test']
    )

    # Pad token add
    processed_trg['train'] = pad_add(train_trg_input_ids, args.trg_max_len)
    processed_trg['valid'] = pad_add(valid_trg_input_ids, args.trg_max_len)
    processed_trg['test'] = pad_add(test_trg_input_ids, args.trg_max_len)

    return processed_src, processed_trg, word2id