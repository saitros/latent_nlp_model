import numpy as np
import sentencepiece as spm

def spm_tokenizing(src_sequences, trg_sequences, args):
    # 0) Dictionary setting
    processed_src, processed_trg, word2id = dict(), dict(), dict()
    processed_src['train'], processed_src['valid'], processed_src['test'] = dict(), dict(), dict()
    processed_trg['train'], processed_trg['valid'], processed_trg['test'] = dict(), dict(), dict()

    # 1) Source lanugage
    # Split to train, valid, test
    train_src_sequences = src_sequences['train']
    valid_src_sequences = src_sequences['valid']
    test_src_sequences = src_sequences['test']

    # Make text to train vocab
    with open(f'{args.preprocess_path}/src.txt', 'w') as f:
        for text in train_src_sequences:
            f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.preprocess_path}/src.txt --model_prefix={args.preprocess_path}/m_src_{args.src_vocab_size} '
        f'--vocab_size={args.src_vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--model_type={args.sentencepiece_model}')

    src_vocab = list()
    with open(f'{args.preprocess_path}/m_src_{args.src_vocab_size}.vocab') as f:
        for line in f:
            src_vocab.append(line[:-1].split('\t')[0])

    word2id['src'] = {w: i for i, w in enumerate(src_vocab)}
    spm_src = spm.SentencePieceProcessor()
    spm_src.Load(f'{args.preprocess_path}/m_src_{args.src_vocab_size}.model')

    input_id_list = tuple(
        [args.bos_id] + spm_src.encode(
                            text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [args.eos_id] for text in train_src_sequences
    )
    processed_src['train']['input_ids'] = [np.array(input_, dtype=np.int32) for input_ in input_id_list]
    input_id_list = tuple(
        [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in valid_src_sequences
    )
    processed_src['valid']['input_ids'] = [np.array(input_, dtype=np.int32) for input_ in input_id_list]
    input_id_list = tuple(
        [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in test_src_sequences
    )
    processed_src['test']['input_ids'] = [np.array(input_, dtype=np.int32) for input_ in input_id_list]

    # 2) Target lanugage
    # Split to train, valid, test
    train_trg_sequences = trg_sequences['train']
    valid_trg_sequences = trg_sequences['valid']
    test_trg_sequences = trg_sequences['test']

    # Make text to train vocab
    with open(f'{args.preprocess_path}/trg.txt', 'w') as f:
        for text in train_trg_sequences:
            f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.preprocess_path}/trg.txt --model_prefix={args.preprocess_path}/m_trg_{args.trg_vocab_size} '
        f'--vocab_size={args.trg_vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--model_type={args.sentencepiece_model}')

    trg_vocab = list()
    with open(f'{args.preprocess_path}/m_trg_{args.trg_vocab_size}.vocab') as f:
        for line in f:
            trg_vocab.append(line[:-1].split('\t')[0])

    word2id['trg'] = {w: i for i, w in enumerate(trg_vocab)}
    spm_trg = spm.SentencePieceProcessor()
    spm_trg.Load(f'{args.preprocess_path}/m_trg_{args.trg_vocab_size}.model')

    input_id_list = tuple(
        [args.bos_id] + spm_trg.encode(
                            text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [args.eos_id] for text in train_trg_sequences
    )
    processed_trg['train']['input_ids'] = [np.array(input_, dtype=np.int32) for input_ in input_id_list]
    input_id_list = tuple(
        [args.bos_id] + spm_trg.encode(text, out_type=int) + [args.eos_id] for text in valid_trg_sequences
    )
    processed_trg['valid']['input_ids'] = [np.array(input_, dtype=np.int32) for input_ in input_id_list]
    input_id_list = tuple(
        [args.bos_id] + spm_trg.encode(text, out_type=int) + [args.eos_id] for text in test_trg_sequences
    )
    processed_trg['test']['input_ids'] = [np.array(input_, dtype=np.int32) for input_ in input_id_list]

    return processed_src, processed_trg, word2id