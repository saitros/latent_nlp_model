import argparse
from transformers import  BertTokenizer, BartTokenizer, T5Tokenizer

def plm_tokenizeing(sequence_dict: dict, args: argparse.Namespace, domain: str = 'src'):

    # 1) Pre-setting
    processed_sequences = dict()
    processed_sequences['train'] = dict()
    processed_sequences['valid'] = dict()
    processed_sequences['test'] = dict()

    if domain == 'src':
        vocab_size = args.src_vocab_size
        max_len = args.src_max_len
    if domain == 'trg':
        vocab_size = args.trg_vocab_size
        max_len = args.trg_max_len

    if args.tokenizer == 'bert':
        tokenizer =  BertTokenizer.from_pretrained('bert-base-cased')
    elif args.tokenizer == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    elif args.tokenizer == 'T5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")

    for phase in ['train', 'valid', 'test']:
        encoded_dict = \
        tokenizer(
            src_sequences[phase],
            max_length=max_len,
            padding='max_length',
            truncation=True
        )
        processed_sequences[phase]['input_ids'] = encoded_dict['input_ids']
        processed_sequences[phase]['attention_mask'] = encoded_dict['attention_mask']
    # BART는 model.config.decoder_start_token_id로 시작해야함 => 추후 수정 필요
    
    word2id = tokenizer.get_vocab()

    return processed_sequences, word2id