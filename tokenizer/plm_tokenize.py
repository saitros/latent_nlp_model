import argparse
from transformers import  BertTokenizer, BartTokenizer, T5Tokenizer

def plm_tokenizing(sequence_dict: dict, args: argparse.Namespace, 
                    domain: str = 'src', language: str = 'en'):

    # 1) Pre-setting
    processed_sequences = dict()
    processed_sequences['train'] = dict()
    processed_sequences['valid'] = dict()
    processed_sequences['test'] = dict()

    if domain == 'src':
        max_len = args.src_max_len
    if domain == 'trg':
        max_len = args.trg_max_len

    if args.tokenizer == 'bert':
        if language == 'en':
            tokenizer =  BertTokenizer.from_pretrained('bert-base-cased')
        elif language == 'kr':
            tokenizer =  BertTokenizer.from_pretrained('beomi/kcbert-base')
    elif args.tokenizer == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    elif args.tokenizer == 'T5':
        if language == 'en':
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
        elif language == 'kr':
            tokenizer = T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-base')

    for phase in ['train', 'valid', 'test']:
        encoded_dict = \
        tokenizer(
            sequence_dict[phase],
            max_length=max_len,
            padding='max_length',
            truncation=True
        )
        processed_sequences[phase]['input_ids'] = encoded_dict['input_ids']
        processed_sequences[phase]['attention_mask'] = encoded_dict['attention_mask']
    # BART는 model.config.decoder_start_token_id로 시작해야함 => 추후 수정 필요
    
    word2id = tokenizer.get_vocab()

    return processed_sequences, word2id