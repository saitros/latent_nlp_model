from transformers import  BertTokenizer, BartTokenizer, T5Tokenizer

def plm_tokenizeing(src_sequences, trg_sequences, args):
    processed_src, processed_trg, word2id = dict(), dict(), dict()

    if args.tokenizer == 'bert':
        tokenizer =  BertTokenizer.from_pretrained('bert-base-cased')
    elif args.tokenizer == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    elif args.tokenizer == 'T5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # 1) Source lanugage
    for phase in ['train', 'valid', 'test']:
        processed_src[phase] = \
        tokenizer(
            src_sequences[phase],
            max_length=args.src_max_len,
            padding='max_length',
            truncation=True
        )

    # 이 부분 수정 필요 => BART는 model.config.decoder_start_token_id로 시작해야함
    for phase in ['train', 'valid', 'test']:
        processed_trg[phase] = \
        tokenizer(
            trg_sequences[phase],
            max_length=args.trg_max_len,
            padding='max_length',
            truncation=True
        )
    
    word2id['src'] = tokenizer.get_vocab()
    word2id['trg'] = tokenizer.get_vocab()

    return processed_src, processed_trg, word2id