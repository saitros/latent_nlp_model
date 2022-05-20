def preprocessing(args):

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.tokenizer == 'spm':
        processed_src, word2id_src = spm_tokenizing(src_sequences, args)
        processed_trg, word2id_trg = spm_tokenizing(src_sequences, args)
    elif args.tokenizer == 'spacy':
        processed_src, processed_trg, word2id = spacy_tokenizing(src_sequences, trg_sequences, args)
    else:
        processed_src, processed_trg, word2id = plm_tokenizeing(src_sequences, trg_sequences, args)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.task, args.tokenizer)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.tokenizer == 'spm':
        save_name = f'processed_{args.data_name}_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed_{args.data_name}_{args.tokenizer}.hdf5'

    with h5py.File(os.path.join(save_path, save_name), 'w') as f:
        f.create_dataset('train_src_input_ids', data=processed_src['train'])
        f.create_dataset('train_trg_input_ids', data=processed_trg['train'])
        f.create_dataset('valid_src_input_ids', data=processed_src['valid'])
        f.create_dataset('valid_trg_input_ids', data=processed_trg['valid'])

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'w') as f:
        f.create_dataset('test_src_input_ids', data=processed_src['test'])
        f.create_dataset('test_trg_input_ids', data=processed_trg['test'])

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'wb') as f:
        pickle.dump({
            'src_word2id': word2id['src'],
            'trg_word2id': word2id['trg']
        }, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')