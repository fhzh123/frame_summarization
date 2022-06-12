import os
import h5py
import time
import pickle
import logging
import argparse
import pandas as pd
from model.tokenizer.spm_tokenize import spm_tokenizing
from model.tokenizer.plm_tokenize import plm_tokenizing
from utils import TqdmLoggingHandler, write_log

def preprocessing(args: argparse.Namespace):

    start_time = time.time()

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
    #=============Data Load=============#
    #===================================#

    write_log(logger, 'Start preprocessing!')

    src_list, trg_list = dict(), dict()

    args.data_path = os.path.join(args.data_path, 'cnn_dailymail', args.cnn_dailymail_ver)

    train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    valid = pd.read_csv(os.path.join(args.data_path, 'valid.csv'))
    test = pd.read_csv(os.path.join(args.data_path, 'test.csv'))

    src_list['train'] = train['article'].tolist()
    src_list['valid'] = valid['article'].tolist()
    src_list['test'] = test['article'].tolist()

    trg_list['train'] = train['summary'].tolist()
    trg_list['valid'] = valid['summary'].tolist()
    trg_list['test'] = test['summary'].tolist()

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.tokenizer == 'spm':
        processed_src, word2id_src = spm_tokenizing(src_list, args, domain='src')
        processed_trg, word2id_trg = spm_tokenizing(trg_list, args, domain='trg', src_trg_identical=args.src_trg_identical)
    else:
        processed_src, word2id_src = plm_tokenizing(src_list, args, domain='src')
        processed_trg, word2id_trg = plm_tokenizing(trg_list, args, domain='trg')

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.tokenizer == 'spm':
        save_path = os.path.join(save_path, f'{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    with h5py.File(os.path.join(save_path, 'processed.hdf5'), 'w') as f:
        f.create_dataset('train_src_input_ids', data=processed_src['train']['input_ids'])
        f.create_dataset('train_src_attention_mask', data=processed_src['train']['attention_mask'])
        f.create_dataset('valid_src_input_ids', data=processed_src['valid']['input_ids'])
        f.create_dataset('valid_src_attention_mask', data=processed_src['valid']['attention_mask'])
        f.create_dataset('train_trg_input_ids', data=processed_trg['train']['input_ids'])
        f.create_dataset('train_trg_attention_mask', data=processed_trg['train']['attention_mask'])
        f.create_dataset('valid_trg_input_ids', data=processed_trg['valid']['input_ids'])
        f.create_dataset('valid_trg_attention_mask', data=processed_trg['valid']['attention_mask'])

    with h5py.File(os.path.join(save_path, 'test_processed.hdf5'), 'w') as f:
        f.create_dataset('test_src_input_ids', data=processed_src['test']['input_ids'])
        f.create_dataset('test_src_attention_mask', data=processed_src['test']['attention_mask'])
        f.create_dataset('test_trg_input_ids', data=processed_trg['test']['input_ids'])
        f.create_dataset('test_trg_attention_mask', data=processed_trg['test']['attention_mask'])

    # Word2id pickle file save
    word2id_dict = {
        'src_word2id': word2id_src,
        'trg_word2id': word2id_trg
    }
    
    with open(os.path.join(save_path, 'word2id.pkl'), 'wb') as f:
        pickle.dump(word2id_dict, f)