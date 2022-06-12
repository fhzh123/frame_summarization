import argparse
from transformers import  BertTokenizer, BartTokenizer, T5Tokenizer

def plm_tokenizing(sequence_dict: dict, args: argparse.Namespace, domain: str = 'src'):

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
        tokenizer =  BertTokenizer.from_pretrained('bert-base-cased')
    elif args.tokenizer == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    elif args.tokenizer == 'T5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")

    for phase in ['train', 'valid', 'test']:
        if phase == 'train':
            encoded_dict = \
            tokenizer(
                sequence_dict[phase],
                padding='longest',
            )
        else:
            train_max_len = max([len(x) for x in encoded_dict['input_ids']])
            encoded_dict = \
            tokenizer(
                sequence_dict[phase],
                max_length=train_max_len, 
                truncation=True,
                padding='max_length'
            )
        processed_sequences[phase]['input_ids'] = encoded_dict['input_ids']
        processed_sequences[phase]['attention_mask'] = encoded_dict['attention_mask']

    # BART's decoder input id need to start with 'model.config.decoder_start_token_id'
    if args.tokenizer == 'bart' and domain == 'trg':
        for i in range(len(processed_sequences[phase]['input_ids'])):
            processed_sequences[phase]['input_ids'][i][0] = 2
    
    word2id = tokenizer.get_vocab()

    return processed_sequences, word2id