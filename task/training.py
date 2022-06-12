# Import modules
import os
import gc
import h5py
import pickle
import logging
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.dataset import CustomDataset
from model.custom_transformer.transformer import Transformer
from model.custom_plm.T5 import custom_T5
from model.custom_plm.bart import custom_Bart
from model.optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log

def label_smoothing_loss(pred, gold, trg_pad_idx, smoothing_eps=0.1):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - smoothing_eps) + (1 - one_hot) * smoothing_eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    non_pad_mask = gold.ne(trg_pad_idx)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).mean()
    return loss

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start training!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
    if args.tokenizer == 'spm':
        save_path = os.path.join(save_path, f'{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}')

    with h5py.File(os.path.join(save_path, 'processed.hdf5'), 'r') as f:
        train_src_input_ids = f.get('train_src_input_ids')[:]
        train_src_attention_mask = f.get('train_src_attention_mask')[:]
        valid_src_input_ids = f.get('valid_src_input_ids')[:]
        valid_src_attention_mask = f.get('valid_src_attention_mask')[:]
        train_trg_input_ids = f.get('train_trg_input_ids')[:]
        train_trg_attention_mask = f.get('train_trg_attention_mask')[:]
        valid_trg_input_ids = f.get('valid_trg_input_ids')[:]
        valid_trg_attention_mask = f.get('valid_trg_attention_mask')[:]

    with open(os.path.join(save_path, 'word2id.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_word2id = data_['src_word2id']
        src_vocab_num = len(src_word2id)
        trg_word2id = data_['trg_word2id']
        trg_vocab_num = len(trg_word2id)
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    if args.model_type == 'custom_transformer':
        model = Transformer(src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                            pad_idx=args.pad_id, bos_idx=args.bos_id, eos_idx=args.eos_id,
                            d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                            dim_feedforward=args.dim_feedforward,
                            num_common_layer=args.num_common_layer, num_encoder_layer=args.num_encoder_layer,
                            num_decoder_layer=args.num_decoder_layer,
                            src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                            dropout=args.dropout, embedding_dropout=args.embedding_dropout,
                            trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
                            emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing, 
                            variational_mode=args.variational_mode, z_var=args.z_var,
                            parallel=args.parallel)
        tgt_subsqeunt_mask = model.generate_square_subsequent_mask(args.trg_max_len - 1, device)
    elif args.model_type == 'T5':
        model = custom_T5(isPreTrain=args.isPreTrain, d_latent=args.d_latent, 
                          variational_mode=args.variational_mode, z_var=args.z_var,
                          decoder_full_model=True, device=device)
        tgt_subsqeunt_mask = None
    elif args.model_type == 'bart':
        model = custom_Bart(isPreTrain=args.isPreTrain, PreTrainMode='large',
                            variational_mode=args.variational_mode, z_var=args.z_var,
                            d_latent=args.d_latent, emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing)
        tgt_subsqeunt_mask = None
    model = model.to(device)

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                               trg_list=train_trg_input_ids, trg_att_list=train_trg_attention_mask,
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                               pad_idx=model.pad_idx, eos_idx=model.eos_idx),
        'valid': CustomDataset(src_list=valid_src_input_ids, src_att_list=valid_src_attention_mask,
                               trg_list=valid_trg_input_ids, trg_att_list=valid_trg_attention_mask,
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                               pad_idx=model.pad_idx, eos_idx=model.eos_idx),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")
    
    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_path = os.path.join(args.model_save_path, args.task, args.data_name, args.tokenizer)
        save_file_name = os.path.join(save_path, 
                                        f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
        checkpoint = torch.load(save_file_name)
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_loss = 1e+100

    write_log(logger, 'Traing start!')

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                write_log(logger, 'Validation start...')
                val_loss = 0
                val_acc = 0
                model.eval()
            for i, batch_iter in enumerate(tqdm(dataloader_dict[phase], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

                # Optimizer setting
                optimizer.zero_grad(set_to_none=True)

                # Input, output setting
                src_sequence = batch_iter[0]
                src_att = batch_iter[1]
                trg_sequence = batch_iter[2]
                trg_att = batch_iter[3]

                src_sequence = src_sequence.to(device, non_blocking=True)
                src_att = src_att.to(device, non_blocking=True)
                trg_sequence = trg_sequence.to(device, non_blocking=True)
                trg_att = trg_att.to(device, non_blocking=True)

                # Output pre-processing
                trg_sequence_gold = trg_sequence[:, 1:]
                non_pad = trg_sequence_gold != model.pad_idx
                trg_sequence_gold = trg_sequence_gold[non_pad].contiguous().view(-1)

                # Train
                if phase == 'train':
                    with autocast():
                        predicted, dist_loss = model(src_input_ids=src_sequence, src_attention_mask=src_att,
                                                     trg_input_ids=trg_sequence, trg_attention_mask=trg_att,
                                                     non_pad_position=non_pad, tgt_subsqeunt_mask=tgt_subsqeunt_mask)
                        predicted = predicted.view(-1, predicted.size(-1))
                        sum_loss = label_smoothing_loss(predicted, trg_sequence_gold, 
                                                        trg_pad_idx=model.pad_idx,
                                                        smoothing_eps=args.label_smoothing_eps)
                        total_loss = sum_loss + dist_loss

                    scaler.scale(total_loss).backward()
                    if args.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    if args.scheduler in ['constant', 'warmup']:
                        scheduler.step()
                    if args.scheduler == 'reduce_train':
                        scheduler.step(total_loss)

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                        acc = (predicted.max(dim=1)[1] == trg_sequence_gold).sum() / len(trg_sequence_gold)
                        iter_log = "[Epoch:%03d][%03d/%03d] train_seq_loss:%03.2f | train_latent_loss:%03.2f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                            (epoch, i, len(dataloader_dict['train']), 
                            nmt_loss.item(), dist_loss.item(), acc*100, optimizer.param_groups[0]['lr'], 
                            (time() - start_time_e) / 60)
                        write_log(logger, iter_log)
                        freq = 0
                    freq += 1

                # Validation
                if phase == 'valid':
                    with torch.no_grad():
                        predicted, dist_loss = model(src_input_ids=src_sequence, src_attention_mask=src_att,
                                                     trg_input_ids=trg_sequence, trg_attention_mask=trg_att,
                                                     non_pad_position=non_pad, tgt_subsqeunt_mask=tgt_subsqeunt_mask)
                        sum_loss = F.cross_entropy(predicted, trg_sequence_gold, ignore_index=model.pad_idx)
                        total_loss = sum_loss + dist_loss
                    val_loss += total_loss.item()
                    val_acc += (predicted.max(dim=1)[1] == trg_sequence_gold).sum() / len(trg_sequence_gold)

            if phase == 'valid':

                if args.scheduler == 'reduce_valid':
                    scheduler.step(val_loss)
                if args.scheduler == 'lambda':
                    scheduler.step()

                val_loss /= len(dataloader_dict[phase])
                val_acc /= len(dataloader_dict[phase])
                write_log(logger, f'Variational_mode: {args.variational_mode} | Parallel: {args.parallel}')
                write_log(logger, 'Validation Loss: %3.3f' % val_loss)
                write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))
                save_path = os.path.join(args.model_save_path, args.data_name, args.tokenizer)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if args.tokenizer == 'spm':
                    save_file_name = os.path.join(save_path, 
                                                f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
                else:
                    save_file_name = os.path.join(save_path, 
                                                f'checkpoint_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
                                                
                if val_loss < best_val_loss:
                    write_log(logger, 'Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict()
                    }, save_file_name)
                    best_val_loss = val_loss
                    best_epoch = epoch
                else:
                    else_log = f'Still {best_epoch} epoch loss({round(best_val_loss.item(), 4)}) is better...'
                    write_log(logger, else_log)

    # 3) Print results
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_loss.item(), 4)}')