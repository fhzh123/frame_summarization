# Import modules
import os
import gc
import psutil
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score
# Import custom modules
from task.frame.dataset import VOX_CustomDataset
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name

from transformers import BertForSequenceClassification, RobertaForSequenceClassification, T5Model, BartForSequenceClassification

def data_split_index(seq):

    paired_data_len = len(seq)
    valid_num = int(paired_data_len * 0.05)
    test_num = int(paired_data_len * 0.03)

    valid_index = np.random.choice(paired_data_len, valid_num, replace=False)
    train_index = list(set(range(paired_data_len)) - set(valid_index))
    test_index = np.random.choice(train_index, test_num, replace=False)
    train_index = list(set(train_index) - set(test_index))

    return train_index, valid_index, test_index

def get_culture(d):
    if 'Mad Men' in d or 'Game of Thrones' in d or 'True Detective' in d or 'Westworld' in d or 'Hannibal' in d or 'Fear the Walking Dead' in d:
        return 'Culture'
    else:
        return d

def frame_detection(args):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.tensorboard_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    write_log(logger, 'Start training!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    # Data Load
    dat = pd.read_csv('./dsjVoxArticles.tsv', sep='\t')
    dat = dat.dropna()

    # Preprocessing category
    # 1) Sports
    dat['category'] = dat['category'].replace('2016 Rio Olympics', 'Sports')
    dat['category'] = dat['category'].replace('College Football', 'Sports')
    dat['category'] = dat['category'].replace('NFL', 'Sports')
    dat['category'] = dat['category'].replace('2016 Golden Globes', 'Sports')

    # 2) Politics
    dat['category'] = dat['category'].replace('Voting Rights', 'Politics')
    dat['category'] = dat['category'].replace('Hillary Clinton', 'Politics')
    dat['category'] = dat['category'].replace('Donald Trump', 'Politics')
    dat['category'] = dat['category'].replace('Joe Biden', 'Politics')
    dat['category'] = dat['category'].replace('Politics & Policy', 'Politics')

    # 3) Culture
    dat['category'] = dat['category'].replace('Music', 'Culture')
    dat['category'] = dat['category'].replace('Movies', 'Culture')
    dat['category'] = dat['category'].replace('Television', 'Culture')
    dat['category'] = dat['category'].replace('Hollywood', 'Culture')
    dat['category'] = dat['category'].replace('Television', 'Culture')
    dat['category'] = dat['category'].replace('Game of Thrones', 'Culture')
    dat['category'] = dat['category'].replace('2016 Grammys', 'Culture')
    dat['category'] = dat['category'].apply(get_culture)

    # 4) Health
    dat['category'] = dat['category'].replace('Ebola', 'Health')
    dat['category'] = dat['category'].replace('Health Care', 'Health')
    dat['category'] = dat['category'].replace('Science & Health', 'Health')
    dat['category'] = dat['category'].replace('Reproductive Health', 'Health')

    # 5) Business & Finance
    dat['category'] = dat['category'].replace('Labor Market', 'Business & Finance')
    dat['category'] = dat['category'].replace('Campaign Finance', 'Business & Finance')
    dat['category'] = dat['category'].replace('Small Business', 'Business & Finance')

    # Data Processing
    processed_dat = pd.DataFrame()

    for cat in ['Business & Finance', 'Health', 'Sports', 'Politics', 'Culture']:
        pre_dat = dat[dat['category'] == cat]
        processed_dat = pd.concat((processed_dat, pre_dat))

    # Data Labeling
    processed_dat['category'] = processed_dat['category'].replace('Business & Finance', 0)
    processed_dat['category'] = processed_dat['category'].replace('Health', 1)
    processed_dat['category'] = processed_dat['category'].replace('Sports', 2)
    processed_dat['category'] = processed_dat['category'].replace('Politics', 3)
    processed_dat['category'] = processed_dat['category'].replace('Culture', 4)

    # Index Reset
    processed_dat = processed_dat.reset_index()

    src_list = processed_dat['title']
    trg_list = processed_dat['category']

    train_index, valid_index, test_index = data_split_index(src_list)

    train_src_list = [src_list[i] for i in train_index]
    valid_src_list = [src_list[i] for i in valid_index]
    test_src_list = [src_list[i] for i in test_index]

    train_trg_list = [trg_list[i] for i in train_index]
    valid_trg_list = [trg_list[i] for i in valid_index]
    test_trg_list = [trg_list[i] for i in test_index]

    gc.enable()
    write_log(logger, "Finished loading data!")

    # 2) Dataloader setting
    dataset_dict = {
        'train': VOX_CustomDataset(src_list=train_src_list, trg_list=train_trg_list, 
                                   max_len=args.src_max_len, tokenizer=args.tokenizer),
        'valid': VOX_CustomDataset(src_list=valid_src_list, trg_list=valid_trg_list, 
                                   max_len=args.src_max_len, tokenizer=args.tokenizer)
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

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    if args.model_type == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)
    elif args.model_type == 'bart':
        model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=5)
    elif args.model_type == 'T5':
        model = T5Model.from_pretrained('facebook/bart-base', num_labels=5)
    elif args.model_type == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
    model = model.to(device)
    
    # 2) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    f1_score = F1Score(num_classes=5).to(device)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        checkpoint = torch.load(os.path.join(args.model_save_path, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

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
                trg_label = batch_iter[2]

                src_sequence = src_sequence.to(device, non_blocking=True)
                src_att = src_att.to(device, non_blocking=True)
                trg_label = trg_label.to(device, non_blocking=True)

                # Train
                if phase == 'train':

                    with autocast():
                        predicted = model(input_ids=src_sequence, attention_mask=src_att)
                        out = predicted.logits
                        loss = criterion(out, trg_label)
                        # total_loss = loss + dist_loss
                        total_loss = loss

                    scaler.scale(total_loss).backward()
                    if args.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    if args.scheduler in ['constant', 'warmup']:
                        scheduler.step()
                    if args.scheduler == 'reduce_train':
                        scheduler.step(loss)

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                        acc = (out.max(dim=1)[1] == trg_label).sum() / len(trg_label)
                        iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.3f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                            (epoch, i, len(dataloader_dict['train']), 
                            total_loss.item(), acc*100, optimizer.param_groups[0]['lr'], 
                            (time() - start_time_e) / 60)
                        write_log(logger, iter_log)
                        freq = 0
                    freq += 1

                    if args.use_tensorboard:
                        acc = (out.max(dim=1)[1] == trg_label).sum() / len(trg_label)
                        
                        writer.add_scalar('TRAIN/Loss', total_loss.item(), (epoch-1) * len(dataloader_dict['train']) + i)
                        writer.add_scalar('TRAIN/Accuracy', acc*100, (epoch-1) * len(dataloader_dict['train']) + i)
                        writer.add_scalar('USAGE/CPU_Usage', psutil.cpu_percent(), (epoch-1) * len(dataloader_dict['train']) + i)
                        writer.add_scalar('USAGE/RAM_Usage', psutil.virtual_memory().percent, (epoch-1) * len(dataloader_dict['train']) + i)
                        writer.add_scalar('USAGE/GPU_Usage', torch.cuda.memory_allocated(device=device), (epoch-1) * len(dataloader_dict['train']) + i) # MB Size

                # Validation
                if phase == 'valid':
                    with torch.no_grad():
                        predicted = model(input_ids=src_sequence, attention_mask=src_att)
                        out = predicted.logits
                        loss = criterion(out, trg_label)
                        f1 = f1_score(out.max(dim=1)[1], trg_label)
                    val_loss += total_loss.item()
                    val_acc += (out.max(dim=1)[1] == trg_label).sum() / len(trg_label)

            if phase == 'valid':

                if args.scheduler == 'reduce_valid':
                    scheduler.step(val_loss)
                if args.scheduler == 'lambda':
                    scheduler.step()

                val_loss /= len(dataloader_dict[phase])
                val_acc /= len(dataloader_dict[phase])
                write_log(logger, 'Validation Loss: %3.3f' % val_loss)
                write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))
                save_path = os.path.join(args.model_save_path, args.tokenizer)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_file_name = os.path.join(save_path, 
                                              f'checkpoint_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
                if val_acc > best_val_acc:
                    write_log(logger, 'Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict()
                    }, save_file_name)
                    best_val_acc = val_acc
                    best_epoch = epoch
                else:
                    else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc.item()*100, 2)})% is better...'
                    write_log(logger, else_log)
                
                if args.use_tensorboard:
                    writer.add_scalar('VALID/Loss', val_loss, epoch)
                    writer.add_scalar('VALID/Accuracy', val_acc * 100, epoch)

    # 3) Print results
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_acc.item(), 2)}')