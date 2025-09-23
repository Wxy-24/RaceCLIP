import os
import socket
import subprocess
import numpy as np
import torch
import shutil
import torchvision
import argparse
import datetime
import time
import pickle
import torch.nn.functional as F
from torch import nn

torch.autograd.set_detect_anomaly(True)
import transformers
from transformers import AutoTokenizer

import sys
sys.path.append('/gpfsdswork/projects/rech/dvj/uyk23wk/ConVIRT/clip/lib/python3.6/site-packages/clip')
from clip import *

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# ConVIRT
import clip
from convirt.modules.transformations import TransformsConVIRT
from convirt.modules.sync_batchnorm import convert_model
from convirt.modules.dataloader import CLRDataset,MTDataset
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from model import load_optimizer
from utils import yaml_config_hook


def save_model(args, model, optimizer, best=False):
    if best:
        out = os.path.join(args.model_path, "best_checkpoint_{}.pth".format(args.current_epoch))
    else:
        out = os.path.join(args.model_path, "checkpoint_{}.pth".format(args.current_epoch))
    torch.save(model, out)

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad != None:
            p.grad.data = p.grad.data.float()
def convert_models_to_mix(model):
    clip.model.convert_weights(model)

def sim_cos(vec1,vec2):
    vec1=np.array(vec1)
    vec2=np.array(vec2)
    return vec1.dot(vec2)/np.linalg.norm(vec1)/np.linalg.norm(vec2)


def train(args, train_loader, model, tokenizer, optimizer,writer):
    loss_epoch = 0
    for step, (images, texts, aug1,aug2,aug3,aug4) in enumerate(train_loader):
        optimizer.zero_grad()
        x_v = images.to(args.device)
        v=model.visual(x_v)
        labels = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
        loss_CE = torch.nn.CrossEntropyLoss()

        if 'bert' in args.resnet:
            with torch.no_grad():
                x_u=torch.tensor(tokenizer.batch_encode_plus(texts, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                x_aug1=torch.tensor(tokenizer.batch_encode_plus(aug1, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                x_aug2=torch.tensor(tokenizer.batch_encode_plus(aug2, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                x_aug3=torch.tensor(tokenizer.batch_encode_plus(aug3, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                x_aug4=torch.tensor(tokenizer.batch_encode_plus(aug4, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                u=model.transformer(x_u)
                u1=model.transformer(x_aug1)
                u2=model.transformer(x_aug2)
                u3=model.transformer(x_aug3)
                u4=model.transformer(x_aug4)

        else:
            x_u = tokenizer(texts, truncate=True).to(args.device)
            x_aug1 = tokenizer(aug1, truncate=True).to(args.device)
            x_aug2 = tokenizer(aug2, truncate=True).to(args.device)
            x_aug3 = tokenizer(aug3, truncate=True).to(args.device)
            x_aug4 = tokenizer(aug4, truncate=True).to(args.device)
            u=model.encode_text(x_u)
            u1=model.encode_text(x_aug1)
            u2=model.encode_text(x_aug2)
            u3=model.encode_text(x_aug3)
            u4=model.encode_text(x_aug4)        

        # normalized features
        image_features = v / v.norm(dim=1, keepdim=True)
        text_features = u / u.norm(dim=1, keepdim=True)
        aug1_features = u1 / u1.norm(dim=1, keepdim=True)
        aug2_features = u2 / u2.norm(dim=1, keepdim=True)
        aug3_features = u3 / u3.norm(dim=1, keepdim=True)
        aug4_features = u4 / u4.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        logits_per_image_aug1 = logit_scale * image_features @ aug1_features.t()
        logits_per_aug1 = logits_per_image_aug1.t()
        logits_per_image_aug2 = logit_scale * image_features @ aug2_features.t()
        logits_per_aug2 = logits_per_image_aug2.t()
        logits_per_image_aug3 = logit_scale * image_features @ aug3_features.t()
        logits_per_aug3 = logits_per_image_aug3.t()
        logits_per_image_aug4 = logit_scale * image_features @ aug4_features.t()
        logits_per_aug4 = logits_per_image_aug4.t()
    
        loss_v = loss_CE(logits_per_image,labels)+loss_CE(logits_per_image_aug1,labels)+loss_CE(logits_per_image_aug2,labels)+loss_CE(logits_per_image_aug3,labels)+loss_CE(logits_per_image_aug4,labels)
        loss_t = loss_CE(logits_per_text,labels)+loss_CE(logits_per_aug1,labels)+loss_CE(logits_per_aug2,labels)+loss_CE(logits_per_aug3,labels)+loss_CE(logits_per_aug4,labels)
        loss=loss_v+loss_t

        loss.backward()
        convert_models_to_fp32(model)
        optimizer.step()
        # convert_models_to_mix(model)

        if args.nr == 0 and step % 1000 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def validate(args, val_loader, model, tokenizer, optimizer,writer):
    with torch.no_grad():
        model.eval()
        loss_epoch = 0
        for step, (x_v, x_u,aug1,aug2,aug3,aug4) in enumerate(val_loader):
            x_v = x_v.to(args.device)
            v=model.visual(x_v)
            labels = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
            loss_CE = torch.nn.CrossEntropyLoss()

            if 'bert' in args.resnet:
                with torch.no_grad():
                    x_u=torch.tensor(tokenizer.batch_encode_plus(x_u, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                    x_aug1=torch.tensor(tokenizer.batch_encode_plus(aug1, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                    x_aug2=torch.tensor(tokenizer.batch_encode_plus(aug2, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                    x_aug3=torch.tensor(tokenizer.batch_encode_plus(aug3, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                    x_aug4=torch.tensor(tokenizer.batch_encode_plus(aug4, padding='max_length',max_length=512, truncation=True, add_special_tokens=True)['input_ids']).cuda()
                    u=model.transformer(x_u)
                    u1=model.transformer(x_aug1)
                    u2=model.transformer(x_aug2)
                    u3=model.transformer(x_aug3)
                    u4=model.transformer(x_aug4)

            else:
                x_u = tokenizer(x_u, truncate=True).to(args.device)
                x_aug1 = tokenizer(aug1, truncate=True).to(args.device)
                x_aug2 = tokenizer(aug2, truncate=True).to(args.device)
                x_aug3 = tokenizer(aug3, truncate=True).to(args.device)
                x_aug4 = tokenizer(aug4, truncate=True).to(args.device)
                u=model.encode_text(x_u)
                u1=model.encode_text(x_aug1)
                u2=model.encode_text(x_aug2)
                u3=model.encode_text(x_aug3)
                u4=model.encode_text(x_aug4)  
        

            # normalized features
            image_features = v / v.norm(dim=1, keepdim=True)
            text_features = u / u.norm(dim=1, keepdim=True)
            aug1_features = u1 / u1.norm(dim=1, keepdim=True)
            aug2_features = u2 / u2.norm(dim=1, keepdim=True)
            aug3_features = u3 / u3.norm(dim=1, keepdim=True)
            aug4_features = u4 / u4.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            logits_per_image_aug1 = logit_scale * image_features @ aug1_features.t()
            logits_per_aug1 = logits_per_image_aug1.t()
            logits_per_image_aug2 = logit_scale * image_features @ aug2_features.t()
            logits_per_aug2 = logits_per_image_aug2.t()
            logits_per_image_aug3 = logit_scale * image_features @ aug3_features.t()
            logits_per_aug3 = logits_per_image_aug3.t()
            logits_per_image_aug4 = logit_scale * image_features @ aug4_features.t()
            logits_per_aug4 = logits_per_image_aug4.t()


            loss_v = loss_CE(logits_per_image,labels)+loss_CE(logits_per_image_aug1,labels)+loss_CE(logits_per_image_aug2,labels)+loss_CE(logits_per_image_aug3,labels)+loss_CE(logits_per_image_aug4,labels)
            loss_t = loss_CE(logits_per_text,labels)+loss_CE(logits_per_aug1,labels)+loss_CE(logits_per_aug2,labels)+loss_CE(logits_per_aug3,labels)+loss_CE(logits_per_aug4,labels)
            loss=loss_v+loss_t


            loss_epoch += loss.item()

    model.train()
    return loss_epoch


def main(gpu, args):
    # number of nodes / node ID
    job_name = os.environ['SLURM_JOB_NAME']
    job_id = os.environ['SLURM_JOB_ID']
    log_filename = job_name + job_id + '.loss'


    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])

    # local rank on the current node / global rank
    local_rank = int(os.environ['SLURM_LOCALID'])
    global_rank = int(os.environ['SLURM_PROCID'])

    # number of processes / GPUs per node
    world_size = int(os.environ['SLURM_NTASKS'])
    n_gpu_per_node = world_size // n_nodes

    # define master address and master port
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    master_addr = hostnames.split()[0].decode('utf-8')

    # set environment variables for 'env://'
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(29500)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)

    # define whether this is the master process / if we are in distributed mode
    is_master = node_id == 0 and local_rank == 0
    multi_node = n_nodes > 1
    multi_gpu = world_size > 1

    # summary
    PREFIX = "%i - " % global_rank
    print(PREFIX + "Number of nodes: %i" % n_nodes)
    print(PREFIX + "Node ID        : %i" % node_id)
    print(PREFIX + "Local rank     : %i" % local_rank)
    print(PREFIX + "Global rank    : %i" % global_rank)
    print(PREFIX + "World size     : %i" % world_size)
    print(PREFIX + "GPUs per node  : %i" % n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(is_master))
    print(PREFIX + "Multi-node     : %s" % str(multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(multi_gpu))
    print(PREFIX + "Hostname       : %s\n" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('torch.cuda.is_available:',torch.cuda.is_available())


    # initialize model
    if is_master:
        print("Initializing model... ", end="", flush=True)
    model, preprocess = clip.load(args.resnet.split("@")[-1], device=args.device, jit=False)
    model = model.float()
    tokenizer = tokenize
 
    if 'bert' in args.resnet:
        print('replace GPT-2 with PubMedCLIP')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        model.transformer= torch.load(args.bert)

    model.to(args.device)

    train_fonction = train
    validate_fonction = validate


    if is_master:
        print("Image encoder:", args.resnet,
            "\tPretrained:", args.pretrain)
        print("Text encoder:", args.bert.split("/")[-1],
            "\tFreezed layers:", args.freeze_layers, "\n")
        print("Tokenizer:",tokenizer)


    # optimizer / loss
    print("Loss:", args.criterion, "\t")
    optimizer, scheduler = load_optimizer(args, model, lr=float(args.lr))

   ### MOVED ###

    if is_master:
        print("Loading dataset...", end="", flush=True)

    if "clip" in args.resnet or "RN50" in args.resnet:
        transform = preprocess

    train_dataset = MTDataset(csv_file=args.csv_file,
                               root_dir=args.root_dir,
                               transform=transform,
                               clip = ("clip" in args.resnet or "RN50" in args.resnet)
                               )
    print('training set:',args.csv_file,'len(dataset):',train_dataset.__len__())

    val_dataset = MTDataset(csv_file=args.val_csv_file,
                             root_dir=args.val_root_dir,
                             transform=transform,
                             clip = ("clip" in args.resnet or "RN50" in args.resnet)
                             )
    print('validation set:',args.val_csv_file,'len(csv):',val_dataset.__len__())

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers
    )

    if is_master:
        print("[DONE]\n")

    # print(list(train_loader)[0])

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    if is_master:
        print("STARTING TRAINING")
        print('Start Time =', datetime.datetime.now().strftime("%H:%M:%S"), '\n')

    t0 = time.time()
    args.global_step = 0
    args.current_epoch = 0
    best_val_loss = np.inf


    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if args.gs==0:
            lr = optimizer.param_groups[0]["lr"]
        else:        
            lr = optimizer._optim.param_groups[0]["lr"]      #  Gradient Surgery
        loss_epoch = train_fonction(args, train_loader, model, tokenizer, optimizer,writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0 and is_master:
            save_model(args, model, optimizer)

        if args.nr == 0 and is_master:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            val_loss = validate_fonction(args, val_loader, model, tokenizer, optimizer,writer)
            if val_loss < best_val_loss:
                out = os.path.join(args.model_path, "ViT_best_epoch_{}_{}.tar".format(args.current_epoch, val_loss / len(val_loader)))
                torch.save(model.visual, out)
                save_model(args, model, optimizer, best=True)
                best_val_loss = val_loss
            else:
                save_model(args, model, optimizer, best=False)

            epoch_counter = epoch - args.start_epoch
            elapsed = time.time() - t0
            epoch_time = elapsed/(epoch_counter+1)
            remaining = (args.epochs - (epoch_counter+1))*epoch_time
            remaining = str(datetime.timedelta(seconds=round(remaining)))
            elapsed = str(datetime.timedelta(seconds=round(elapsed)))
            print(f'Epoch {epoch_counter+1}/{args.epochs} [{elapsed}<{remaining}, {round(epoch_time, 2)}s/epoch] {round((epoch_counter+1)/args.epochs*100, 1)}% loss: {loss_epoch / len(train_loader)}\t val_loss: {val_loss / len(val_loader)} lr: {lr}')

            with open(os.path.join(args.log_loss_dir,log_filename), 'a') as f:
                f.write(str(loss_epoch / len(train_loader)) + ',' + str(val_loss / len(val_loader)) + '\n')
            args.current_epoch += 1


    # end training
    if is_master:
        save_model(args, model.model.visual, optimizer)
    writer.close()


if __name__ == "__main__":
    t1=time.time()
    parser = argparse.ArgumentParser(description="ConVIRT")
    config = yaml_config_hook("./config/config_CLIP_ViT16_MT.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    shutil.copy('./config/config_CLIP_ViT16_MT.yaml',os.path.join(args.model_path, 'config.yaml'))

    print("args.model_path",args.model_path)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", args.device)

    main(0, args)
