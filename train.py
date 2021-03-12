import numpy as np
import pandas as pd
from PIL import Image
import os
import pdb
import json
import tqdm
import math
import random
import argparse

import horovod.torch as hvd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from cutmix import CutMix, CutMixCrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from cosine_annealing_with_warmup import CosineAnnealingWarmupRestarts

class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index])).convert('RGB')

        if "train" in self.dir:
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:
            return self.transform(x), self.fnames[index]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_model(model_name, model, optimizer, scheduler=None):
    if scheduler is not None:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
    else:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

    torch.save(state, os.path.join(model_name + '.pth'))
    print('model saved')


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')

if __name__ == '__main__':
    random_seed = 7
    # torch reproducible randomness
    torch.manual_seed(random_seed)
    #torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--cuda", type=str2bool, default="TRUE")
    args.add_argument("--num_epochs", type=int, default=6)
    args.add_argument("--img_size", type=int, default=224)
    args.add_argument("--optim", type=str, default="adam")
    args.add_argument("--model_name", type=str, default="resnet34")
    args.add_argument("--ckpt_dir", type=str, default="./ckpt/")
    args.add_argument("--train_dir", type=str, default="./train/")
    args.add_argument("--test_dir", type=str, default="./test/")
    args.add_argument("--batch", type=int, default=512)
    args.add_argument("--valid_dataset_ratio", type=float, default=0.01)
    args.add_argument("--use_cutmix", type=str2bool, default="TRUE")
    args.add_argument("--use_cosine_annealing_with_warmup", type=str2bool, default="TRUE")

    config = args.parse_args()
    
    LR = config.lr
    CUDA = config.cuda
    EPOCHS = config.num_epochs
    IMG_SIZE = config.img_size
    TRAIN_DIR = config.train_dir
    TEST_DIR = config.test_dir
    BATCH = config.batch
    OPTIM = config.optim
    MODEL_NAME = config.model_name
    CKPT_DIR = config.ckpt_dir
    VALID_RATIO = config.valid_dataset_ratio
    USE_CUTMIX = config.use_cutmix
    USE_CA = config.use_cosine_annealing_with_warmup

    SAVE_PATH = os.path.join(CKPT_DIR,"cutmix_{}_CA_with_warmup_{}".format(USE_CUTMIX, USE_CA))
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    """ TRAIN """
    if hvd.rank() == 0: print("loading data...")
    with open(TRAIN_DIR + 'metadata.json', "r", encoding="ISO-8859-1") as file:
        train = json.load(file)

    train_img = pd.DataFrame(train['images'])
    train_ann = pd.DataFrame(train['annotations']).drop(columns='image_id')
    train_df = train_img.merge(train_ann, on='id')
    mean_h = train_img['height'].mean()
    mean_w = train_img['width'].mean()
    NUM_CL = len(train_df['category_id'].value_counts())

    if hvd.rank() == 0: print("image mean height {} , width {} classes {}".format(mean_h, mean_w, NUM_CL))

    X_Train, Y_Train = train_df['file_name'].values, train_df['category_id'].values
    # shuffle
    shuffle_list = list(zip(X_Train, Y_Train))
    random.shuffle(shuffle_list)
    X_Train, Y_Train = zip(*shuffle_list)
    # split valid and train set
    X_Valid = X_Train[:int(len(X_Train)*VALID_RATIO)]
    Y_Valid = Y_Train[:int(len(Y_Train)*VALID_RATIO)]
    X_Train = X_Train[int(len(X_Train)*VALID_RATIO):]
    Y_Train = Y_Train[int(len(Y_Train)*VALID_RATIO):]

    TrainTransform = transforms.Compose(
        [ transforms.Resize((IMG_SIZE, IMG_SIZE)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
    ValidTransform = transforms.Compose(
        [ transforms.Resize((IMG_SIZE, IMG_SIZE)),
          transforms.ToTensor(),
          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    if hvd.rank() == 0: print("define dataset and create dataloader")
    trainset = GetData(TRAIN_DIR, X_Train, Y_Train, TrainTransform)

    if USE_CUTMIX:
        if hvd.rank() == 0: print("use cutmix")
        trainset = CutMix(trainset, num_class=NUM_CL, num_mix=1, beta=1.0, prob=0.49)
        criterion = CutMixCrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset, num_replicas=hvd.size(), rank=hvd.rank())
    trainloader = DataLoader(trainset, batch_size=BATCH, num_workers=8, sampler=train_sampler)

    validset = GetData(TRAIN_DIR, X_Valid, Y_Valid, ValidTransform)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        validset, num_replicas=hvd.size(), rank=hvd.rank())
    validloader = DataLoader(validset, batch_size=BATCH, num_workers=8, sampler=valid_sampler)


    if hvd.rank() == 0: print("define model and optimizer, scheduler")
    if MODEL_NAME == "resnet34":
        model = torchvision.models.resnet34(pretrained=True)

    model.fc = nn.Linear(512, NUM_CL, bias=True)
    if CUDA:
        model = model.cuda()

    if OPTIM == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    if USE_CA:
        steps_per_epoch = int((len(X_Train)/BATCH)/hvd.size())
        if hvd.rank() == 0: print("use scheduler : steps_per_epoch {}".format(steps_per_epoch))
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=steps_per_epoch,
                                                  cycle_mult=1.0,
                                                  max_lr=LR,
                                                  min_lr=1e-6,
                                                  warmup_steps=int(steps_per_epoch*0.1),
                                                  gamma=0.5)
    else:
        scheduler = None
        
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)        
    if hvd.rank() == 0: print("train start")
    for epoch in range(EPOCHS):
        tr_loss = 0.0
        correct = 0
        total = 0
        model = model.train()
        disable_pbar = True
        if hvd.rank () == 0:
            disable_pbar = False
        pbar = tqdm.tqdm(trainloader, disable=disable_pbar)

        for i, (images, labels) in enumerate(pbar):
            images = images.cuda()
            labels = labels.cuda()
            logits = model(images.float())
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(logits.data, 1)
            if USE_CUTMIX:
                _, labels = torch.max(labels.data, 1)
            correct += (pred == labels).float().sum().item()
            total += labels.size(0)
            accuracy = 100 * correct / total

            tr_loss += loss.detach().item()
            avr_loss = tr_loss / (i+1)
            if hvd.rank () == 0:
                pbar.set_description("Train Epoch {} | Loss {:.4f} ACC {:.4f} ".format(epoch, avr_loss, accuracy))
                pbar.update(1)

        torch.cuda.empty_cache()
        tr_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm.tqdm(validloader, disable=disable_pbar)
            model.eval()
            for i, (images, labels) in enumerate(pbar):
                images = images.cuda()
                labels = labels.cuda()
                logits = model(images.float())
                loss = criterion(logits, labels)
                _, pred = torch.max(logits.data, 1)
                correct += (pred == labels).float().sum().item()
                total += labels.size(0)
                accuracy = 100 * correct / total

                tr_loss += loss.detach().item()
                avr_loss = tr_loss / (i+1)
                if hvd.rank () == 0:    
                    pbar.set_description("Valid Epoch {} | Loss {:.4f} ACC {:.4f}".format(epoch, avr_loss, accuracy))
                    pbar.update(1)

        torch.cuda.empty_cache()
        if hvd.rank () == 0:
            ckpt_name = "{}_epoch_{}_loss_{:.2f}_optim_{}_lr_{}.pth".format(MODEL_NAME, epoch, avr_loss, OPTIM, LR)
            model_path = os.path.join(SAVE_PATH, ckpt_name)
            save_model(model_path, model, optimizer, scheduler)

    """ TEST """
    torch.cuda.empty_cache()
    with open(TEST_DIR + 'metadata.json', "r", encoding="ISO-8859-1") as file:
        test = json.load(file)

    test_df = pd.DataFrame(test['images'])
    if hvd.rank() == 0: print(len(test_df))

    X_Test = test_df['file_name'].values
    Transform = transforms.Compose(
        [ transforms.Resize((IMG_SIZE, IMG_SIZE)),
          transforms.ToTensor(),
          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    testset = GetData(TEST_DIR, X_Test, None, Transform)
    testloader = DataLoader(testset, batch_size=20, num_workers=8)

    s_ls = []

    with torch.no_grad():
        pbar = tqdm.tqdm(testloader, disable=disable_pbar)
        model.eval()
        for image, fname in pbar:
            image = image.cuda()

            logits = model(image)
            ps = torch.exp(logits)
            _, top_class = ps.topk(1, dim=1)

            for pred in top_class:
                s_ls.append([fname[0].split('/')[-1][:-4], pred.item()])
            pbar.update(1)


    sub = pd.DataFrame.from_records(s_ls, columns=['Id', 'Predicted'])
    sub.head()

    sub.to_csv(os.path.join(SAVE_PATH,"submission.csv"), index=False)    
