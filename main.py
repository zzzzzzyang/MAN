# -*- coding: utf-8 -*-
import sys
import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
import transform as T
from randaugument import RandomAugment

from dataset import RafDataset
from util import *
from cnn import resModel
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--raf_path', type=str, default='/xxx/RAF-DB',help='raf_dataset_path')
parser.add_argument('--train_label_path', type=str, default='/xxx/RAF-DB/EmoLabel/train_label.txt', help='train_label_path')
parser.add_argument('--test_label_path', type=str, default='/xxx/RAF-DB/EmoLabel/test_label.txt', help='test_label_path')
parser.add_argument('--pretrained_backbone_path', type=str, default='/MAN/pretrained_model/resnet18_msceleb.pth', help='pretrained_backbone_path')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--warmup', type=int, default=10, help='number of epochs for warmup')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
parser.add_argument('--w_gamma', type=int, default=1, help='the weight of Lusc')
parser.add_argument('--w_omiga', type=float, default=0.6, help='the weight of Lmut')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=30)
parser.add_argument('--num_class', type=int, default=7)
args = parser.parse_args()
device = torch.device('cuda:0')

def warmup(epoch, net1, net2, optimizer, train_loader, CEloss):
    net1.train()
    net2.train()
    num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1
    for batch_idx, (inputs, inputs2, labels, indexes) in enumerate(train_loader):      
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs1 = net1(inputs)               
        loss1 = CEloss(outputs1, labels)  

        outputs2 = net2(inputs)
        loss2 = CEloss(outputs2, labels)     
        L = 0.5 * (loss1 + loss2)
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t loss1: %.4f loss2: %.4f CE-loss: %.4f'
                %(epoch, args.epochs, batch_idx+1, num_iter, loss1.item(), loss2.item(), L.item()))
        sys.stdout.flush()
    
def test(epoch,net1,net2,test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    correct1 = 0
    correct2 = 0
    total = 0
    total1 = 0
    total2 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = net1(inputs)
            _, predicted1 = torch.max(outputs1, 1)
            total1 += targets.size(0)
            correct1 += predicted1.eq(targets).cpu().sum().item()   

            outputs2 = net2(inputs)           
            _, predicted2 = torch.max(outputs2, 1)
            total2 += targets.size(0)
            correct2 += predicted2.eq(targets).cpu().sum().item()   

            outputs = 0.5 * (outputs1+outputs2)
            _, predicted = torch.max(outputs, 1)            
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    acc1 = 100.*correct1/total1
    acc2 = 100.*correct2/total2
    print("\nTest Epoch #%d\t Accuracy1: %.2f%%, Accuracy2: %.2f%%, Accuracy: %.2f%%\n" %(epoch,acc1, acc2, acc))  
    return acc
    
class Partition():
    def __init__(self):
        self.clean_list = []
        self.ambiguous_list = []
        self.noise_list = []

    def judge(self, pred1, pred2, labels, indexes):
        # clean
        correct1 = pred1.eq(labels)
        correct1_idx = torch.nonzero(correct1)
        correct2 = pred2.eq(labels)
        correct2_idx = torch.nonzero(correct2)
        clean_idx_bz = self.intersection(correct1_idx, correct2_idx)
        clean_idx_ds = indexes[clean_idx_bz]
        self.clean_list.append(clean_idx_ds)
        
        # noise
        error1_idx = torch.nonzero(correct1==False)
        error2_idx = torch.nonzero(correct2==False)
        noise_idx_bz = self.intersection(error1_idx, error2_idx)
        noise_idx_ds = indexes[noise_idx_bz]
        self.noise_list.append(noise_idx_ds)

        # ambiguous
        nonambi_idx_bz = torch.cat((clean_idx_bz, noise_idx_bz), dim=0)
        all_idx_bz = torch.arange(labels.size()[0], device=device)
        ambiguous_idx_bz = self.diff(all_idx_bz, nonambi_idx_bz)
        ambiguous_idx_ds = indexes[ambiguous_idx_bz]
        self.ambiguous_list.append(ambiguous_idx_ds)
    
    def intersection(self, t1, t2):
        indices = torch.ones_like(t1, dtype=torch.uint8, device=device)
        for ele in t2:
            indices = indices & (t1 != ele)
        return t1[1-indices]

    def diff(self, t1, t2):
        indices = torch.ones_like(t1, dtype=torch.uint8, device=device)
        for ele in t2:
            indices = indices & (t1 != ele)
        return t1[indices]

    def concat(self):
        self.clean_t, self.noise_t, self.ambiguous_t = torch.cat(self.clean_list, dim=0), torch.cat(self.noise_list, dim=0), torch.cat(self.ambiguous_list, dim=0)
    
    def split(self, indexes):
        clean_idx = []
        noise_idx = []
        ambiguous_idx = []
        for idx in indexes:
            if self.clean_t.eq(idx).sum() == 1:
                clean_idx.append(idx)
            elif self.noise_t.eq(idx).sum() == 1:
                noise_idx.append(idx)
            elif self.ambiguous_t.eq(idx).sum() == 1:
                ambiguous_idx.append(idx)
        return torch.tensor(clean_idx, dtype=torch.int64, device=device), torch.tensor(noise_idx, dtype=torch.int64, device=device), torch.tensor(ambiguous_idx, dtype=torch.int64, device=device)

# convert dataset-index to batch-index
def ds_idx2bz_idx(sample_idx, indexes):
    result = []
    for idx in sample_idx:
        result.append(torch.nonzero(indexes.eq(idx))[0])
    return torch.tensor(result, dtype=torch.int64, device=device)

def eval_train(net1, net2, train_loader):
    tripart = Partition()
    net1.eval()
    net2.eval()
    
    with torch.no_grad():
        for batch_i, (imgs, imgs2, labels, indexes) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            indexes = indexes.to(device)
            outputs1 = net1(imgs) 
            _, predicted1 = torch.max(outputs1, 1) # 1d tensor
            outputs2 = net2(imgs)
            _, predicted2 = torch.max(outputs2, 1)
            tripart.judge(predicted1, predicted2, labels, indexes)
        tripart.concat()   
    return tripart 

def partition_train(args, epoch, net1, net2, optimizer, train_loader, tripart):
    net1.train()
    net2.train()
    num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1
    for batch_idx, (inputs, inputs2, labels, indexes) in enumerate(train_loader):    
        loss_c = 0.
        loss_n = 0.
        loss_a = 0.  
        inputs, labels, indexes = inputs.to(device), labels.to(device), indexes.to(device)
        clean_idx, noise_idx, ambiguous_idx = tripart.split(indexes)
        if clean_idx.shape[0] != 0:
            clean_imgs_idx = ds_idx2bz_idx(clean_idx, indexes)
            clean_imgs = inputs[clean_imgs_idx]
            clean_labels = labels[clean_imgs_idx]

            clean_outputs1 = net1(clean_imgs)
            clean_outputs2 = net2(clean_imgs)
            
            loss_c1 = F.cross_entropy(clean_outputs1, clean_labels)
            loss_c2 = F.cross_entropy(clean_outputs2, clean_labels)
            
            loss_c = (loss_c1 + loss_c2) * 0.5
            loss_c_item = loss_c.item()
        else:
            loss_c_item = 0.

        if noise_idx.shape[0] != 0:
            noise_imgs_idx = ds_idx2bz_idx(noise_idx, indexes)
            noise_imgs_weak, noise_imgs_strong = inputs[noise_imgs_idx], inputs2[noise_imgs_idx]
            noise_imgs_strong = noise_imgs_strong.to(device)

            noise_outputs1_weak = net1(noise_imgs_weak)
            noise_outputs1_strong = net1(noise_imgs_strong)
            p_1 = F.softmax(noise_outputs1_weak, dim=1)
            p_2 = F.softmax(noise_outputs1_strong, dim=1)
            loss_n1 = F.mse_loss(p_1, p_2)

            noise_outputs2_weak = net2(noise_imgs_weak)
            noise_outputs2_strong = net2(noise_imgs_strong)
            p_3 = F.softmax(noise_outputs2_weak, dim=1)
            p_4 = F.softmax(noise_outputs2_strong, dim=1)
            loss_n2 = F.mse_loss(p_3, p_4)
            
            loss_n = 0.5 * (loss_n1 + loss_n2)
            loss_n_item = args.w_gamma * loss_n.item() 
        else:
            loss_n_item = 0.

        if ambiguous_idx.shape[0] != 0:
            ambi_imgs_idx = ds_idx2bz_idx(ambiguous_idx, indexes)
            ambi_imgs = inputs[ambi_imgs_idx]
            ambi_labels = labels[ambi_imgs_idx]
            ambi_outputs1 = net1(ambi_imgs)
            ambi_outputs2 = net2(ambi_imgs)

            loss_a = Mutual_loss(ambi_outputs1, ambi_outputs2, ambi_labels, epoch_num=epoch, max_epochs=args.epochs)

            loss_a_item = args.w_omiga * loss_a.item()
        else:
            loss_a_item = 0.   

        optimizer.zero_grad()           
        L = loss_c + args.w_omiga * loss_a + args.w_gamma * loss_n
        L.backward()  
        optimizer.step() 

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Clean CELoss: %.4f, Noise MSELoss: %.4f, Ambiguous MutualLoss: %.4f '
                % (epoch, args.epochs, batch_idx + 1, num_iter, loss_c_item, loss_n_item, loss_a_item ))


def kl_loss(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def Mutual_loss(y_1, y_2, t, lambda_max = 0.9, beta = 0.65, epoch_num = 1, max_epochs = 100): 
    ''' 
        y_1, y_2 are predictions of two networks and t is target labels. 
    '''
    e = epoch_num
    e_r = 0.9 * max_epochs
    lambda_ = lambda_max * math.exp(-1.0 * beta * (1.0 - e / e_r ) ** 2) #Dynamic balancing factor using Gaussian like ramp-up function
    
    loss_ce_1 = F.cross_entropy(y_1, t) 
    loss_ce_2 = F.cross_entropy(y_2, t)
    loss_ce = (1 - lambda_) * 0.5 * (loss_ce_1 + loss_ce_2)    #Supervision Loss weighted by (1 - dynamic balancing factor)
    
    loss_mimicry =  lambda_ * 0.5 * ( kl_loss(y_1, y_2) +  kl_loss(y_2, y_1))  #Mimicry Loss weighted by dynamic balancing factor
    loss = loss_mimicry + loss_ce
    
    return loss     
    
def train():
    setup_seed(0)
    net1 = resModel(args, device)
    net2 = resModel(args, device)

    optimizer = torch.optim.Adam(net1.parameters(),weight_decay = 1e-4)
    optimizer.add_param_group({'params': net2.parameters(), 'weigh_decay': 1e-4})

    trans_weak = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),        
        transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                        transforms.RandomAffine(degrees=0, translate=(.1, .1),
                                               scale=(1.0, 1.25),
                                               resample=Image.BILINEAR)],p=0.5),
        
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))
        ])
        
    trans_strong = T.Compose([
        T.Resize((224, 224)),
        T.PadandRandomCrop(border=4, cropsize=(224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        RandomAugment(2, 10),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.ToTensor(),
    ])
        
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
                                 
    train_dataset = RafDataset(args, phase = 'train', transform = [trans_weak, trans_strong], basic_aug = True)    
    print('Train set size:', train_dataset.__len__())       
                 
    test_dataset = RafDataset(args, phase = 'test', transform = data_transforms_val)    
    print('Validation set size:', test_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = True,  
                                               pin_memory = True) 
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = False,  
                                               pin_memory = True) 


    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    CEloss = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0
    for i in range(1, args.epochs + 1):
        if i <= args.warmup:
            print('Warmup')
            warmup(i, net1, net2, optimizer, train_loader, CEloss)
            scheduler.step()
        else:
            # Partition
            print('Partition Train')
            tripart = eval_train(net1, net2, train_loader)
            print('Clean Data:%d    Noise Data:%d   Ambiguous Data:%d' 
                    % (tripart.clean_t.shape[0], tripart.noise_t.shape[0], tripart.ambiguous_t.shape[0]))
            partition_train(args, i, net1, net2, optimizer, train_loader, tripart)
            scheduler.step()

        test_acc = test(i,net1,net2,test_loader)  
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = i
            if best_acc >= 90.2:
                torch.save({'model1_state_dict': net1.state_dict(),
                            'model2_state_dict': net2.state_dict()},
                        "checkpoints/epoch%d_acc%.4f.pth" % (i, best_acc))
                print('Model saved.')
    print('best acc: ', best_acc, 'best epoch: ', best_epoch)

if __name__ == '__main__':
    train()
