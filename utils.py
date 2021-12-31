import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def cross_entropy_OH(score,one_hot):
    
    log_prob = F.log_softmax(score, dim=1) ### Scores are the logits (not normalized)
    #pdb.set_trace()
    loss = -torch.sum(log_prob * one_hot) / one_hot.shape[0]
    return loss

# custom loss: CE of hard-label & CE of soft-label
def CE_HL_SL(hs_alpha,y_true,y_pred, num_classes):
    
    log_prob = F.log_softmax(y_pred, dim=1)
    ce_hard = -torch.sum(log_prob * y_true[:,:num_classes]) / (y_true[:,:num_classes].shape[0])
    ce_soft = -torch.sum(log_prob * y_true[:,num_classes:]) / (y_true[:,num_classes:].shape[0])
#     ce_hard = cross_entropy_OH(y_pred, y_true[:,:num_classes])
#     ce_soft = cross_entropy_OH(y_pred, y_true[:,num_classes:])
    loss = (1-hs_alpha) * ce_hard + hs_alpha * ce_soft
    retrun loss
        
# custom loss: CE of labeled data & Entropy of unlabeled data
def lu_loss_semi_SL_HL(lu_alpha,hs_alpha,y_true,y_pred, num_classes):

    unlbl_indc = (1-torch.sum(y_true,axis=1)).float()
    log_prob = F.log_softmax(y_pred, dim=1)
    ce_lbl_HL = -torch.sum(log_prob * y_true[:,:num_classes]) / (torch.sum(1-unlbl_indc))# avg of CE of labeled samples
    ce_lbl_SL = -torch.sum(log_prob * y_true[:,num_classes:]) / (torch.sum(1-unlbl_indc))# avg of CE of labeled samples
    
    # y_pred: B * num_classes ; ce_y_pred_ysl: B * num_classes = - log(y_pred) * y_sl 
    ce_ypred_ysl = torch.matmul(-log_prob,torch.transpose(q_dist,0,1)) ### cast to float32????
    
    expect_ce = torch.sum(F.softmax(y_pred) * ce_ypred_ysl,axis=1) # expectation of CE 
    h_unlbl = torch.sum(unlbl_indc * expect_ce,axis=-1)/torch.sum(unlbl_indc) # avg of H of unlabeled samples
    # final loss
    loss_labeled = (1-hs_alpha) * ce_lbl_HL + hs_alpha * ce_lbl_SL
    loss = (1-lu_alpha) * loss_labeled + lu_alpha * h_unlbl
    return loss

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
