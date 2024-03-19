# -*- coding:utf-8 -*-
# @Time       :2023/4/27 上午10.55
# @AUTHOR     :Jiaqing Zhang
# @FileName   :utils.py
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from calen import calen
class OhemCELoss(nn.Module):
 
    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
 
    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        # print(min(loss))
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

class FocalLoss(nn.Module):
    def __init__(self, loss_weight,alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets,weight = self.loss_weight, reduction= 'none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


def normalize(input2):
    input2_normalize = np.zeros(input2.shape)
    for i in range(input2.shape[2]):
        input2_max = np.max(input2[:, :, i])
        input2_min = np.min(input2[:, :, i])
        input2_normalize[:, :, i] = (input2[:, :, i] - input2_min) / (input2_max - input2_min)

    return input2_normalize


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False


def train_patch_tsne(Data1, Data2, patchsize, pad_width, Label):
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2
    # construct the training and testing set
    # ind1 = np.zeros(int(100*Label.max()))
    # ind2 = np.zeros(int(100*Label.max()))
    # for i in range(1,Label.max()+1):
    #     [ind_1, ind_2] = np.where(Label == i)
    #     ind1[(i-1)*100:i*100] =  ind_1[:100]
    #     ind2[(i-1)*100:i*100] = ind_2[:100]
    
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainLabel

def train_patch(Data1, Data2, patchsize, pad_width, Label):
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2
    # construct the training and testing set
    [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainLabel

def test_patch(Data1, Data2, patchsize, pad_width, Label):
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    # for i in range(l1):
    #     Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    # for i in range(l2):
    #     Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2
    # construct the training and testing set
    [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainLabel

# def test_patch(Data1, Data2, patchsize, pad_width, Label):
#     [m1, n1, l1] = np.shape(Data1)
#     Data2 = Data2.reshape([m1, n1, -1])
#     [m2, n2, l2] = np.shape(Data2)
#     for i in range(l1):
#         Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
#     x1 = Data1
#     for i in range(l2):
#         Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
#     x2 = Data2
#     x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
#     x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
#     for i in range(l1):
#         temp = x1[:, :, i]
#         temp2 = np.pad(temp, pad_width, 'symmetric')
#         x1_pad[:, :, i] = temp2
#     for i in range(l2):
#         temp = x2[:, :, i]
#         temp2 = np.pad(temp, pad_width, 'symmetric')
#         x2_pad[:, :, i] = temp2
#     # construct the training and testing set
#     [ind1, ind2] = np.where(Label > 0)
#     TrainNum = len(ind1)
#     TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
#     TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
#     TrainLabel = np.empty(TrainNum)
#     ind3 = ind1 + pad_width
#     ind4 = ind2 + pad_width
#     for i in range(len(ind1)):
#         patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
#         patch1 = np.transpose(patch1, (2, 0, 1))
#         TrainPatch1[i, :, :, :] = patch1
#         patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
#         patch2 = np.transpose(patch2, (2, 0, 1))
#         TrainPatch2[i, :, :, :] = patch2
#         patchlabel = Label[ind1[i], ind2[i]]
#         TrainLabel[i] = patchlabel
#     # change data to the input type of PyTorch
#     TrainPatch1 = torch.from_numpy(TrainPatch1)
#     TrainPatch2 = torch.from_numpy(TrainPatch2)
#     TrainLabel = torch.from_numpy(TrainLabel) - 1
#     TrainLabel = TrainLabel.long()
#     return TrainPatch1, TrainPatch2, TrainLabel

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


def cal_loss(f_s, f_t,reduction='sum'):
    p_s = F.log_softmax(f_s, dim=1)
    p_t = F.softmax(f_t, dim=1)
    loss = F.kl_div(p_s, p_t, reduction=reduction) / f_t.shape[0]
    return loss

def adjust(init, fin, step, fin_step):
    if fin_step == 0:
        return  fin
    deta = fin - init
    adj = min(init + deta * step / fin_step, fin)
    return adj

def train_epoch(model, train_loader, loss_weight, optimizer,distillation,epoch,num_epochs,datasetname,lambda1=5,lambda2=0.1,lambda3=0.3,lambda4=1):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])
    # MSE = nn.MSELoss()
    # L1 = nn.L1Loss()
    # Ohem = OhemCELoss(0.9)
    criterion = FocalLoss(loss_weight,gamma=2,alpha=None) #更换新的损失函数
    # CE = torch.nn.BCELoss(reduction='mean')
    for batch_idx, (batch_data11, batch_data21, batch_data12, batch_data22, batch_data13, batch_data23, batch_target) in enumerate(train_loader):
        batch_data11 = batch_data11.cuda()
        batch_data21 = batch_data21.cuda()
        batch_data12 = batch_data12.cuda()
        batch_data22 = batch_data22.cuda()
        batch_data13 = batch_data13.cuda()
        batch_data23 = batch_data23.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        batch_pred,x_cls_cnn,x_cls_trans,x1_out,x2_out, con_loss,x1c_out,loss_ml,x_fuse1,x_fuse2,x_transfusion = model(batch_data11, batch_data21, batch_data12, batch_data22, batch_data13, batch_data23)
        # x_cls,x_cls_cnn,x_cls_trans,x1_out,x2_out, con_loss,x1c_out,mutual_loss,x_fuse1,x_fuse2,x_transfusion
        # if batch_pred.equal(batch_pred_2):
        # if datasetname=='Augsburg':
        #     loss = criterion(batch_pred, batch_target)  #为什么在重建损失前面加了一个5？？？ 如果不做重建的话效果如何
        # else:
        loss = criterion(batch_pred, batch_target) + lambda1*con_loss
        # else:
        #     loss = criterion(batch_pred, batch_target) + criterion(batch_pred_2, batch_target)+ 5*con_loss
        if distillation==1:
            # if datasetname=='Augsburg':
            #     loss += 0.3*criterion(x1_out, batch_target)
            #     loss += 0.3*criterion(x2_out, batch_target)
            #     loss += 0.3*criterion(x1c_out, batch_target)
                
            #     #distillation loss
            #     loss += cal_loss(x1_out,batch_pred) #蒸馏损失
            #     loss += cal_loss(x2_out,batch_pred)
            #     loss += cal_loss(x1c_out,batch_pred)
            # # cross-entropy loss
            # else:
            loss += lambda3*criterion(x1_out, batch_target)
            loss += lambda3*criterion(x2_out, batch_target)
            loss += lambda3*criterion(x1c_out, batch_target)
            # loss += lambda3*criterion(x_cls_cnn, batch_target)
            # loss += lambda3*criterion(x_cls_trans, batch_target)
            
            #distillation loss
            loss += lambda4*cal_loss(x1_out,batch_pred) #蒸馏损失
            loss += lambda4*cal_loss(x2_out,batch_pred)
            loss += lambda4*cal_loss(x1c_out,batch_pred)
            # loss += lambda4*cal_loss(x_cls_cnn,batch_pred)
            # loss += lambda4*cal_loss(x_cls_trans,batch_pred)

            # mutual information loss
            # if datasetname=='Augsburg':
            #     loss  +=  loss_ml * 0.01 #* adjust(0, 1, epoch, num_epochs)
            # else:
            #     loss  +=  loss_ml * 0.1
            loss  +=  loss_ml * lambda2



        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data11.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

def valid_epoch(model, valid_loader, loss_weight,pred_flag):
    objs = AverageMeter()
    top1 = AverageMeter()
    objs2 = AverageMeter()
    top12 = AverageMeter()
    tar = np.array([])
    pre = np.array([])
    pre1 = np.array([])
    emb = np.array([])
    criterion = FocalLoss(loss_weight,gamma=2,alpha=None)
    # labels = []
    embs = []
    embs1 = []
    embs2 = []
    for batch_idx, (batch_data11, batch_data21, batch_data12, batch_data22, batch_data13, batch_data23, batch_target) in enumerate(valid_loader):
        batch_data11 = batch_data11.cuda()
        batch_data21 = batch_data21.cuda()
        batch_data12 = batch_data12.cuda()
        batch_data22 = batch_data22.cuda()
        batch_data13 = batch_data13.cuda()
        batch_data23 = batch_data23.cuda()
        batch_target = batch_target.cuda()

        batch_pred,x_cls_cnn,x_cls_trans ,x1_out,x2_out, con_loss,x1c_out, loss_ml,x_fuse1,x_fuse2,x_transfusion = model(batch_data11, batch_data21, batch_data12, batch_data22, batch_data13, batch_data23)
        if pred_flag == 'o_fuse':
            choice_pred = batch_pred
        elif pred_flag == 'o_1':
            choice_pred = x1_out
        elif pred_flag == 'o_2':
            choice_pred = x2_out
        elif pred_flag == 'o_cnn':
            choice_pred = x_cls_cnn    
        elif pred_flag == 'o_trans':
            choice_pred = x_cls_trans   
        loss = criterion(choice_pred, batch_target)  + con_loss #+ criterion(batch_pred_2, batch_target)

        # if batch_idx == 70:
        #     calen(x_fuse1,x_fuse2,"co_1_2.png")
        #     calen(x_fuse1,x_transfusion,"co_1_f.png")
        #     calen(x_fuse2,x_transfusion,"co_2_f.png")

        prec1, t, p = accuracy(choice_pred, batch_target, topk=(1,))
        n = batch_data11.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy()) #(29395,)
        pre = np.append(pre, p.data.cpu().numpy()) 
        #emb = np.append(emb, batch_pred.data.cpu().numpy()) #(176370,)
        # embs.append(batch_pred.data.cpu().numpy())
        x_transfusion = x_transfusion.reshape(x_transfusion.shape[0],-1)
        embs.append(x_transfusion.data.cpu().numpy())
        x_fuse1 = x_fuse1.reshape(x_fuse1.shape[0],-1)
        embs1.append(x_fuse1.data.cpu().numpy())
        x_fuse2 = x_fuse2.reshape(x_fuse2.shape[0],-1)
        embs2.append(x_fuse2.data.cpu().numpy())


        
    embs = np.concatenate(embs)
    embs1 = np.concatenate(embs1)
    embs2 = np.concatenate(embs2)

    return top1.avg,tar, pre, pre1,embs,embs1,embs2

def test_epoch(model, valid_loader, loss_weight):
    objs = AverageMeter()
    top1 = AverageMeter()
    objs2 = AverageMeter()
    top12 = AverageMeter()
    tar = np.array([])
    pre = np.array([])
    pre1 = np.array([])
    emb = np.array([])
    criterion = FocalLoss(loss_weight,gamma=2,alpha=None)
    # labels = []
    embs = []
    for batch_idx, (batch_data11, batch_data21, batch_data12, batch_data22, batch_data13, batch_data23, batch_target) in enumerate(valid_loader):
        batch_data11 = batch_data11.cuda()
        batch_data21 = batch_data21.cuda()
        batch_data12 = batch_data12.cuda()
        batch_data22 = batch_data22.cuda()
        batch_data13 = batch_data13.cuda()
        batch_data23 = batch_data23.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data11, batch_data21, batch_data12, batch_data22, batch_data13, batch_data23)[0]

        loss = criterion(batch_pred, batch_target)  #+ criterion(batch_pred_2, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data11.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy()) #(29395,)
        pre = np.append(pre, p.data.cpu().numpy()) 
        #emb = np.append(emb, batch_pred.data.cpu().numpy()) #(176370,)
        embs.append(batch_pred.data.cpu().numpy())
    embs = np.concatenate(embs)

    return top1.avg,tar, pre, pre1,embs


## To show the last all classification result
def test(model,batch_data11, batch_data21, batch_data12, batch_data22, batch_data13, batch_data23, TestLabel, Classes,height1, width1):

    pred_y = np.empty((len(TestLabel)), dtype='float32')
    number = len(TestLabel) // 100
    for i in range(number):
        temp1_1 = batch_data11[i * 100:(i + 1) * 100, :, :, :]
        temp2_1 = batch_data21[i * 100:(i + 1) * 100, :, :, :]
        temp1_2 = batch_data12[i * 100:(i + 1) * 100, :, :, :]
        temp2_2 = batch_data22[i * 100:(i + 1) * 100, :, :, :]
        temp1_3 = batch_data13[i * 100:(i + 1) * 100, :, :, :]
        temp2_3 = batch_data23[i * 100:(i + 1) * 100, :, :, :]
        temp1_1 = temp1_1.cuda()
        temp2_1 = temp2_1.cuda()
        temp1_2 = temp1_2.cuda()
        temp2_2 = temp2_2.cuda()
        temp1_3 = temp1_3.cuda()
        temp2_3 = temp2_3.cuda()
        temp2 = model(temp1_1, temp2_1, temp1_2,  temp2_2,  temp1_3,  temp2_3)[0]
        temp3 = torch.max(temp2, 1)[1].squeeze()

        pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
        print(i)
        del temp1_1, temp1_2, temp2, temp3

    if (i + 1) * 100 < len(TestLabel):
        temp1_1 = batch_data11[(i + 1) * 100:len(TestLabel), :, :, :]
        temp2_1 = batch_data21[(i + 1) * 100:len(TestLabel), :, :, :]
        temp1_2 = batch_data12[(i + 1) * 100:len(TestLabel), :, :, :]
        temp2_2 = batch_data22[(i + 1) * 100:len(TestLabel), :, :, :]
        temp1_3 = batch_data13[(i + 1) * 100:len(TestLabel), :, :, :]
        temp2_3 = batch_data23[(i + 1) * 100:len(TestLabel), :, :, :]
        temp1_1 = temp1_1.cuda()
        temp2_1 = temp2_1.cuda()
        temp1_2 = temp1_2.cuda()
        temp2_2 = temp2_2.cuda()
        temp1_3 = temp1_3.cuda()
        temp2_3 = temp2_3.cuda()
        temp2 = model(temp1_1, temp2_1, temp1_2,  temp2_2,  temp1_3,  temp2_3)[0]
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
        del temp1_1, temp1_2, temp2, temp3

    pred_y = torch.from_numpy(pred_y).long()
    # Classes = np.unique(TestLabel)
    if Classes == 15:
        colormap = colormap_houston
    elif Classes == 6:
        colormap = colormap_trento
    elif Classes == 10:
        colormap = colormap_rochester
    elif Classes == 11:
        colormap = colormap_muufl
    # num_class = 15
    colormap = colormap * 1.0 / 255
    X_result = np.zeros((height1*width1, 3))  # 创建一个RGB的图像

    for i in range(0, Classes):
        index = np.where(pred_y == i)[0]  # 取出所有值等于i的元素在X_result数组中的下标
        X_result[index, 0] = colormap[i, 0]  # 将这些元素的第0列赋值为colormap中第i个元素的第0列 R
        X_result[index, 1] = colormap[i, 1]  # 将这些元素的第1列赋值为colormap中第i个元素的第1列 G
        X_result[index, 2] = colormap[i, 2]  # 将这些元素的第2列赋值为colormap中第i个元素的第2列 B

    X_result = np.reshape(X_result, (height1, width1, 3))
    return X_result


def loss_weight_calculation(TrainLabel):
    loss_weight = torch.ones(int(TrainLabel.max())+1)
    sum_num = 0
    for i in range(TrainLabel.max()+1):
        loss_weight[i] = len(torch.where(TrainLabel ==i)[0])
        sum_num = sum_num + len(torch.where(TrainLabel ==i)[0])
    sum_mean = sum_num/(int(TrainLabel.max())+1)
    print(loss_weight)
    print(sum_num)
    print(sum_mean)
    weight_out = (sum_mean-loss_weight)/loss_weight
    weight_out[torch.where(weight_out <1)] = 1
    return weight_out#(1-loss_weight/sum)/((1-loss_weight/sum).sum())

def loss_weight_calculation_np(TrainLabel):
    loss_weight = np.ones(int((TrainLabel).max()))
    sum = 0
    for i in range(1,TrainLabel.max()+1):
        loss_weight[i-1] = len(np.where(TrainLabel ==i)[0])
        sum = sum + len(np.where(TrainLabel ==i)[0])
    print(loss_weight)
    print(sum)
    return (1-loss_weight/sum)/((1-loss_weight/sum).sum())

colormap_houston = np.array([[0, 0, 205],
                [0, 8, 255],
                [0, 77, 255],
                [0, 145, 255],
                [0, 212, 255],
                [41, 255, 206],
                [96, 255, 151],
                [151, 255, 96],
                [206, 255, 41],
                [255, 230, 0],
                [255, 167, 0],
                [255, 104, 0],
                [255, 41, 0],
                [205, 0, 0],
                [128, 0, 0]])

# colormap_trento = np.array([[61, 86, 168],
#                 [80, 200, 235],
#                 [154, 204, 105],
#                 [255, 209, 12],
#                 [238, 52, 39],
#                 [124, 21, 22],
#                 ])


# colormap_rochester = np.array([[34, 0, 255],
#                 #[0, 255, 0],
#                 [0, 128, 1],
#                 [0, 255, 255],
#                 [255, 0, 0],
#                 [255, 0, 255],
#                 [253, 255, 0],
#                 # [253, 255, 0],
#                 [128,128,128],
#                 [128,128,255],
#                 [128,255,128],
#                 [255,128,128],
#                 ])

colormap_rochester = np.array([[34, 0, 255],
                [0, 128, 1],
                [0, 255, 255],
                [255, 0, 0],
                [255, 0, 255],
                [253, 255, 0],
                [128,128,128],
                [128,128,255],
                [128,255,128],
                [255,128,128],
                ])

# colormap_muufl = np.array([[0,128,1],
#                 [0, 255, 1],
#                 [0, 255, 255],
#                 [254, 203, 0],
#                 [252, 0, 49],
#                 [2, 1, 203],
#                 [102,1,205],
#                 [254,126,151],
#                 [201,102,0],
#                 [254,254,0],
#                 [204,26,100],
#                 ])

#ldx
colormap_muufl = np.array([[0,0,205],
                         [0,8,255],
                         [0,77,255],
                         [0,145,255],
                         [0,212,255],
                         [41,255,206],
                         [96,255,151],
                         [151,255,96],
                         [206,255,41],
                         [255,230,0],
                         [255,167,0],
                         [255,104,0],
                         [255,41,0],
                         [205,0,0],
                         [128,0,0]])

colormap_trento = colormap_muufl

colormap_Augsburg = np.array([
    [32,109,45],
                [249, 20, 10],
                [228, 241, 33],
                [137, 208, 5],
                [109, 225, 3],
                [154, 242, 236],
                [70,134,198],
                ])