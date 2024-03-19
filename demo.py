# -*- coding:utf-8 -*-
# @Time       :2023/4/27 上午10.55
# @AUTHOR     :Jiaqing Zhang
# @FileName   :demo.py
import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
from MMA import MMA
import numpy as np
import time
import os
from utils import train_patch, setup_seed, output_metric, print_args, train_epoch, valid_epoch,loss_weight_calculation,loss_weight_calculation_np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("MMA")
parser.add_argument('--gpu_id', default='5', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--distillation', type=int, default=1, help='distillation') #是否加蒸馏
parser.add_argument('--fusion', choices=['TTOA', 'trans','trans_s1_c1','trans_c1_s1','trans_TTOA_attention','trans_cs_sc'], default='TTOA', help='fusion method') #确认用哪个做融合
parser.add_argument('--test_freq', type=int, default=1, help='number of evaluation')
parser.add_argument('--pred_flag', choices=['o_fuse','o_1','o_2','o_cnn','o_trans'], default='o_fuse', help='dataset to use')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')  
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')  # diffGrad 1e-3
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston','Houston_nocloud','Berlin','Augsburg','Rochester'], default='Houston', help='dataset to use')
parser.add_argument('--model_name', choices=['MMA'], default='MMA', help='dataset to use')
parser.add_argument('--num_classes', type=int,choices=[11, 6, 15, 8, 7, 10], default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patches1', type=int, default=8, help='number1 of patches')
parser.add_argument('--patches2', type=int, default=16, help='number2 of patches')
parser.add_argument('--patches3', type=int, default=24, help='number3 of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')  
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def train_1times():
    # setup_seed(args.seed)
    # -------------------------------------------------------------------------------
    # prepare data
    if args.dataset == 'Houston':
        DataPath1 = './dataset/Houston.mat'
        DataPath2 = './dataset/LiDAR_MP.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
        TrLabel_10TIMES = loadmat(DataPath1)['trainlabels']  # 349*1905 完整训练集
        TsLabel_10TIMES = loadmat(DataPath1)['testlabels']  # 349*1905
    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    patchsize1 = args.patches1  # input spatial size for 2D-CNN
    pad_width1 = np.floor(patchsize1 / 2)
    pad_width1 = int(pad_width1)  # 8
    patchsize2 = args.patches2  # input spatial size for 2D-CNN
    pad_width2 = np.floor(patchsize2 / 2)
    pad_width2 = int(pad_width2)  # 8
    patchsize3 = args.patches3  # input spatial size for 2D-CNN
    pad_width3 = np.floor(patchsize3 / 2)
    pad_width3 = int(pad_width3)  # 8

    if args.flag_test == 'train':
        TrainPatch11, TrainPatch21, TrainLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TrLabel_10TIMES)
        TestPatch11, TestPatch21, TestLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)
        TrainPatch12, TrainPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TrLabel_10TIMES)
        TestPatch12, TestPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TsLabel_10TIMES)
        TrainPatch13, TrainPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TrLabel_10TIMES)
        TestPatch13, TestPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TsLabel_10TIMES)

        train_dataset = Data.TensorDataset(TrainPatch11, TrainPatch21, TrainPatch12, TrainPatch22, TrainPatch13, TrainPatch23, TrainLabel)
        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
        test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatch12, TestPatch22, TestPatch13, TestPatch23, TestLabel)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
        loss_weight = loss_weight_calculation(TrainLabel).cuda()
        print("the weight of loss is ",loss_weight)
    else:
        TestPatch11, TestPatch21, TestLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)
        TestPatch12, TestPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TsLabel_10TIMES)
        TestPatch13, TestPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TsLabel_10TIMES)

        test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatch12, TestPatch22, TestPatch13, TestPatch23, TestLabel)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
        loss_weight = torch.ones(int(TestLabel.max())+1).cuda()
        print("the weight of loss is ",loss_weight)
    [m1, n1, l1] = np.shape(Data1)
    

    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
    # -------------------------------------------------------------------------------
    # create model
    model = MMA(l1=band1, l2=band2, patch_size=args.patches1, num_patches=64, num_classes=args.num_classes,
        encoder_embed_dim=64, decoder_embed_dim=32,
        en_depth=5, en_heads=4, de_depth=5, de_heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1,fusion=args.fusion)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    # -------------------------------------------------------------------------------
    # train and test
    #lambda1,lambda2,lambda3,lambda4
    lambda3 = 0.3
    lambda4 = 1
    lambda1 = 5
    lambda2 = 0.1
    if args.flag_test == 'train':
        BestAcc = 0
        BestAA = 0
        BestKappa = 0
        val_acc = []
        val_acc2 = []
        print("start training")
        tic = time.time()
        if args.distillation==1:
            print("we use the distillation")
        else:
            print("we do not use the distillation")
        for epoch in range(args.epoches):
            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, loss_weight, optimizer,args.distillation,epoch,args.epoches,args.dataset,lambda1,lambda2,lambda3,lambda4)
            print("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f}"
                .format(epoch + 1, train_obj, train_acc))
            scheduler.step()

            if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                model.eval()
                val_acc_, tar_v, pre_v, pre_v2,_,_,_ = valid_epoch(model, test_loader, loss_weight,args.pred_flag)
                OA2, AA2, Kappa2, CA2 = output_metric(tar_v, pre_v)
                val_acc.append(val_acc_)
                print("Every {} epochs' records:".format(args.test_freq))
                print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))

                if val_acc_ > BestAcc and AA2>BestAA and Kappa2>BestKappa:
                    print('save the model',args.model_name+'_'+args.fusion+'_distillation'+str(args.distillation)+'_'+args.dataset+'_ps'+str(args.patches1)+'_'+str(args.epoches)+'e'+'_'+str(lambda1)+str(lambda2)+str(lambda3)+str(lambda4)+'.pkl')
                    torch.save(model.state_dict(), args.model_name+'_'+args.fusion+'_distillation'+str(args.distillation)+'_'+args.dataset+'_ps'+str(args.patches1)+'_'+str(args.epoches)+'e'+'_'+str(lambda1)+str(lambda2)+str(lambda3)+str(lambda4)+'.pkl')
                    BestAcc = val_acc_
                    BestAA = AA2
                    BestKappa = Kappa2

        toc = time.time()
        model.eval()
        model.load_state_dict(torch.load(args.model_name+'_'+args.fusion+'_distillation'+str(args.distillation)+'_'+args.dataset+'_ps'+str(args.patches1)+'_'+str(args.epoches)+'e'+'_'+str(lambda1)+str(lambda2)+str(lambda3)+str(lambda4)+'.pkl'))
        val_acc_,tar_v, pre_v,pre_v2,_,_,_ = valid_epoch(model, test_loader, loss_weight,args.pred_flag)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)
        print("Final records:")
        print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        print(CA)
        print("Running Time: {:.2f}".format(toc - tic))
        print("**************************************************")
        print("Parameter:")
        print_args(vars(args))
    else:
        model.eval()
        model.load_state_dict(torch.load('/home/zjq/MIF/MMAv1_3TTOA_mi_our_0.1_weightnew_oa_aa_k/use/'+args.model_name+'_'+args.fusion+'_distillation'+str(args.distillation)+'_'+args.dataset+'_ps'+str(args.patches1)+'_'+str(args.epoches)+'e'+'.pkl'))
        val_acc_,tar_v, pre_v,pre_v2,_,_,_ = valid_epoch(model, test_loader, loss_weight,args.pred_flag)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)
        print("Final records:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        print(CA)
        print("**************************************************")
if __name__ == '__main__':
    setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1times()



