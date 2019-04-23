import os
import os.path as osp
import random
import time
import sys
sys.path.append("C:/Users/sport/Desktop/SiamMask-Pytorch/SiamMask")#sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
from data_loader import TrainDataLoader
from torch.nn import init
from tools.test import *
from shapely.geometry import Polygon

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='C:\\Users\\sport\\Desktop\\SiamMask-Pytorch\\DAVIS-4\\JPEGImages\\480p', metavar='DIR',help='path to dataset')

parser.add_argument('--weight_dir', default='C:\\Users\\sport\\Desktop\\SiamMask-Pytorch\\DAVIS-4\\JPEGImages\\480p', metavar='DIR',help='path to weight')

parser.add_argument('--checkpoint_path', default=None, help='resume')

parser.add_argument('--max_epoches', default=10000, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--max_batches', default=0, type=int, metavar='N', help='number of batch in one epoch')

parser.add_argument('--init_type',  default='xavier', type=str, metavar='INIT', help='init net')

parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='momentum', help='momentum')

parser.add_argument('--weight_decay', '--wd', default=5e-5, type=float, metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--debug', default=False, type=bool,  help='whether to debug')

parser.add_argument('--resume', default='', type=str, required=False,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')


def main():
    args = parser.parse_args()

    """ train dataloader """
    data_loader = TrainDataLoader("C:\\Users\\sport\\Desktop\\SiamMask-Pytorch\\DAVIS-4\\JPEGImages\\480p")

    print('-')

    """ compute max_batches """
    for root, dirs, files in os.walk(args.train_path):
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            args.max_batches += len(os.listdir(dir_path))

    # Setup Model
    cfg = load_config(args)
    from experiments.siammask.custom import Custom
    model = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model = load_pretrain(model, args.resume)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval().to(device)
    cudnn.benchmark = True

    """ loss and optimizer """
    criterion = MultiBoxLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)

    """ train phase """
    closses, rlosses, tlosses = AverageMeter(), AverageMeter(), AverageMeter()
    steps = 0
    start = 0
    for epoch in range(start, args.max_epoches):
        cur_lr = adjust_learning_rate(args.lr, optimizer, epoch, gamma=0.1)
        index_list = range(data_loader.__len__()) 
        for example in range(args.max_batches):
            ret = data_loader.__get__(random.choice(index_list)) 
            template = ret['template_tensor'].to(device)
            detection= ret['detection_tensor'].to(device)
            mask_target = ret['mask_template_tensor'].to(device)
            pos_neg_diff = ret['pos_neg_diff_tensor'].to(device)
            cout, rout, mask = model(template, detection)
            predictions, targets = (cout, rout, mask), pos_neg_diff
            closs, rloss, mloss, loss, reg_pred, reg_target, pos_index, neg_index = criterion(predictions, targets, mask_target)
            closs_ = closs.cpu().item()

            if np.isnan(closs_): 
               sys.exit(0)

            closses.update(closs.cpu().item())
            rlosses.update(rloss.cpu().item())
            tlosses.update(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

                        
            cout = cout.cpu().detach().numpy()
            # score = 1/(1 + np.exp(cout[:,0]-cout[:,1]))
            print("Epoch:{:04d}\texample:{:06d}/{:06d}({:.2f})%\tsteps:{:010d}\tlr:{:.7f}\tcloss:ss:{:.4f}\ttloss:{:.4f}".format(epoch, example+1, args.max_batches, 100*(example+1)/args.max_batches, steps, cur_lr, closses.avg, rlosses.avg, tlosses.avg ))

        if epoch % 5 == 0 :
            file_path = os.path.join(args.weight_dir, 'epoch_{:04d}_weights.pth.tar'.format(epoch))
            state = {
            'epoch' :epoch+1,
            'state_dict' :model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, file_path)




class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()

    def forward(self, predictions, targets, mask_target):
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
        cout, rout, mask = predictions
        """ class """
        class_pred, class_target = cout, targets[:, 0].long()
        pos_index , neg_index    = list(np.where(class_target.cpu() == 1)[0]), list(np.where(class_target.cpu() == 0)[0])
        pos_num, neg_num         = len(pos_index), len(neg_index)
        class_pred, class_target = class_pred[pos_index + neg_index], class_target[pos_index + neg_index]

        closs = F.cross_entropy(class_pred, class_target, size_average=False, reduce=False)
        closs = torch.div(torch.sum(closs), 64)

        """ regression """
        reg_pred = rout
        reg_target = targets[:, 1:]
        rloss = F.smooth_l1_loss(reg_pred, reg_target, size_average=False, reduce=False) #1445, 4
        rloss = torch.div(torch.sum(rloss, dim = 1), 4)
        rloss = torch.div(torch.sum(rloss[pos_index]), 16)

        """ mask """
        cls_score = targets[:,0].cpu()
        cls_idx = (np.array(np.where(cls_score == 1))/5).astype(int).flatten()
        loc = np.unravel_index(cls_idx, (17,17))
        pos = []
        x,y = loc
        for x,y in zip(x,y):
            if (x,y) not in pos:
                pos.append((x,y))
        # c = mask_target.view(-1)
        # BCELoss = torch.nn.BCELoss()
        mloss = 0
        # for i in range(len(pos)):
        #     m = mask[pos[i][0]*17 + pos[i][1]].view(-1)
        #     mloss += BCELoss(m,c)/(127*127)



        loss = closs + rloss + 32*mloss
        return closs, rloss, mloss, loss, reg_pred, reg_target, pos_index, neg_index

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
