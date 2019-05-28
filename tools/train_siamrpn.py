# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import argparse
import logging
import os
import shutil
import time
import torch
from torch.utils.data import DataLoader

from utils.log_helper import init_log, print_speed, add_file_handler, Dummy
from utils.load_helper import load_pretrain, restore_from
from utils.average_meter_helper import AverageMeter

from datasets.siam_rpn_dataset import DataSets
import models as models
import math

from utils.lr_helper import build_lr_scheduler
from tensorboardX import SummaryWriter

from utils.config_helper import load_config
import json
import cv2
from torch.utils.collect_env import get_pretty_env_info

torch.backends.cudnn.benchmark = True

model_zoo = sorted(name for name in models.__dict__
            if not name.startswith("__")
            and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Tracking Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip', default=10.0, type=float,
                    help='gradient clip value')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of SiamRPN in json format')
parser.add_argument('--arch', dest='arch', default='', choices=model_zoo + ['Custom',''],
                    help='architecture of pretrained model')
parser.add_argument('-l', '--log', default="log.txt", type=str,
                    help='log file')
parser.add_argument('-s', '--save_dir', default='snapshot', type=str,
                    help='save dir')
parser.add_argument('--log-dir', default='board', help='TensorBoard log dir')

best_acc = 0.


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += "\n        OpenCV ({})".format(cv2.__version__)
    return env_str


def build_data_loader(cfg):
    logger = logging.getLogger('global')

    logger.info("build train dataset")  # train_dataset
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'], args.epochs)
    train_set.shuffle()

    logger.info("build val dataset")  # val_dataset
    if not 'val_datasets' in cfg.keys():
        cfg['val_datasets'] = cfg['train_datasets']
    val_set = DataSets(cfg['val_datasets'], cfg['anchors'])
    val_set.shuffle()

    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=True, sampler=None)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=True, sampler=None)

    logger.info('build dataset done')
    return train_loader, val_loader


def build_opt_lr(model, cfg, args, epoch):
    trainable_params = model.features.param_groups(cfg['lr']['start_lr'], cfg['lr']['feature_lr_mult']) + \
            model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult'])

    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = build_lr_scheduler(optimizer, cfg['lr'], epochs=args.epochs)

    lr_scheduler.step(epoch)

    return optimizer, lr_scheduler


def main():
    global args, best_acc, tb_writer, logger
    args = parser.parse_args()

    init_log('global', logging.INFO)

    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info(args)

    cfg = load_config(args)

    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
    
    logger.info("\n" + collect_env_info())

    if args.log_dir:
        tb_writer = SummaryWriter(args.log_dir)
    else:
        tb_writer = Dummy()

    # build dataset
    train_loader, val_loader = build_data_loader(cfg)

    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(pretrain=True, anchors=cfg['anchors'])
    else:
        model = models.__dict__[args.arch](anchors=cfg['anchors'])

    logger.info(model)

    if args.pretrained:
        model = load_pretrain(model, args.pretrained)

    model = model.cuda()
    dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()

    if args.resume and args.start_epoch != 0:
        model.features.unfix((args.start_epoch - 1) / args.epochs)

    optimizer, lr_scheduler = build_opt_lr(model, cfg, args, args.start_epoch)
    logger.info(lr_scheduler)
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch, best_acc, arch = restore_from(model, optimizer, args.resume)
        dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()
        epoch = args.start_epoch
        if dist_model.module.features.unfix(epoch/args.epochs):
            logger.info('unfix part model.')
            optimizer, lr_scheduler = build_opt_lr(dist_model.module, cfg, args, epoch)
        lr_scheduler.step(epoch)
        cur_lr = lr_scheduler.get_cur_lr()
        logger.info('epoch:{} resume lr {}'.format(epoch, cur_lr))

    logger.info('model prepare done')

    train(train_loader, dist_model, optimizer, lr_scheduler, args.start_epoch, cfg)


def train(train_loader, model, optimizer, lr_scheduler, epoch, cfg):
    global tb_index, best_acc, cur_lr
    cur_lr = lr_scheduler.get_cur_lr()
    logger = logging.getLogger('global')
    avg = AverageMeter()
    model.train()
    model = model.cuda()
    end = time.time()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    num_per_epoch = len(train_loader.dataset) // args.epochs // args.batch
    start_epoch = epoch
    epoch = epoch
    for iter, input in enumerate(train_loader):
        # next epoch
        if epoch != iter // num_per_epoch + start_epoch:
            epoch = iter // num_per_epoch + start_epoch

            if not os.path.exists(args.save_dir):  # makedir/save model
                os.makedirs(args.save_dir)

            save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'anchor_cfg': cfg['anchors']
                }, False,
                os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch)),
                os.path.join(args.save_dir, 'best.pth'))

            if epoch == args.epochs:
                return

            if model.module.features.unfix(epoch/args.epochs):
                logger.info('unfix part model.')
                optimizer, lr_scheduler = build_opt_lr(model.module, cfg, args, epoch)

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()

            logger.info('epoch:{}'.format(epoch))

        tb_index = iter
        if iter % num_per_epoch == 0 and iter != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info("epoch {} lr {}".format(epoch, pg['lr']))
                tb_writer.add_scalar('lr/group%d'%(idx+1), pg['lr'], tb_index)

        data_time = time.time() - end
        avg.update(data_time=data_time)
        x = {
            'cfg': cfg,
            'template': torch.autograd.Variable(input[0]).cuda(),
            'search': torch.autograd.Variable(input[1]).cuda(),
            'label_cls': torch.autograd.Variable(input[2]).cuda(),
            'label_loc': torch.autograd.Variable(input[3]).cuda(),
            'label_loc_weight': torch.autograd.Variable(input[4]).cuda(),
        }

        optimizer.zero_grad()
        outputs = model(x)

        rpn_cls_loss, rpn_loc_loss = outputs['losses']

        rpn_cls_loss, rpn_loc_loss = torch.mean(rpn_cls_loss), torch.mean(rpn_loc_loss)

        cls_weight, reg_weight = cfg['loss']['weight']

        loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight

        loss.backward()

        if cfg['clip']['split']:
            torch.nn.utils.clip_grad_norm_(model.module.features.parameters(), cfg['clip']['feature'])
            torch.nn.utils.clip_grad_norm_(model.module.rpn_model.parameters(), cfg['clip']['rpn'])
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # gradient clip

        siamrpn_loss = loss.item()

        if is_valid_number(siamrpn_loss):
            optimizer.step()

        batch_time = time.time() - end

        avg.update(batch_time=batch_time, rpn_cls_loss=rpn_cls_loss,
                   rpn_loc_loss=rpn_loc_loss, siamrpn_loss=siamrpn_loss)

        tb_writer.add_scalar('loss/cls', rpn_cls_loss, tb_index)
        tb_writer.add_scalar('loss/loc', rpn_loc_loss, tb_index)

        end = time.time()

        if (iter + 1) % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}] lr: {lr:.6f}\t{batch_time:s}\t{data_time:s}'
                        '\t{rpn_cls_loss:s}\t{rpn_loc_loss:s}\t{siamrpn_loss:s}'.format(
                        epoch+1, (iter + 1) % num_per_epoch, num_per_epoch, lr=cur_lr,
                        batch_time=avg.batch_time, data_time=avg.data_time, rpn_cls_loss=avg.rpn_cls_loss,
                        rpn_loc_loss=avg.rpn_loc_loss, siamrpn_loss=avg.siamrpn_loss))
            print_speed(iter + 1, avg.batch_time.avg, args.epochs * num_per_epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_file='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)


if __name__ == '__main__':
    main()
