# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Jiwoong Choi (jiwoong.choi@nearthlab.com)
# --------------------------------------------------------
import os
import torch
import argparse

from tools.test import load_config, isfile, load_pretrain, siamese_init, siamese_track, join
from custom import Custom

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)

    siammask = Custom(anchors=cfg['anchors'])
    print(type(siammask))
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    outdir = './torch_scripts'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    scripted_feature_extractor = torch.jit.script(siammask.features.features)
    scripted_feature_extractor.save(os.path.join(outdir, 'feature_extractor.pt'))

    scripted_feature_downsampler = torch.jit.script(siammask.features.downsample)
    scripted_feature_downsampler.save(os.path.join(outdir, 'feature_downsampler.pt'))

    scripted_rpn_model = torch.jit.script(siammask.rpn_model)
    scripted_rpn_model.save(os.path.join(outdir, 'rpn_model.pt'))

    scripted_mask_conv_kernel = torch.jit.script(siammask.mask_model.mask.conv_kernel)
    scripted_mask_conv_kernel.save(os.path.join(outdir, 'mask_conv_kernel.pt'))

    scripted_mask_conv_search = torch.jit.script(siammask.mask_model.mask.conv_search)
    scripted_mask_conv_search.save(os.path.join(outdir, 'mask_conv_search.pt'))

    scripted_mask_depthwise_conv = torch.jit.script(siammask.mask_model.mask.dw_conv2d_group)
    scripted_mask_depthwise_conv.save(os.path.join(outdir, 'mask_depthwise_conv.pt'))

    scripted_refine_model = torch.jit.script(siammask.refine_model)
    scripted_refine_model.save(os.path.join(outdir, 'refine_model.pt'))
