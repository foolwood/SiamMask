# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import json
from os.path import exists


def load_config(args):
    assert exists(args.config), '"{}" not exists'.format(args.config)
    config = json.load(open(args.config))

    # deal with network
    if 'network' not in config:
        print('Warning: network lost in config. This will be error in next version')

        config['network'] = {}

        if not args.arch:
            raise Exception('no arch provided')
    args.arch = config['network']['arch']

    return config

