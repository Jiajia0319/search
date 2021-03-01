from option import args
from utils import mkExpDir
from dataset import dataloader
from model import TTSR
from loss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
import numpy as np
warnings.filterwarnings('ignore')



import time
if __name__ == '__main__':
    ### make save_dir

    _logger = mkExpDir(args)
    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = TTSR.TTSR(args).to(device)
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### test / eval / train
    # t.load(model_path=args.model_path)

    #
    # ## 一对比较
    # s_time = time.time()
    # ref = '/home/hjj/Downloads/test/_122/465139.jpg'
    # lr_path =  '/home/hjj/Downloads/test/yy搜索测试/其他/todo/465207.png'
    # print(ref,'--------')
    # # k = 4
    # res = t.test_pair(ref,lr_path, 2)
    # print(time.time() - s_time)

    #

    # # # 文件夹search
    # for file_name in ['波点','卡通','抽象','民族风','几何','其他']:
    for file_name in ['卡通']:
        print('===========:',file_name)
        indir = '/home/hjj/Downloads/test/yy测试花型/'+file_name + '/todo'
        s_time = time.time()
        res = t.search_list(indir,file_name,'ref-4-4-conv10.npy')
        print(time.time() - s_time)

    # ## 单张search
    # s_time = time.time()
    # input_path = '/home/hjj/Downloads/test/yy搜索测试/其他/todo/465233.png'
    # print(input_path,'--------')
    # res = t.search(input_path,'ref-4-4-conv10.npy')
    # print(time.time() - s_time)


    # ## 单张ref
    # s_time = time.time()
    # input_path = '/home/hjj/Downloads/test/_122/465207.jpg'
    # print(input_path,'--------')
    # # k = 4
    # res = t.ref_one(input_path, 6)
    # print(time.time() - s_time)


    indir = '/home/hjj/Downloads/test/_122'
    inlist = os.listdir(indir)
    inlist.sort()
    k = 4 # 4: 16倍, 3: 12倍
    res = t.ref(indir, inlist,k)
    np.save('ref-4-4-conv10.npy', res)