from model import MainNet, LTE, SearchTransfer

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTSR(nn.Module):
    def __init__(self, args):
        super(TTSR, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, 
            res_scale=args.res_scale)
        self.LTE      = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        print(self.LTE , '==========')
    # def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
    #    ### 制作数据特征

    #     _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)  ## ref 插值缩小放大
    #
    #     refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
    #     refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)
    #     refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
    #
    #
    #     return sr, refsr_lv3_unfold, None, None, None

    def forward(self, input_tensor):
       ### 制作数据特征, *8/8 500--->64
        _, _, refsr_lv3 = self.LTE((input_tensor.detach() + 1.) / 2.)  ## ref 插值缩小放大
        return refsr_lv3

    # def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
    #    ### 制作数据特征, *8/8 500--->64
    #     _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)  ## ref 插值缩小放大
    #     return refsr_lv3, None, None, None, None

    #
    # def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
    #     ### test
    #     print('lrsr: ',lrsr.shape)
    #     _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)  ## low-re插值放大
    #
    #     lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(5, 5), padding=1)
    #     print('lrsr_lv3_unfold: ',lrsr_lv3_unfold.shape)
    #
    #     lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]
    #     print('lrsr_lv3_unfold normalize : ',lrsr_lv3_unfold.shape)
    #
    #     return sr, lrsr_lv3_unfold, None, None, None

