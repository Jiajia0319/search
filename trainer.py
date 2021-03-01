from utils import calc_psnr_and_ssim
from model import Vgg19

import os
import numpy as np
from imageio import imread, imsave
from PIL import Image
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils


class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters() if 
             args.num_gpu==1 else self.model.module.MainNet.parameters()),
             "lr": args.lr_rate
            },
            {"params": filter(lambda p: p.requires_grad, self.model.LTE.parameters() if 
             args.num_gpu==1 else self.model.module.LTE.parameters()), 
             "lr": args.lr_rate_lte
            }
        ]
        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            #model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            print('model_state_dict:-----',model_state_dict.keys())
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))

        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref = sample_batched['Ref']
            ref_sr = sample_batched['Ref_sr']
            sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)

            ### calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss = rec_loss
            if (is_print):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()) )

            if (not is_init):
                if ('per_loss' in self.loss_all):
                    sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                    with torch.no_grad():
                        hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                    per_loss = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                    loss += per_loss
                    if (is_print):
                        self.logger.info( 'per_loss: %.10f' %(per_loss.item()) )
                if ('tpl_loss' in self.loss_all):
                    sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1, 
                        S, T_lv3, T_lv2, T_lv1)
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info( 'tpl_loss: %.10f' %(tpl_loss.item()) )
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr)
                    loss += adv_loss
                    if (is_print):
                        self.logger.info( 'adv_loss: %.10f' %(adv_loss.item()) )

            loss.backward()
            self.optimizer.step()

        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        if (self.args.dataset == 'CUFED'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']

                    sr, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'.png'), sr_save)
                    
                    ### calculate psnr and ssim
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())

                    psnr += _psnr
                    ssim += _ssim

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f' %(psnr_ave, ssim_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)' 
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

        self.logger.info('Evaluation over.')



    def get_reftensor_lr(self,lr_path,k):
        ### Ref and Ref_sr
        ### LR and LR_sr
        LR_sr = imread(lr_path)[:,:,:3]
        h1, w1 = LR_sr.shape[:2]
        LR_sr = np.array(Image.fromarray(LR_sr).resize((int(w1//k), int(h1//k)), Image.BICUBIC))
        LR_sr = LR_sr / 127.5 - 1.
        Ref_sr_t = torch.from_numpy(LR_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

        Ref_sr_t = self.model(Ref_sr_t)
        # print('Ref_sr_t shape', Ref_sr_t.shape)

        return Ref_sr_t

    def get_reftensor_ref(self,Ref,k):
        ### Ref and Ref_sr
        # Ref = imread(refpath)
        # print('Ref',Ref.shape)
        Ref = Ref[:,:,:3]
        # Image.fromarray(Ref.astype('uint8')).save('test.png')
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//k*k, w2//k*k
        Ref = Ref[:h2, :w2, :]

        w3 = w2//k
        h3 = h2//k

        if w3 < 4 or h3 < 4:
            w3 = w2
            h3 = h2

        Ref_sr = np.array(Image.fromarray(Ref).resize((w3, h3), Image.BICUBIC))
        # print(Ref_sr.shape)
        Ref_sr = Ref_sr.astype(np.float16)
        Ref_sr = Ref_sr / 127.5 - 1.

        Ref_sr_t = torch.from_numpy(Ref_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        return Ref_sr_t

    def ref(self, indir, inlist,k):
        self.model.eval()
        res = dict()
        for i, refpath in enumerate(inlist):
            print(i, refpath)
            im = Image.open(os.path.join(indir, refpath))
            # im = ref_padding(im)
            # im.save('tmp.png')
            w,h = im.size
            if w < 40 or h < 40:
                print(i, refpath,'=============================')
                continue
            with torch.no_grad():
                Ref_sr_t = self.get_reftensor_ref(np.array(im),k)
                re = self.model(Ref_sr_t)
                # re = re[:,:64,:,:]
                # re = torch.sum(re, 1, True)
                refsr_lv3_unfold = re.squeeze().cpu().numpy()
                torch.cuda.empty_cache()
            refsr_lv3_unfold = refsr_lv3_unfold*127.5
            res[refpath] = refsr_lv3_unfold.astype('uint8')
        return res
    #
    # def ref_with_unfold(self, indir, inlist,k):
    #     self.model.eval()
    #     res = dict()
    #     for i, refpath in enumerate(inlist):
    #         print(i, refpath)
    #         im = Image.open(os.path.join(indir, refpath))
    #         w,h = im.size
    #         if w < 4 or h < 4:
    #             print(i, refpath,'=============================')
    #             continue
    #         with torch.no_grad():
    #             Ref_sr_t = self.get_reftensor_ref(os.path.join(indir, refpath),k)
    #             re = self.model(Ref_sr_t)
    #             re = get_ref_unfold(re)
    #             re = torch.sum(re, 1, True)
    #
    #             refsr_lv3_unfold = re.squeeze().cpu().numpy()
    #             torch.cuda.empty_cache()
    #         refsr_lv3_unfold = refsr_lv3_unfold*127.5
    #         res[refpath] = refsr_lv3_unfold.astype('uint8')
    #     return res

    def ref_one(self, inpath,k):
        self.model.eval()
        im = Image.open(inpath)
        im = ref_padding(im)
        w,h = im.size
        if w < 4 or h < 4:
            print(w,h,inpath,'=============================')
        with torch.no_grad():
            Ref_sr_t = self.get_reftensor_ref(np.array(im),k)
            re = self.model(Ref_sr_t)
            refsr_lv3_unfold = re.squeeze().cpu().numpy()
            torch.cuda.empty_cache()
        refsr_lv3_unfold = refsr_lv3_unfold*127.5
        print('refsr_lv3_unfold',refsr_lv3_unfold.shape)
        return refsr_lv3_unfold

    def test_pair(self, inpath,lr_path,k):
        self.model.eval()
        im = Image.open(inpath)
        # im = ref_padding(im)
        w,h = im.size
        if w < 4 or h < 4:
            print(w,h,inpath,'=============================')
        with torch.no_grad():
            Ref_sr_t = self.get_reftensor_ref(np.array(im),k)
            re = self.model(Ref_sr_t)
            ref = get_ref_unfold(re)
            lr = self.get_reftensor_lr(lr_path,1.5)
            lr = get_lr_unfold(lr)

            print(ref.shape, lr.shape)

            R_lv3 = torch.bmm(ref, lr)  # [N, Hr*Wr, H*W]
            R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]  h*w原图上找到的在ref上最相似的那个
        R_lv3_star = R_lv3_star.squeeze().cpu().numpy()
        print('0.5:',len(R_lv3_star[R_lv3_star > 0.5]) / len(R_lv3_star))
        print('0.6:',len(R_lv3_star[R_lv3_star > 0.6]) / len(R_lv3_star))
        print('0.7:',len(R_lv3_star[R_lv3_star > 0.7]) / len(R_lv3_star))


        R_lv3_star = sorted(R_lv3_star, reverse=True)

        print('R_lv3_star',np.mean(R_lv3_star[:100]),np.mean(R_lv3_star[:50]),'-----',R_lv3_star[:10])


    def search_list(self, indir,file_name,ref_feature):
        self.logger.info('Test process...')
        self.model.eval()
        data = np.load(ref_feature, allow_pickle=True)
        data = data[()]
        search_res = dict()
        inlist = os.listdir(indir)
        inlist.sort()
        for imgpath in inlist[:1]:
            lrpath = os.path.join(indir, imgpath)
            lrid = lrpath.split('/')[-1].split('.')[0]
            t0 = time.time()

            lr1 = self.get_reftensor_lr(lrpath, 1.5)
            lr2 = self.get_reftensor_lr(lrpath, 2)


            # lr = self.get_reftensor_lr(lrpath, 1)
            # lr1 = F.interpolate(lr, scale_factor=1 / 1.5, mode='bilinear', align_corners=True)
            # lr2 = F.interpolate(lr, scale_factor=1 / 2, mode='bilinear', align_corners=True)
            # print('get_reftensor_lr --------- : time ', time.time() - t0)
            t1 = time.time()

            tmp_res1,tmp1_sim = get_scale_res(data, lr1,lr2, lrid)
            # print('search 1 --------- : time ', time.time() - t1)

            tmp_res = [tmp_res1,tmp1_sim]
            search_res[lrid] = tmp_res
            print('-----lrid:{}, -----rank: {} ----, ------sim: {}'.format(lrid , tmp_res1,tmp1_sim))

        np.save('search_res_16_padding' + file_name + '.npy',search_res)
        return search_res

    def search(self, lrpath,datapath):
        self.logger.info('Test process...')
        self.model.eval()
        data = np.load(datapath, allow_pickle=True)
        data = data[()]
        lrid = lrpath.split('/')[-1].split('.')[0]
        lr1 = self.get_reftensor_lr(lrpath, 1.5)
        lr2 = self.get_reftensor_lr(lrpath, 2)

        # lr = self.get_reftensor_lr(lrpath, 1)
        # lr1 = F.interpolate(lr, scale_factor=1/1.5, mode='bilinear', align_corners=True)
        # lr2 = F.interpolate(lr, scale_factor=1/2, mode='bilinear', align_corners=True)


        tmp_res1, tmp1_sim = get_single_res(data, lr1, lrid)   #  [b,c],[res_40_sim[b],res_50_sim[c]]
        print('lrid-----:', lrid, tmp_res1, tmp1_sim)
        tmp_res2, tmp2_sim = get_single_res(data, lr2, lrid)
        print('lrid-----:', lrid, tmp_res2, tmp2_sim)


        ### 阀值0.5
        if tmp2_sim[0] > tmp1_sim[0]:
            print('res==0.5==== sorted:',tmp_res2[0],'sim:', tmp2_sim[0])
        else:
            print('res===0.5=== sorted:',tmp_res1[0],'sim:', tmp1_sim[0])
        ### 阀值0.6
        if tmp2_sim[1] > tmp1_sim[1]:
            print('res===0.6=== sorted:',tmp_res2[1],'sim:', tmp2_sim[1])
        else:
            print('res===0.6=== sorted:',tmp_res1[1],'sim:', tmp1_sim[1])


        # if tmp2_sim > tmp1_sim:
        #     print('res====== sorted:',tmp_res2,'sim:', tmp2_sim)
        # else:
        #     print('res====== sorted:',tmp_res1,'sim:', tmp1_sim)



        # tmp_res1, tmp1_sim = get_scale_res(data, lr1, lr2, lrid)
        # print('lrid-----:', lrid, tmp_res1, tmp1_sim)

        # tmp_res = get_single_res(data, l2, lrid)
        # print('lrid-----:', lrid, tmp_res)


def get_res(res,lrid):
    res = sorted(res, key=lambda x: x[1], reverse=True)  # 按照某一列进行排序的方法
    res_20 = [i[0].split('.')[0] for i in res]
    res_20_sim = [i[1] for i in res]
    res_20.append(lrid)
    res_20_sim.append(0)
    return res_20,res_20_sim


def get_ref_unfold(refsr_lv3):
    refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
    refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)
    refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
    return refsr_lv3_unfold


def get_lr_unfold(lrsr_lv3):
    lrsr_lv3_unfold = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
    lrsr_lv3_unfold = F.normalize(lrsr_lv3_unfold, dim=1)  # [N, C*k*k, H*W]

    # lrsr_lv3_unfold = torch.sum(lrsr_lv3_unfold, 2, True)
    # lrsr_lv3_unfold = lrsr_lv3_unfold.squeeze(0)

    return lrsr_lv3_unfold


def get_R_lv3_star(ref,lr):
    R_lv3 = torch.bmm(ref, lr)  # [N, Hr*Wr, H*W]
    R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]  h*w原图上找到的在ref上最相似的那个
    R_lv3_star = R_lv3_star.squeeze().cpu().numpy()
    return len(R_lv3_star[R_lv3_star > 0.6]) / len(R_lv3_star)


def get_res_top20(res1,res2,lrid):
    res = res1+res2
    res = sorted(res, key=lambda x: x[1], reverse=True)  # 按照某一列进行排序的方法
    res_20 = [i[0].split('.')[0] for i in res]
    res_20_sim = [i[1] for i in res]
    res_20.append(lrid)
    res_20_sim.append(0)
    return res_20,res_20_sim



def get_scale_res(data,lr1,lr2,lrid):
    res_1 = []
    res_2 = []
    lr1 = get_lr_unfold(lr1)
    lr2 = get_lr_unfold(lr2)
    for datakey in data:
        with torch.no_grad():
            ref = torch.from_numpy(data[datakey] / 127.5).unsqueeze(0).float().to('cuda')
            # print('ref shape',ref.shape)
            try:
                ref = get_ref_unfold(ref)
            except:
                continue
            R1 = get_R_lv3_star(ref, lr1)
            R2 = get_R_lv3_star(ref, lr2)
            res_1.append([datakey, R1])
            res_2.append([datakey, R2])

    # res1,res1_sim = get_res(res_1, lrid)
    # res2,res2_sim = get_res(res_2, lrid)
    # a = res1.index(lrid)
    # b = res2.index(lrid)
    # if res1_sim[a] > res2_sim[b]:
    #     return [a,res1_sim[a]]
    # else:
    #     return [b, res2_sim[b]]

    res1,res1_sim  = get_res_top20(res_1,res_2, lrid)
    a = res1.index(lrid)
    # print(a,res1_sim[a])
    return a,res1_sim[a]


def get_single_res(data,lr,lrid):

    res_30 = []
    res_35 = []
    res_50 = []
    res_40 = []
    res_60 = []
    res_R_lv3_star = []

    lr = get_lr_unfold(lr)
    for datakey in data:
        with torch.no_grad():
            ref = torch.from_numpy(data[datakey] / 127.5).unsqueeze(0).float().to('cuda')
            # print('ref shape',ref.shape)
            try:
                ref = get_ref_unfold(ref)

            except:
                continue
            # print(ref.shape,lr.shape)
            # ref = ref.squeeze(0)   ## ref_unfold
            # lr = lr.squeeze(-1)
            #

            # print('unfold shape', ref.shape,lr.shape)
            # R_lv3_star = torch.dot(ref, lr)
            R_lv3 = torch.bmm(ref, lr)  # [N, Hr*Wr, H*W]
            # # print(ref.shape,lr.shape, R_lv3.shape)
            R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]  h*w原图上找到的在ref上最相似的那个
        R_lv3_star = R_lv3_star.squeeze().cpu().numpy()

        # # rate = len(R_lv3_star[R_lv3_star > 0.5]) / len(R_lv3_star)
        # res_30.append([datakey, len(R_lv3_star[R_lv3_star > 0.4]) / len(R_lv3_star)])
        # # res_35.append([datakey, len(R_lv3_star[R_lv3_star > 0.35]) / len(R_lv3_star)])
        res_40.append([datakey, len(R_lv3_star[R_lv3_star > 0.5]) / len(R_lv3_star)])
        res_50.append([datakey, len(R_lv3_star[R_lv3_star > 0.6]) / len(R_lv3_star)])
        # res_60.append([datakey, len(R_lv3_star[R_lv3_star > 0.6]) / len(R_lv3_star)])

        # R_lv3_star = sorted(R_lv3_star, reverse=True)
        # res_R_lv3_star.append([datakey, np.mean(R_lv3_star[:100])])
        # res_40.append([datakey, np.mean(R_lv3_star[:100])])
        # res_50.append([datakey, np.mean(R_lv3_star[:100])])

    # res_30,res_30_sim = get_res(res_30, lrid)
    res40,res_40_sim = get_res(res_40, lrid)
    res50,res_50_sim = get_res(res_50, lrid)
    # res60,res_60_sim = get_res(res_60, lrid)

    # res_R_lv3_star,res_R_lv3_star_sim = get_res(res_R_lv3_star, lrid)
    # a = res_30.index(lrid)
    b = res40.index(lrid)
    c = res50.index(lrid)
    # d = res60.index(lrid)

    # d = res_R_lv3_star.index(lrid)
    # return [a,b,c,d],[res_30_sim[a],res_40_sim[b],res_50_sim[c],res_60_sim[d]]
    return [b,c],[res_40_sim[b],res_50_sim[c]]

#
# from multiprocessing import Pool
#
#
# def do_R_lv3_star(data,datakey,lr,res_30,res_40,res_50):
#     with torch.no_grad():
#         ref = torch.from_numpy(data[datakey] / 127.5).unsqueeze(0).float().to('cuda')
#         try:
#             ref = get_ref_unfold(ref)
#         except:
#             return res_30, res_40, res_50, False
#         R_lv3 = torch.bmm(ref, lr)  # [N, Hr*Wr, H*W]
#         R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]  h*w原图上找到的在ref上最相似的那个
#     R_lv3_star = R_lv3_star.squeeze().cpu().numpy()
#
#
#     return R_lv3_star,datakey,True


# def get_single_res(data,lr,lrid):
#     ### multiprocessing
#     res_30 = []
#     res_50 = []
#     res_40 = []
#     _process = []
#     pool = Pool(2)
#     for datakey in data:
#         t = pool.apply_async(func=do_R_lv3_star, args=(data, datakey, lr, res_30, res_40, res_50,))
#         _process.append(t)
#     pool.close()
#     pool.join()
#     print('{} images'.format(len(_process) - 1))
#
#     for t in _process:
#         R_lv3_star,datakey, ifadd = t.get()
#         if ifadd:
#             res_30.append([datakey, len(R_lv3_star[R_lv3_star > 0.3]) / len(R_lv3_star)])
#             res_40.append([datakey, len(R_lv3_star[R_lv3_star > 0.4]) / len(R_lv3_star)])
#             res_50.append([datakey, len(R_lv3_star[R_lv3_star > 0.5]) / len(R_lv3_star)])
#
#     res_30,res_30_sim = get_res(res_30, lrid)
#     res40,res_40_sim = get_res(res_40, lrid)
#     res50,res_50_sim = get_res(res_50, lrid)
#
#     a = res_30.index(lrid)
#     b = res40.index(lrid)
#     c = res50.index(lrid)
#     return [a,b,c],[res_30_sim[a],res_40_sim[b],res_50_sim[c]]


def ref_padding(im):
    w, h = im.size
    # im_new = Image.new('RGB', (int(1.75 * w), int(1.75 * h)), (0, 0, 0, 0))
    # im_new.paste(im, (0, 0))
    # im_new.paste(im, (w, 0))
    # im_new.paste(im, (0, h))
    # im_new.paste(im, (w, h))
    # ref = im_new.crop((int(0.25 * w), int(0.25 * h), int(1.75 * w), int(1.75 * h)))

    a = 1.1
    im_new = Image.new('RGB', (int(a * w), int(a * h)), (0, 0, 0, 0))
    a0 = a - 1
    a1 = 1 - a0

    im_corner4 = im.crop((int(a1 * w), int(a1 * h), w, h))
    im_crop2 = im.crop((int(a1 * w), 0, w, h))
    im_crop4 = im.crop((0, int(a1 * h), w, h))
    im_new.paste(im_corner4, (0, 0))
    im_new.paste(im_crop2, (0, int(a0 * h)))
    im_new.paste(im_crop4, (int(a0 * w), 0))
    im_new.paste(im, (int(a0 * w), int(a0 * h)))

    return im_new