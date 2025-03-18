#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
# from model_unet import *
import config as c
from tensorboardX import SummaryWriter
# import datasets
import datasets as datasets
# import datasets_MSRS as datasets
import viz
import modules.Unet_common as common
import warnings
import logging
import util
import time
import os
from pytorch_ssim import ssim,gradient,Fusionloss

warnings.filterwarnings("ignore")
device = torch.device(f"cuda:{str(c.device_ids[0])}" if torch.cuda.is_available() else "cpu")


def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

def AG(img):  # Average gradient
    Gx, Gy = np.zeros_like(img), np.zeros_like(img)
    Gx[:, 0] = img[:, 1] - img[:, 0]
    Gx[:, -1] = img[:, -1] - img[:, -2]
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    Gy[0, :] = img[1, :] - img[0, :]
    Gy[-1, :] = img[-1, :] - img[-2, :]
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

def CC(image_F, image_A, image_B):
    rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
    rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
    return (rAF + rBF) / 2

def SCD(image_F, image_A, image_B): # The sum of the correlations of differences
    imgF_A = image_F - image_A
    imgF_B = image_F - image_B
    corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
    corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
    return corr1 + corr2

def evaluator(A, B, F):
    A = np.array(A).astype(np.float32)
    B = np.array(B).astype(np.float32)
    F = np.array(F).astype(np.float32)
    # print(A.shape, B.shape, F.shape)
    SF = np.sqrt(np.mean((F[:, 1:] - F[:, :-1]) ** 2) + np.mean((F[1:, :] - F[:-1, :]) ** 2))
    SD = np.std(F)
    AG_f = AG(F)
    CC_ALL = CC(F, A, B)
    SCD_ALL = SCD(F, A, B)
    return SD+CC_ALL+SCD_ALL, SF+AG_f

def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def rgb2ycbcr_t(img_rgb):
    R = img_rgb[:,0, :, :].unsqueeze(1)
    G = img_rgb[:,1, :, :].unsqueeze(1)
    B = img_rgb[:,2, :, :].unsqueeze(1)

    # Y = 0.257 * R + 0.504 * G + 0.098 * B + 16/255
    # Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128/255
    # Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128/255
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255
    img_ycbcr = torch.cat([Y, Cb, Cr], axis=1)
    return img_ycbcr, Y, Cb, Cr


def rgb2ycbcr_t1(img_rgb):
    R = img_rgb[0, :, :]
    G = img_rgb[1, :, :]
    B = img_rgb[2, :, :]

    # Y = 0.257 * R + 0.504 * G + 0.098 * B + 16/255
    # Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128/255
    # Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128/255
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255
    # img_ycbcr = torch.cat([Y, Cb, Cr], axis=1)
    return Y, Y, Cb, Cr

def Fro_LOSS(batchimg):
    HW =  int(batchimg.shape[2]) * int(batchimg.shape[3])
    fro_norm = torch.square(torch.norm(batchimg, dim = [2,3], p = 'fro')) / HW
    E = torch.mean(fro_norm, dim = 0)
    return E.to(device)

def L1_LOSS(A, B):
    HW = int(A.shape[2]) * int(A.shape[3])
    l1_norm = torch.norm(A-B, dim = [2,3], p = 1) / HW
    value = torch.mean(l1_norm, dim=0)
    return value.to(device)

def L1_b_LOSS(A, B):
    CHW = int(A.shape[1]) * int(A.shape[2]) * int(A.shape[3])
    l1_norm = torch.norm(A-B, dim = [1,2,3], p = 1) / CHW
    return l1_norm.to(device)

def dual_L1_loss(F, A, B):
    # F, A, B ==> torch.Size([B, C, H, W])
    A_std = A.std([1,2,3])
    B_std = B.std([1,2,3])
    A_std1 = torch.exp(A_std)/(torch.exp(A_std) + torch.exp(B_std))
    A_std2 = 1-A_std1
    A_std + B_std
    loss = torch.dot(L1_b_LOSS(F, A),A_std1) + torch.dot(L1_b_LOSS(F, B), A_std2)
    return loss.to(device)

def Grad_loss(output):
    # out_grad = torch.mean(gradient(output))
    # A_grad = torch.mean(gradient(output))
    # B_grad = torch.mean(gradient(output))
    out_grad = torch.mean(gradient(output), dim=[1, 2,3])
    out_grad = 1 - torch.mean(out_grad / (out_grad + 1.0))
    return out_grad

loss_func = Fusionloss().to(device)
#####################
# Model initialize: #
#####################
net = Model()
net.to(device)
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
para = get_parameter_number(net)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

# dwt = common.DWT()
# iwt = common.IWT()

if c.tain_next:
    load(c.MODEL_PATH + c.suffix)

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

util.setup_logger('train', './logging/', 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')

# save model path
localtime = time.strftime("%Y%m%d_%H%M", time.localtime())
param_loss = f"a{str(c.l_alpha)}_b{str(c.l_beta)}_c{str(c.l_gamma)}_d{str(c.l_ks)}"
# SAVE_PATH = c.MODEL_PATH + 'localtime/'
SAVE_PATH = c.MODEL_PATH + f'{param_loss}/{localtime}_{str(c.mse_w)}/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

logger_train.info(f"Loss param: alpha={c.l_alpha}, beta={c.l_beta}, gamma={c.l_gamma}, kesa={c.l_ks},mse_w={c.mse_w}")
# logger_train.info(net)

try:
    writer = SummaryWriter(comment='hinet', filename_suffix="steg")
    MAX_SD = 0
    MAX_AG = 0
    sum_AS = 0
    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []
        g_loss_history = []
        r_loss_history = []
        l_loss_history = []
        #################
        #     train:    #
        #################

        for i_batch, (data0, data1) in enumerate(datasets.trainloader):
            cover = data0.to(device)
            secret = data1.to(device)

            input_img = torch.cat((cover, secret), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            
            FUSION_ycbcr, FUSION_Y, FUSION_Cb, FUSION_Cr = rgb2ycbcr_t(output)
            OTHER_ycbcr, OTHER_Y, OTHER_Cb, OTHER_Cr = rgb2ycbcr_t(secret)
            MRI_Y = cover[:, 0, :, :].unsqueeze(1)  # ir
            LOSS_SSIM = 1 - ssim(FUSION_Y, MRI_Y, OTHER_Y)
            mse_chro = guide_loss(FUSION_Cb, OTHER_Cb) + guide_loss(FUSION_Cr, OTHER_Cr)
            mse_Y = guide_loss(FUSION_Y, MRI_Y) + c.mse_w * guide_loss(FUSION_Y, OTHER_Y)  # MMIF 3111-0.01
            # mse_chro = L1_LOSS(FUSION_Cb, OTHER_Cb) + L1_LOSS(FUSION_Cr, OTHER_Cr)
            # mse_Y = L1_LOSS(FUSION_Y, MRI_Y) + c.mse_w * L1_LOSS(FUSION_Y, OTHER_Y) # IR-VIS 3112-1
            grad_loss = Grad_loss(FUSION_Y)
            
            l_loss = c.l_gamma * mse_Y + c.l_ks * mse_chro
            g_loss = grad_loss
            r_loss = LOSS_SSIM
            # total_loss = c.l_alpha * r_loss + c.l_beta * g_loss +  l_loss
            """"""
            # color,gray,fusion
            # loss_in, loss_grad = loss_func(OTHER_Y, MRI_Y, FUSION_Y)
            # r_loss = loss_in
            # g_loss = loss_grad
            # l_loss = mse_chro
            # total_loss = r_loss + g_loss + l_loss
            total_loss = c.l_alpha * r_loss + c.l_beta * g_loss +  l_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])

            g_loss_history.append([g_loss.item(), 0.])
            r_loss_history.append([r_loss.item(), 0.])
            l_loss_history.append([l_loss.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        r_epoch_losses = np.mean(np.array(r_loss_history), axis=0)
        g_epoch_losses = np.mean(np.array(g_loss_history), axis=0)
        l_epoch_losses = np.mean(np.array(l_loss_history), axis=0)

        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])
        #################
        #     val:    #
        #################
        
        psnr_s = []
        psnr_c = []
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                net.eval()
                for (x0, x1) in datasets.testloader:
                    cover = x0.to(device)
                    secret = x1.to(device)
                    input_img = torch.cat((cover, secret), 1)

                    #################
                    #    forward:   #
                    #################
                    output = net(input_img)

                    #################
                    #   backward:   #
                    #################
                    output = output.cpu().numpy().squeeze() * 255
                    np.clip(output, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)

                    _, output_Y, _, _ = rgb2ycbcr_t1(output)
                    _, secret_Y, _, _ = rgb2ycbcr_t1(secret)
                    sd_et, ag_et = evaluator(secret_Y, cover, output_Y)
                    psnr_s.append(sd_et)
                    # psnr_temp = computePSNR(output, secret)
                    # psnr_s.append(psnr_temp)
                    # psnr_temp_c = computePSNR(cover, output)
                    psnr_c.append(ag_et)

                writer.add_scalars("SD+CC+SCD", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("SF+AG", {"average psnr": np.mean(psnr_c)}, i_epoch)
                logger_train.info(
                    f"TEST:   "
                    f'SD+CC+SCD: {np.mean(psnr_s):.4f} | '
                    f'SF+AG: {np.mean(psnr_c):.4f} |'
                )
        # viz.show_loss(epoch_losses)
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

        logger_train.info(
            f"Learning rate: {optim.param_groups[0]['lr']:.8f} == "
            f"Train epoch {i_epoch}:   "
            f'Loss: {epoch_losses[0].item():.4f} | '
            f'r_Loss: {r_epoch_losses[0].item():.4f} | '
            f'g_Loss: {g_epoch_losses[0].item():.4f} | '
            f'l_Loss: {l_epoch_losses[0].item():.4f} | '
        )

        if i_epoch > 600 and (MAX_SD < np.mean(psnr_s) or MAX_AG<np.mean(psnr_c) or sum_AS<(np.mean(psnr_s)+np.mean(psnr_c))) and np.mean(psnr_c)<50 and np.mean(psnr_s)<90:
            if MAX_SD < np.mean(psnr_s):
                MAX_SD = np.mean(psnr_s)
            if MAX_AG<np.mean(psnr_c):
                MAX_AG = np.mean(psnr_c)
            if sum_AS<(np.mean(psnr_s)+np.mean(psnr_c)):
                sum_AS = np.mean(psnr_s)+np.mean(psnr_c)
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, SAVE_PATH + f'model_checkpoint_{i_epoch}_{np.mean(psnr_s):.4f}_{np.mean(psnr_c):.4f}'  + '.pt')
        weight_scheduler.step()

        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, SAVE_PATH + 'model' + '.pt')
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, SAVE_PATH + 'model_ABORT' + '.pt')
    raise

finally:
    viz.signal_stop()
