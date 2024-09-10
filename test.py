# Please enter the absolute path of your own file.
# import sys
# main_path = r"E:/python/pytorch/Medical_image_fusion/MyMethod/MMIF_INet_code"
# sys.path.append(main_path)
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import os
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import time


cuda_my = f"cuda:{str(c.device_ids[0])}"
device = torch.device(cuda_my if torch.cuda.is_available() else "cpu")


def resize(x, size):
    transform1 = T.Compose([T.CenterCrop(size),])
    x = transform1(x)
    return x


def process(out, cb, cr):
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
    return out_img


def load(net, name):
    state_dicts = torch.load(name, map_location=cuda_my)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)
    return noise


def test(dataset_name, data_root, test_out_path):
    Time = []
    test_folder = os.path.join(data_root, dataset_name)
    test_out_folder = os.path.join(test_out_path, dataset_name)
    if not os.path.exists(test_out_folder):
        os.makedirs(test_out_folder)
    # Choose different pre-training models according to different tasks.
    if dataset_name in ['ir-vi', 'IR-VIS']:
        model_path = './model/model-VIF.pt'
    elif dataset_name in ['MRI-PET', 'MRI-SPECT']:
        model_path = './model/model-MIF.pt'

    net = Model()
    net.to(device)
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    load(net, model_path)
    net.eval()

    for img_name in os.listdir(os.path.join(test_folder, dataset_name.split('-')[0])):
        other_path = os.path.join(test_folder, dataset_name.split('-')[1], img_name)
        MRI_path = os.path.join(test_folder, dataset_name.split('-')[0], img_name)
        OTHER_img = Image.open(other_path).convert("RGB")
        MRI_img = Image.open(MRI_path).convert("RGB")

        data0 = MRI_img
        data1 = OTHER_img
        data0 = Variable(ToTensor()(data0)).unsqueeze(0)
        data1 = Variable(ToTensor()(data1)).unsqueeze(0)

        tic = time.time()
        with torch.no_grad():
            cover = data0.to(device)
            secret = data1.to(device)
            input_img = torch.cat((cover, secret), 1)
            output = net(input_img)
        end = time.time()
        Time.append(end - tic)
        torchvision.utils.save_image(output, os.path.join(test_out_folder, img_name))

        del output, secret, cover, data0, data1, MRI_img, OTHER_img
        torch.cuda.empty_cache()

    Time = Time[2:len(Time) - 2]
    return (sum(Time))


if __name__ == '__main__':
    data_root = f'./test_data'
    test_out_folder = f'./result'
    # Please pay attention! Our method is only suitable for the fusion of color images and gray images.
    dataset_name = 'ir-vi'
    # dataset_name = 'MRI-PET'
    # dataset_name = 'MRI-SPECT'
    test_time_avg = test(dataset_name, data_root, test_out_folder)
    print(test_time_avg)


