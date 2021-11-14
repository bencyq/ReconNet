"""
没有用测试集
直接两次loss相差0.000001就停止
或者step超过10000
"""

import cv2 as cv
import numpy as np
import os
import torch
from NET import Net
import random
import math
import time
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import random_noise


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def train_data(batch_size, step, path_file_original, Phi):
    file_nameList_original = os.listdir(path_file_original)
    hei = len(file_nameList_original)
    hei_list = np.linspace(0, hei - 1, hei).astype(int)
    get_original = []
    get_cs = []
    if step == 0:
        random.shuffle(hei_list)
        print(hei_list)
    for i in range(step * batch_size, (step + 1) * batch_size):
        image = cv.imread(path_file_original + '\\' + file_nameList_original[hei_list[i]], 0)
        image = image / 255.0
        cs = np.dot(Phi, np.reshape(image, [-1, 1]))
        get_original.append(image)
        get_cs.append(cs.squeeze())
    return np.expand_dims(np.array(get_original), axis=1), np.array(get_cs)


def dev(path_file_original, model, device, batchSize, Phi):
    """
    训练时用来判断是否保存参数
    """
    model.eval()
    loss_fuction = nn.MSELoss()
    mse_sum = 0

    for step in range(0, 5):
        get_original, get_cs = train_data(batchSize, step, path_file_original, Phi)
        original_torch = torch.Tensor(get_original).to(device)
        cs_torch = torch.Tensor(get_cs).to(device)
        with torch.no_grad():
            output = model(cs_torch)
            loss = loss_fuction(output, original_torch)
            mse_sum += loss
    mean_mse = mse_sum / 5
    return mean_mse


def train(path_file_train, model, config, device, Phi):
    file_nameList_original = os.listdir(path_file_train)
    hei = len(file_nameList_original)

    if os.path.exists(config['save_path']):  # 权重文件在不在
        model.load_state_dict(torch.load(config['save_path']))
        print('存在')

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fuction = nn.MSELoss()

    ps_q = 100
    num = 0
    register = 10

    file_nameList_original = os.listdir(path_file_train)
    hei = len(file_nameList_original)
    for epoches in range(0, config['cycle_index']):
        print('轮数：', epoches)
        for step in range(0, int(math.floor(hei / config['batch_size']))):
            num += 1
            original_date, cs_date = train_data(config['batch_size'], step, path_file_train, Phi)
            original_torch = torch.Tensor(original_date).to(device)
            cs_torch = torch.Tensor(cs_date).to(device)
            model.train()
            output = model(cs_torch)
            loss = loss_fuction(original_torch, output)

            if num % 20 == 0:
                mean_loss = dev(path_file_train, model, device, config['batch_size'], Phi)
                if mean_loss < ps_q:
                    ps_q = mean_loss
                    print(num)
                    print('保存模型参数')
                    torch.save(model.state_dict(), config['save_path'])

                    for param_group in optimizer.param_groups:  # 改变学习率
                        config['lr'] = config['lr'] * 0.999
                        param_group['lr'] = config['lr']
                        print('lr', config['lr'], 'mean_loss', mean_loss)

                # torch.save(model.state_dict(), config['save_path'])
                # for param_group in optimizer.param_groups:  # 改变学习率
                #     config['lr'] = config['lr'] * 0.999
                #     param_group['lr'] = config['lr']
                #     print('lr:', config['lr'], 'loss', loss)

            if abs(loss - register) < 0.000001:
                print('两次loss不在减小训练结束')
                # exit(0)
                return
            register = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # 参数更新
            # loss = loss / config['accumulation_steps']
            # loss.backward()
            # if (num % config['accumulation_steps']) == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()

            if num % config['early_stop'] == 0 and num != 0:
                print('完成训练次数')
                exit(0)


def test(img, model, device, config, Phi):
    """
    整块测试
    """
    model.eval()
    if device == 'cpu':
        model.load_state_dict(torch.load(config['save_path'], map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(config['save_path']))

    [height, width] = img.shape

    t_h = np.ceil(height / 33).astype(int)  # 向上取整
    t_w = np.ceil(width / 33).astype(int)
    print('t_h,t_w', t_h, t_w)
    imag_block = cv.copyMakeBorder(img, 0, t_h * 33 - height, 0, t_w * 33 - width, cv.BORDER_DEFAULT)  # 翻转扩充

    cs_list = np.zeros([t_h * t_w, 272], dtype=float)
    num = 0
    for i in range(t_h):
        for j in range(t_w):
            image_list = imag_block[(i * 33):(i * 33 + 33), (j * 33):(j * 33 + 33)]
            a = np.dot(Phi, np.reshape(image_list, [1089, 1]))
            cs_list[num, :] = a[:, 0]
            num += 1

    # 测试鲁棒性
    # max_cs = np.max(cs_list)
    # min_cs = np.min(cs_list)
    # normalized_cs = (cs_list-min_cs)/(max_cs-min_cs)
    # # cs_noise = random_noise(normalized_cs,'gaussian', mean=0, var=0.01)
    # cs_noise = random_noise(normalized_cs, 's&p', amount = 0.001)
    # # cs_noise = random_noise(normalized_cs, 'speckle', mean=0, var=0.01)
    # cs_list = cs_noise * (max_cs-min_cs) + min_cs

    cs_torch = torch.Tensor(cs_list).to(device)
    with torch.no_grad():
        output = model(cs_torch)
    output = output.cpu().detach().numpy()
    print('output1.shape:', output.shape)

    num = 0
    new_img = np.zeros([t_h * 33, t_w * 33], dtype=float)
    for i in range(t_h):
        for j in range(t_w):
            new_img[(i * 33):(i * 33 + 33), (j * 33):(j * 33 + 33)] = output[num][0]
            num += 1
    new_img = new_img[:height, :width]
    psnr1 = psnr(np.uint8(img * 255), np.uint8(new_img * 255))
    ssim1 = ssim(np.uint8(img * 255), np.uint8(new_img * 255))

    return psnr1, ssim1, new_img, cs_list



np.random.seed(1)
Phi = np.random.randn(272, 1089)
for j in range(272):
    Phi[j] = Phi[j] / np.linalg.norm(Phi[j])

path_file_train = r'D:\py-project\ReconNet\train_image'
# path_file_dev = '/disks/disk0/jsw/python cod/val_imge_64'
device = get_device()
config = {
    'cycle_index': 3000,  # maximum number of epochs
    'batch_size': 64,  # mini-batch size for dataloader
    'lr': 0.0001,  # learning rate of SGD
    'early_stop': 100000,  # early stopping epochs (the number epochs since your model's last improvement)
    # 'accumulation_steps': 8,
    'save_path': 'model5_quantitative.pth',
    # 'save_path': 'model5_noise.pth',
}

model = Net().to(device)
# train(path_file_train, model, config, device, Phi)



img = cv.imread('test_images11\\flinstones.png', 0) / 255.0
time1 = time.time()
psnr, ssim, new_img, cs_list = test(img, model, device, config, Phi)
time2 = time.time()
print('psnr:', psnr)
print('ssim:', ssim)
print('time:', time2 - time1)

cv.imshow('img', img)
cv.imshow('cs_list', cs_list)
cv.imshow('rimg', new_img)

cv.waitKey(0)

