from DeepFool.Python.deepfool import deepfool
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd.gradcheck import zero_gradients
import math
from PIL import Image
import torchvision.models as models
import sys
import random
import time
from tqdm import tqdm
from resnet import ResNet


def get_model(model, DEVICE):
    if model == 'densenet10':
        net = torch.load("./models/{}.pth".format(model))
    elif model == 'resnet_test':
        net = ResNet()
        net.load_state_dict(torch.load("./models/{}.pth".format(model)))

    net.eval()
    net = net.to(DEVICE)
    return net




def data_input_init(xi):
    mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
    std = [63.0 / 255, 62.1 / 255.0, 66.7 / 255.0]

    transform = transforms.Compose([
        transforms.Resize(32,32),
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])

    return (mean, std, transform)



def get_data_loader(data_name, tf):
    if data_name == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf)
        data_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                               shuffle=True, num_workers=2)
    else :
        testsetout = torchvision.datasets.ImageFolder("./data/{}".format(data_name), transform=tf)
        data_loader = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                                shuffle=True, num_workers=2)

    return data_loader


'''

def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    if p==np.inf:
            v=torch.clamp(v,-xi,xi)
    else:
        v=v * min(1, xi/(torch.norm(v,p)+0.00001))
    return v
    
    
def get_fooling_rate(data_loader, v, model, DEVICE):
    """
    :data_list: list of image paths
    :v: Noise Matrix
    :model: target network
    :device: PyTorch device
    """

    fooled = 0.0

    for batch in tqdm(data_loader):

        image, _ = batch

        image = image.to(DEVICE)
        #label = label.to(DEVICE)

        _, pred = torch.max(model(image), 1)
        _, adv_pred = torch.max(model(image + v), 1)

        if (pred != adv_pred):
            fooled += 1

    # Compute the fooling rate
    fooling_rate = fooled / len(data_loader)
    print('Fooling Rate = ', fooling_rate)

    for param in model.parameters():
        param.requires_grad = False

    return fooling_rate, model



def universal_adversarial_perturbation(data_loader, model, DEVICE, xi=10, delta=0.2, max_iter_uni = 10, p=np.inf,
                                       num_classes=10, overshoot=0.02, max_iter_df=10 ,t_p = 0.2):
    """
    :data_list: list of image paths
    :model: target network
    :device: PyTorch Device
    :param xi: controls the l_p magnitude of the perturbation
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = 10*num_images)
    :param p: norm to be used (default = np.inf)
    :param num_classes: For deepfool: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: For deepfool: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df:For deepfool: maximum number of iterations for deepfool (default = 10)
    :param t_p:For deepfool: truth perentage, for how many flipped labels in a batch.(default = 0.2)

    :return: the universal perturbation matrix.
    """
    time_start = time.time()
    mean, std ,tf = data_input_init(xi)

    v = torch.zeros(1 ,3 ,32 ,32).to(DEVICE)

    v.requires_grad_()


    fooling_rate = 0.0

    itr = 0

    while fooling_rate < 1- delta and itr < max_iter_uni:

        for k, batch in enumerate(tqdm(data_loader)):
            image, _ = batch

            image = image.to(DEVICE)
            # label = label.to(DEVICE)

            # img = img.unsqueeze(0)

            _, pred = torch.max(model(image), 1)
            _, adv_pred = torch.max(model(image + v), 1)

            if (pred == adv_pred):
                dr, iter, _, _, _ = deepfool((image + v).detach()[0], model, DEVICE, num_classes=num_classes,
                                             overshoot=overshoot, max_iter=max_iter_df)
                if (iter < max_iter_df - 1):
                    v = v + torch.from_numpy(dr).to(DEVICE)
                    v = proj_lp(v, xi, p)

            
#            if (k % 10 == 0):
#                pbar.set_description('Norm of v: ' + str(torch.norm(v).detach().cpu().numpy()))
            

        fooling_rate, model = get_fooling_rate(data_loader, v, model, DEVICE)
        print(fooling_rate)
        itr = itr + 1

    return v

'''


def train_noise(model, data_loader, noise, optimizer, epoch, temper):
    model.eval()

    for batch in tqdm(data_loader, desc = 'noise generating epoch{}'.format(epoch)) :
        image, label = batch
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        logit = model(image + noise)
        logit = logit/temper
        logit = logit - logit.max()

        loss = F.cross_entropy(logit, label)

        loss.backward()

        optimizer.step()


    return noise





DEVICE=('cuda' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)


model = get_model('densenet10', DEVICE)
#model = get_model('resnet_test', DEVICE)

data_loader_test = get_data_loader('cifar10', tf)



####노이즈를 생성하는 과정###

lr_list = [0.001]
temper = 500

for learning_rate in lr_list :

    noise = torch.zeros(1, 3, 32, 32).to(DEVICE)
    optimizer = optim.Adam([noise], lr=learning_rate)
    noise.requires_grad = True

    for epoch in range(500):
        print("epoch : ", epoch)
        noise = train_noise(model, train_loader, noise, optimizer, epoch, temper)

    torch.save(noise,'./noise/epoch500/densenet_noise_test_lr{}_T{}.pth'.format(learning_rate, temper))



###생성된 노이즈를 여러 데이터셋에 적용하여 결과를 출력하는 과정###
'''

data_list = ['cifar10', 'Imagenet', 'Imagenet_resize', 'iSUN', 'LSUN', 'LSUN_resize']

temper = 500
#noise_test=torch.load('./noise/densenet_noise_test_T{}.pth'.format(temper))

for dataname in data_list :
    f = open("./result/densenet_base/densenet_softmax_{}_base.txt".format(dataname), 'w')
    data_loader = get_data_loader(dataname, tf)

    for k in tqdm(data_loader):
        image, _ = k
        image = image.to(DEVICE)

        logit = model(image)
        logit = logit
        logit = logit - logit.max()

        softmax = F.softmax(logit).max().item()

        f.write(str(softmax))
        f.write('\n')

    print('{} DONE'.format(dataname))

    f.close()

'''







#noise_test = train_noise(model, data_loader, noise, optimizer)
#torch.save(noise_test,'noise_test.pth')

#noise = universal_adversarial_perturbation(data_loader, model, DEVICE, xi=0.15, delta=0.1, max_iter_uni = 10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10 ,t_p = 0.2)
#torch.save(noise,'noise_test_0.15_90.pth')
