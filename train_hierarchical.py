import torch
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.Hierarchical_network import HierarchicalCSNet
from torch import nn
import time
import os

import argparse
from tqdm import tqdm

from data_utils import TrainDatasetFromFolder
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--pre_epochs', default=200, type=int, help='pre train epoch number')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')

parser.add_argument('--batchSize', default=64, type=int, help='train batch size')
parser.add_argument('--sub_rate', default=0.8, type=float, help='sampling sub rate')

parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
parser.add_argument('--generatorWeights', type=str, default='', help="path to CSNet weights (to continue training)")

parser.add_argument('--group_num', type=int, default=5, help="path to CSNet weights (to continue training)")
parser.add_argument('--loss_mode', type=str, default='normal', help="path to CSNet weights (to continue training)")
parser.add_argument('--fusion_mode',type=str, default='concate', help="path to CSNet weights (to continue training)")


opt = parser.parse_args()

CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
NUM_EPOCHS = opt.num_epochs
PRE_EPOCHS = opt.pre_epochs
GROUP_NUM = opt.group_num
LOSS_MODE = opt.loss_mode
FUSION_MODE = opt.fusion_mode
LOAD_EPOCH = 0


save_dir = '../experiment/hirerachical/epochs' + '_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(BLOCK_SIZE)
argv=sys.argv[1:]
for arg in argv:
    if arg in "--group_num":
        save_dir = save_dir+"_g%d"%(opt.group_num)
    if arg in "--loss_mode":
        save_dir = save_dir+"_l%s"%(opt.loss_mode)
    if arg in "--fusion_mode":
        save_dir = save_dir+"_%s"%(opt.fusion_mode)

train_set = TrainDatasetFromFolder('data/train_crop', crop_size=CROP_SIZE, blocksize=BLOCK_SIZE)
train_loader = DataLoader(dataset=train_set, num_workers=16, batch_size=opt.batchSize, shuffle=True)

net = HierarchicalCSNet(BLOCK_SIZE, opt.sub_rate,group_num=GROUP_NUM,mode=FUSION_MODE)

class MSELoss(nn.Module):
    def __init__(self,mode="normal",group_num=8):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mode = mode
        self.group_num = group_num
        if mode=="weight":
            self.weights = [2*(i+1)/float(group_num*(group_num+1)) for i in range(group_num)]
        else:
            self.weights = [1/float(group_num) for i in range(group_num)]
    
    def forward(self,fake_imgs,real_img):
        if self.mode == "id":
            loss = self.weights[-1]*self.mse(fake_imgs[-1],real_img)
            for i in range(self.group_num-1):
                loss += self.weights[i]*self.mse(fake_imgs[i],fake_imgs[i+1])
        else:
            loss = 0
            for weight, fake_img in zip(self.weights,fake_imgs):
                loss += weight*self.mse(fake_img,real_img)
        
        return loss

mse_loss = MSELoss(mode=LOSS_MODE,group_num=GROUP_NUM)

if opt.generatorWeights != '':
    net.load_state_dict(torch.load(opt.generatorWeights))
    LOAD_EPOCH = opt.loadEpoch

if torch.cuda.is_available():
    net.cuda()
    mse_loss.cuda()

optimizer = optim.Adam(net.parameters(), lr=0.0004, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)



for epoch in range(LOAD_EPOCH, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0, }

    net.train()
    scheduler.step()

    for data, target in train_bar:
        batch_size = data.size(0)
        if batch_size <= 0:
            continue

        running_results['batch_sizes'] += batch_size

        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = net(z)
        optimizer.zero_grad()
        g_loss = mse_loss(fake_img, real_img)

        g_loss.backward()
        optimizer.step()

        running_results['g_loss'] += g_loss.item() * batch_size

        train_bar.set_description(desc='[%d] Loss_G: %.4f lr: %.7f' % (
            epoch, running_results['g_loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if epoch % 1 == 0:
        save_name = save_dir + '/net_epoch_%d_%6f.pth' % (epoch, running_results['g_loss']/running_results['batch_sizes'])
        torch.save(net.state_dict(), save_name)
        torch.save(opt, save_dir+"/opt.pt")
        os.system("python test_h.py --model %s"%(save_name))

    # for saving model
    

