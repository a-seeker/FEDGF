import os
from tqdm import tqdm

import torch
from torchvision.utils import save_image 
from utils.finetune_dataset import LoadData
from torch.utils.data import DataLoader, ConcatDataset
def generatePic(g, args):
    img_number=args.num_per_generator  #每一个数字生成多少张fake图片
    fakedata_save_path= 'data/train/'#生成的fake图片保存目录
    if not os.path.exists(fakedata_save_path):
        os.makedirs(fakedata_save_path)


    for number in range(0,10):
        fake_save_dir=os.path.join(fakedata_save_path,str(number))    #保存图片的目录路径
        if not os.path.exists(fake_save_dir):  #如果没有这个路径则创建
            os.mkdir(fake_save_dir)
        
        g.eval()#进入验证模式，不用计算梯度和参数更新
        g_input=100  #获取模型的输入通道数

        for i in range(img_number):
            z = torch.randn(1,g_input,1,1).cuda()  # 随机生成一些噪声
            fake_img = g(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
            save_image(fake_img,os.path.join(fake_save_dir,
                            str(number)+'_fake_'+str(i)+'.jpg'))  #保存fake样本
        
    
def loadFakeData(args):
    fakedata_save_path= 'data/train/'
    train_dataset = ConcatDataset([LoadData(os.path.join(fakedata_save_path, str(n)), args.num_channels) for n in range(10)])
    return train_dataset
        