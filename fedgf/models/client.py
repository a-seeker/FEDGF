import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class MnistDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.idx = idxs
        self.dataset = dataset

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.dataset[self.idx[index]][0], self.dataset[self.idx[index]][1]

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
# def get_client_label(dict_users, client_id, dataset_train):
#     if dataset_train[dict_users[client_id][0]][1] == dataset_train[dict_users[client_id][-1]][1]:
#         label = dataset_train[dict_users[client_id][-1]][1]
#     else:
#         for i in range(len(dict_users[client_id])):
#             if dataset_train[dict_users[client_id][i]][1] == dataset_train[dict_users[client_id][0]][1]:
#                 k=i
#             else:
#                 break
#         if k >= len(dict_users[client_id]) - k:
#             label = dataset_train[dict_users[client_id][0]][1]
#         else:
#             label = dataset_train[dict_users[client_id][-1]][1]
#     return label

def get_client_label(dataset, idxs):
    class_num = len(dataset.classes)
    label_dict = {i:[] for i in range(class_num)}
    for index in idxs:
        label = dataset[index][1]
        label_dict[label].append(index)
    
    mx_len = 0
    mx_label = 0
    for k,v in label_dict.items():
        if(len(v) > mx_len):
            mx_len = len(v)
            mx_label = k

    d = MnistDataset(dataset=dataset, idxs=np.array(label_dict[mx_label]))
    return mx_label, d

class Client(object):
    def __init__(self, args, dataset, idxs, client_id, global_round):
        self.args = args
        self.client_id = client_id  # 客户端id
        self.global_round = global_round  # 当前是第几轮
        self.dataset = dataset
        d = MnistDataset(dataset=dataset, idxs=idxs)
        if args.iid:
            self.dataloader1 = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs1, shuffle=True)
        else:
            self.dataloader1 = DataLoader(dataset=d, batch_size=args.local_bs1, shuffle=True)

        self.label, d = get_client_label(dataset, idxs)

        # 统计数据集中最多的标签
        # if dataset[idxs[0]][1] == dataset[idxs[-1]][1]:
        #     self.label = dataset[idxs[0]][1]
        #     d = MnistDataset(dataset=dataset, idxs=idxs)
        # else:
        #     # for i in range(len(idxs)):
        #     #     if dataset[idxs[i]][1] == dataset[idxs[0]][1]:
        #     #         k = i
        #     #     else:
        #     #         break
        #     # if k >= len(idxs) - k:
        #     #     self.label = dataset[idxs[0]][1]
        #     #     d = MnistDataset(dataset=dataset, idxs=idxs[0:k+1])
        #     # else:
        #     #     self.label = dataset[idxs[-1]][1]
        #     #     d = MnistDataset(dataset=dataset, idxs=idxs[k+1:])
        #     label_end = [0]
        #     for i in range(1,len(idxs)):
        #         if dataset[idxs[i]][1] != dataset[idxs[i-1]][1]:
        #             if i != 1:
        #                 label_end.append(i-1)
        #     max = 0
        #     start, end = 0, 0
        #     for i in range(1,len(label_end)):
        #         if max < label_end[i] -label_end[i-1]:
        #             max = label_end[i] -label_end[i-1]
        #             start = label_end[i-1] + 1
        #             end = label_end[i] 
        #     self.label = dataset[idxs[end]][1]
        #     d = MnistDataset(dataset=dataset, idxs=idxs[start:end+1])

        self.dataloader2 = DataLoader(dataset=d, batch_size=args.local_bs1, shuffle=True)   # gan train数据集
            

    def trainGan(self, D, G):
        # Training Loop
        # Lists to keep track of progress
        netG = G.cuda()
        netD = D.cuda()

        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        criterion = nn.BCELoss()
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, self.args.nz, 1, 1).cuda()
        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.
        optimizerD = torch.optim.Adam(netD.parameters(), lr=self.args.lr2, betas=(self.args.beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=self.args.lr2, betas=(self.args.beta1, 0.999))

        iters = 0

        print(str(self.client_id)+ " Starting Training Gan...")
        # For each epoch
        for epoch in range(self.args.local_ep2):
            # For each batch in the dataloader
            for i, (data, l) in enumerate(self.dataloader2, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                # print(data[0].shape)
                real_cpu = data.cuda()
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float).cuda()
                # print(label.shape)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # print(output.shape)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.args.nz, 1, 1).cuda()
                # Generate fake image batch with G
                fake = netG(noise)
                # print(fake.shape)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.args.local_ep2, i, len(self.dataloader2),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if self.args.save_gan_pic:
                    if (iters % 10 == 0) or ((epoch == self.args.local_ep2-1) and (i == len(self.dataloader2)-1)):
                        with torch.no_grad():
                            fake = netG(fixed_noise).detach().cpu()
                        img=vutils.make_grid(fake, padding=2, normalize=True)
                        
                        if not os.path.exists("./testoutput/models"):
                            os.makedirs("./testoutput/models")
                        if not os.path.exists("./testoutput/client"+str(self.client_id)+"/round"+str(self.global_round)):
                            os.makedirs("./testoutput/client"+str(self.client_id)+"/round"+str(self.global_round))
                            
                        # 是否保存训练图片
                        save_image(img,"./testoutput/client"+str(self.client_id)+"/round"+str(self.global_round)+"/img"+str(iters)+'.png')
                
                iters += 1  
        return netD, netG, D_losses, G_losses

    def train(self, net):
        print(str(self.client_id)+ " Starting Training local model...")
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr1, momentum=self.args.momentum)
        loss_func = nn.CrossEntropyLoss()
        epoch_loss = []
        for iter in range(self.args.local_ep1):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.dataloader1):
                images, labels = images.cuda(), labels.cuda()
                if self.args.dataset == 'cifar' or self.args.dataset == 'cinic':
                    resize = transforms.Resize([32,32])
                    images = resize(images)

                net.zero_grad()
                log_probs = net(images)
                # if self.client_id == 22:
                #     print(labels)
                #     print(log_probs)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.dataloader1.dataset),
                               100. * batch_idx / len(self.dataloader1), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
