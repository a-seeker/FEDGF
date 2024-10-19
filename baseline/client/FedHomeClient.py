import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
from utils.util import DatasetSplit
from imblearn.over_sampling import SMOTE
from models.Nets import LeNet5_FedHome, LocalModel

class FedHomeClient(object):
    def __init__(self, args, id):
        self.id = id
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.device = torch.device("cuda" if self.args.gpu > -1 and torch.cuda.is_available() else "cpu")

        self.opt_pred = torch.optim.SGD(self.model.predictor.parameters(), lr=self.learning_rate)
        self.trainloader_rep = None
        self.model = LeNet5_FedHome(args.dataset).cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    def train(self, model_params, dataset, idxs):
        self.set_parameters(model_params)
        ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)
        self.model.train()

        # train and update
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def generate_data(self, dataset, idxs):
        ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)
        train_data_rep = []
        train_data_y = []
        trainloader = ldr_train
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                train_data_rep.append(self.model.base(x).detach().cpu().numpy())
                train_data_y.append(y.detach().cpu().numpy())
        train_data_rep = np.concatenate(train_data_rep, axis=0)
        train_data_y = np.concatenate(train_data_y, axis=0)
        if len(np.unique(train_data_y)) > 1: # 样本标签数大于1，用smote随机选择一个少数类，用临近算法合成
            smote = SMOTE()
            X, Y = smote.fit_resample(train_data_rep, train_data_y)
        else:
            X, Y = train_data_rep, train_data_y
        print(f'Client {self.id} data ratio: ', '{:.2f}%'.format(100*(len(Y))/len(train_data_y)))
        X_train = torch.Tensor(X).type(torch.float32)
        y_train = torch.Tensor(Y).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        self.trainloader_rep = DataLoader(train_data, self.batch_size, drop_last=True, shuffle=False)

    def train_pred(self):
        self.model.train()
        for i, (x, y) in enumerate(self.trainloader_rep):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            self.opt_pred.zero_grad()
            output = self.model.predictor(x)
            loss = self.loss_func(output, y)
            loss.backward()
            self.opt_pred.step()

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()