import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt


class InDataset(Dataset):
    def __init__(self, x_path, y_path, train):
        super(InDataset, self).__init__()
        self.train = train

        x_path = x_path
        # print(x_path)
        y_path = y_path
        # print(y_path)
        self.x = np.load(x_path)
        self.y = np.load(y_path)

        if train == True:
            self.x = self.x[0:7000]
            self.y = self.y[0:7000]
        else:
            self.x = self.x[7000:]
            self.y = self.y[7000:]

        self.x = self.x
        self.y = self.y

        print(self.x.shape)
        print(self.y.shape)

        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.y = torch.from_numpy(self.y).to(torch.float32)

    def __getitem__(self, index):
        x_show = self.x[index]
        y_show = self.y[index]

        return x_show, y_show

    def __len__(self):
        return len(self.x)


def train(model, lossLayer, device, train_loader, optimizer):
    model.train()
    #lossLayer = torch.nn.MSELoss()
    #lossLayer = shape_prior_loss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 15 == 0 and batch_idx > 0:
            print('Loss:', loss.item())

    return loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    lossLayer = torch.nn.MSELoss()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss = lossLayer(output, target).item()
        output = output.cpu().detach().numpy()


    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return output


class shape_prior_loss(nn.Module):
    def __init__(self,):
        super(shape_prior_loss, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.a = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.b = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.c = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.reset_params()

    def reset_params(self):
        #initialize
        self.a.data.fill_(0.1)
        self.b.data.fill_(0.1)
        self.c.data.fill_(0.1)

    def show_params(self):
        print('self.a:', self.a.item())
        print('self.b:', self.b.item())
        print('self.c:', self.c.item())

    def forward(self, output, target):
        L_MSE = nn.MSELoss()
        L_REG = 0

        for i in range(target.shape[1]):
            L_REG = L_REG + ((self.a*(i+1)**2 + self.b*(i+1) + self.c) - output[:,i])**2

        L_REG = torch.mean(L_REG)
        #loss = L_MSE(output, target)
        loss = L_MSE(output, target) + 0.000001 * L_REG
        #print(L_MSE(output, target), L_REG)

        return loss


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_features=3, out_features=100, bias=True)
        self.linear2 = nn.Linear(in_features=100, out_features=100, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)

        return x

def draw_loss(Loss_list,epoch):
    plt.cla()
    x1 = range(1, epoch+1)
    print(x1)
    y1 = Loss_list
    print(y1)
    plt.title('Train loss vs. epoches', fontsize=15)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=15)
    plt.ylabel('Train loss', fontsize=15)
    plt.grid()
    #plt.savefig("D:/shape_prior_data/results_basic/prior_m10_sigma01.png")
    #plt.savefig("./lossAndacc/Train_loss.png")
    plt.show()


def createFolderIfNotExists(folderOut):
    ''' creates folder if doesnt already exist
    warns if creation failed '''
    if not os.path.exists(folderOut):
        try:
            os.mkdir(folderOut)
        except OSError:
            print("Creation of the directory %s failed" % folderOut)



if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    epochs = 50
    lr = 0.05
    momentum = 0.5
    save_model = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #folder_data = 'D:/shape_prior_data/data_basic'
    x_path = 'D:/shape_prior_data/data_basic/x_basic_m100_sigma03.npy'
    y_path = 'D:/shape_prior_data/data_basic/y_basic_m100_sigma03.npy'
    train_dataset = InDataset(x_path, y_path, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = InDataset(x_path, y_path, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = Net().to(device)
    lossLayer = shape_prior_loss()
    optimizer = optim.Adam([{"params": model.parameters()}, {"params": lossLayer.parameters()}], lr=lr)  # default lr=0.001

    loss_list = []
    for epoch in range(0, epochs):
        loss_temp = train(model, lossLayer, device, train_loader, optimizer)
        loss_temp = loss_temp.cpu().detach().numpy()
        loss_list.append(loss_temp)
        # test(model, device, test_loader)
    lossLayer.show_params()


    draw_loss (loss_list, epochs) #loss visualization

    output = test(model, device, test_loader)

    """
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output.cpu().detach().numpy()
    """

    print(output.shape)
    print("Prediction", output[10])
    #print(output[2000])
    #print(output[2001])
    label = np.load(y_path)
    label = label[7000:]
    print("The Label",label[10])
    #print(label[2000])
    #print(label[2001])

    """
    if save_model:
        if not os.path.exists('ckpt'):
            os.makedirs('ckpt')
        else:
            torch.save(model.state_dict(), 'ckpt/dense.pt')
    """
