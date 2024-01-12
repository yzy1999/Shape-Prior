import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

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
        loss = L_MSE(output, target)
        #print(L_MSE(output, target), L_REG)

        return loss


class shape_prior_loss1(nn.Module):
    def __init__(self,):
        super(shape_prior_loss1, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.a1 = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.a2 = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.a3 = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.a4 = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.a5 = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.a6 = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.a7 = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.a8 = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.a9 = torch.nn.Parameter(torch.ones(1).to(device).requires_grad_())
        self.reset_params()

    def reset_params(self):
        #initialize
        self.a1.data.fill_(0.1)
        self.a2.data.fill_(0.1)
        self.a3.data.fill_(0.1)
        self.a4.data.fill_(0.1)
        self.a5.data.fill_(0.1)
        self.a6.data.fill_(0.1)
        self.a7.data.fill_(0.1)
        self.a8.data.fill_(0.1)
        self.a9.data.fill_(0.1)

    def show_params(self):
        print('self.a1:', self.a1.item())
        print('self.a2:', self.a2.item())
        print('self.a3:', self.a3.item())
        print('self.a4:', self.a1.item())
        print('self.a5:', self.a2.item())
        print('self.a6:', self.a3.item())
        print('self.a7:', self.a1.item())
        print('self.a8:', self.a2.item())
        print('self.a9:', self.a3.item())


    def forward(self, output, target):
        L_MSE = nn.MSELoss()
        L_REG = 0

        #############
        #############
        l1 = 0
        m1 = 0

        l2 = 1
        m2 = -1

        l3 = 1
        m3 = 0

        l4 = 1
        m4 = 1

        l5 = 2
        m5 = -2

        l6 = 2
        m6 = -1

        l7 = 2
        m7 = 0

        l8 = 2
        m8 = 1

        l9 = 2
        m9 = 2

        # in scipy, theta is azimuthal angle and phi is polar angle
        theta = np.linspace(0, 2 * np.pi, 61)
        phi = np.linspace(0, np.pi, 31)
        theta_3d, phi_3d = np.meshgrid(theta, phi)

        xyz_3d = np.array([np.sin(phi_3d) * np.sin(theta_3d),
                           np.sin(phi_3d) * np.cos(theta_3d),
                           np.cos(phi_3d)])

        Ylm1 = sph_harm(abs(m1), l1, theta_3d, phi_3d)
        if m1 < 0:
            Ylm1 = np.sqrt(2) * (-1) ** m1 * Ylm1.imag
        elif m1 > 0:
            Ylm1 = np.sqrt(2) * (-1) ** m1 * Ylm1.real
        r1 = np.abs(Ylm1.real) * xyz_3d

        Ylm2 = sph_harm(abs(m2), l2, theta_3d, phi_3d)
        if m2 < 0:
            Ylm2 = np.sqrt(2) * (-1) ** m2 * Ylm2.imag
        elif m2 > 0:
            Ylm2 = np.sqrt(2) * (-1) ** m2 * Ylm2.real
        r2 = np.abs(Ylm2.real) * xyz_3d

        Ylm3 = sph_harm(abs(m3), l3, theta_3d, phi_3d)
        if m3 < 0:
            Ylm3 = np.sqrt(2) * (-1) ** m3 * Ylm3.imag
        elif m3 > 0:
            Ylm3 = np.sqrt(2) * (-1) ** m3 * Ylm3.real
        r3 = np.abs(Ylm3.real) * xyz_3d

        Ylm4 = sph_harm(abs(m4), l4, theta_3d, phi_3d)
        if m4 < 0:
            Ylm4 = np.sqrt(2) * (-1) ** m4 * Ylm4.imag
        elif m4 > 0:
            Ylm4 = np.sqrt(2) * (-1) ** m4 * Ylm4.real
        r4 = np.abs(Ylm4.real) * xyz_3d

        Ylm5 = sph_harm(abs(m5), l5, theta_3d, phi_3d)
        if m5 < 0:
            Ylm5 = np.sqrt(2) * (-1) ** m5 * Ylm5.imag
        elif m5 > 0:
            Ylm5 = np.sqrt(2) * (-1) ** m5 * Ylm5.real
        r5 = np.abs(Ylm5.real) * xyz_3d

        Ylm6 = sph_harm(abs(m6), l6, theta_3d, phi_3d)
        if m6 < 0:
            Ylm6 = np.sqrt(2) * (-1) ** m6 * Ylm6.imag
        elif m6 > 0:
            Ylm6 = np.sqrt(2) * (-1) ** m6 * Ylm6.real
        r6 = np.abs(Ylm6.real) * xyz_3d

        Ylm7 = sph_harm(abs(m7), l7, theta_3d, phi_3d)
        if m7 < 0:
            Ylm7 = np.sqrt(2) * (-1) ** m7 * Ylm7.imag
        elif m7 > 0:
            Ylm7 = np.sqrt(2) * (-1) ** m7 * Ylm7.real
        r7 = np.abs(Ylm7.real) * xyz_3d

        Ylm8 = sph_harm(abs(m8), l8, theta_3d, phi_3d)
        if m8 < 0:
            Ylm8 = np.sqrt(2) * (-1) ** m8 * Ylm8.imag
        elif m8 > 0:
            Ylm8 = np.sqrt(2) * (-1) ** m8 * Ylm8.real
        r8 = np.abs(Ylm8.real) * xyz_3d

        Ylm9 = sph_harm(abs(m9), l9, theta_3d, phi_3d)
        if m9 < 0:
            Ylm9 = np.sqrt(2) * (-1) ** m9 * Ylm9.imag
        elif m9 > 0:
            Ylm9 = np.sqrt(2) * (-1) ** m9 * Ylm9.real
        r9 = np.abs(Ylm9.real) * xyz_3d

        sigma = 0.1
        y = torch.zeros([target.shape[0], 3, 31, 61]).to(device)
        r1 = torch.Tensor(r1).to(device)
        r2 = torch.Tensor(r2).to(device)
        r3 = torch.Tensor(r3).to(device)
        r4 = torch.Tensor(r4).to(device)
        r5 = torch.Tensor(r5).to(device)
        r6 = torch.Tensor(r6).to(device)
        r7 = torch.Tensor(r7).to(device)
        r8 = torch.Tensor(r8).to(device)
        r9 = torch.Tensor(r9).to(device)


        for i in range(target.shape[0]):
            y[i] = self.a1 * r1 + self.a2 * r2 + self.a3 * r3 + self.a4 * r4 + self.a5 * r5 + self.a6 * r6 + self.a7 * r7 + self.a8 * r8 + self.a9 * r9

        #############
        #############


        L_REG = (y -  output)**2

        L_REG = torch.mean(L_REG)
        loss = L_MSE(output, target) + 0.000001 * L_REG
        #print(L_MSE(output, target), L_REG)

        return loss




class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_features=9, out_features=5673, bias=True)
        self.linear2 = nn.Linear(in_features=5673, out_features=5673, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(-1,3,31,61)

        return x




def createFolderIfNotExists(folderOut):
    ''' creates folder if doesnt already exist
    warns if creation failed '''
    if not os.path.exists(folderOut):
        try:
            os.mkdir(folderOut)
        except OSError:
            print("Creation of the directory %s failed" % folderOut)

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

if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    epochs = 10
    lr = 0.05
    momentum = 0.5
    save_model = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #folder_data = 'D:/shape_prior_data/data_basic'
    x_path = 'D:/shape_prior_data/data_advanced/x_advanced_sigma01.npy'
    y_path = 'D:/shape_prior_data/data_advanced/y_advanced_sigma01.npy'
    train_dataset = InDataset(x_path, y_path, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    #folderIn = folderOut + '/chb' + '01' + '/test' + str(0)
    test_dataset = InDataset(x_path, y_path, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = Net().to(device)
    lossLayer = shape_prior_loss1()
    optimizer = optim.Adam([{"params": model.parameters()}, {"params": lossLayer.parameters()}], lr=lr)  # default lr=0.05

    loss_list = []
    for epoch in range(0, epochs):
        loss_temp = train(model, lossLayer, device, train_loader, optimizer)
        # test(model, device, test_loader)
        loss_temp = loss_temp.cpu().detach().numpy()
        loss_list.append(loss_temp)
    lossLayer.show_params()

    draw_loss(loss_list, epochs)  # loss visualization

    output = test(model, device, test_loader)

    """
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output.cpu().detach().numpy()
    """

    print(output.shape)
    print(output[10])
    #print(output[2000])
    #print(output[2001])
    label = np.load(y_path)
    label = label[7000:]
    print(label[10])
    #print(label[2000])
    #print(label[2001])

    #### visualization

    X = output[10][0]
    Y = output[10][1]
    Z = output[10][2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
        linewidth=0, antialiased=False, alpha=0.5)

    # below are codes copied from stackoverflow, to make the scaling correct
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ax.view_init(elev=30,azim=0) #调节视角，elev指向上(z方向)旋转的角度，azim指xy平面内旋转的角度

    plt.show()




    """
    if save_model:
        if not os.path.exists('ckpt'):
            os.makedirs('ckpt')
        else:
            torch.save(model.state_dict(), 'ckpt/1d_cnn.pt')
    """
