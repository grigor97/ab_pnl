from src.data_generation.simulate_pnl_data import *
from src.models.hsic import *

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import random


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx, :]

        return data[:-1].reshape((-1, 1)), data[-1].reshape((-1, 1))


class Network(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1)
            #             nn.LeakyReLU()
        )

        self.encode = nn.Sequential(
            nn.Linear(1, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1) # or 5?
            #             nn.LeakyReLU()
        )
        self.decode = nn.Sequential(
            nn.Linear(1, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1)
            #             nn.LeakyReLU()
        )

    def forward(self, x, y):
        g1_x = self.network(x)
        g3_y = self.encode(y)
        y_approx = self.decode(g3_y)

        assert y.shape == y_approx.shape

        return [g1_x, y_approx, g3_y]


def train_model(train_loader, test_loader, lamb, num_epochs, input_dim, log_every_batch=10):
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = Network(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

    train_loss_avgs = []
    test_loss_avgs = []

    min_loss = 10000

    for epoch in range(num_epochs):
        model.train()
        train_loss_trace = []

        for batch, (x, y) in enumerate(train_loader):
            x = x.to(device)
            x = x.float()
            y = y.to(device)
            y = y.float()

            g1_x, y_approx, g3_y = model.forward(x, y)
            noise = g3_y - g1_x

            loss = lamb * F.mse_loss(y_approx, y) + (1 - lamb) * HSIC(x, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_trace.append(loss.detach().item())
            if batch % log_every_batch == 0:
                print(f'Training: epoch {epoch} batch {batch} loss {loss}')

        model.eval()
        test_loss_trace = []
        for batch, (x, y) in enumerate(test_loader):
            x = x.to(device)
            x = x.float()
            y = y.to(device)
            y = y.float()

            g1_x, y_approx, g3_y = model.forward(x, y)
            noise = g3_y - g1_x

            loss = lamb * F.mse_loss(y_approx, y) + (1 - lamb) * HSIC(x, noise)

            test_loss_trace.append(loss.detach().item())
            if batch % log_every_batch == 0:
                print(f'Test: epoch {epoch} batch {batch} loss {loss}')

        train_avg = np.mean(train_loss_trace)
        test_avg = np.mean(test_loss_trace)

        if test_avg < min_loss:
            min_loss = test_avg

        train_loss_avgs.append(train_avg)
        test_loss_avgs.append(test_avg)
        print(f'epoch {epoch} finished - avarage train loss {train_avg} ',
              f'avarage test loss {test_avg}')

    return train_loss_avgs, test_loss_avgs, min_loss


def get_final_median_loss(df, batch_size, lamb, num_epochs, num_trials):
    rand_seed = np.random.randint(0, 1000000)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    input_dim = df.shape[1] - 1

    train, test = train_test_split(df, test_size=0.1, random_state=10, shuffle=True)

    train = np.array(train)
    test = np.array(test)

    train = MyDataset(train)
    test = MyDataset(test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    losses = []
    for trial in range(num_trials):
        train_loss_avgs, test_loss_avgs, min_loss = train_model(train_loader, test_loader,
                                                                lamb, num_epochs, input_dim)
        losses.append(min_loss)

    median_loss = np.median(losses)
    return median_loss, losses


def train_mult_model(train_loader, test_loader, lamb, num_epochs, input_dim, log_every_batch=10):
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = Network(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

    train_loss_avgs = []
    test_loss_avgs = []

    min_loss = 10000

    for epoch in range(num_epochs):
        model.train()
        train_loss_trace = []

        for batch, (x, y) in enumerate(train_loader):
            # print('shape of x is ', x.shape)
            # print('shape of y is ', y.shape)
            x = x.to(device)
            x = torch.squeeze(x, -1)
            x = x.float()
            y = y.to(device)
            y = torch.squeeze(y, -1)
            y = y.float()

            # print('shape of x is ', x.shape)
            # print('shape of y is ', y.shape)

            g1_x, y_approx, g3_y = model.forward(x, y)
            noise = g3_y - g1_x

            # print('shape of g1_x is ', g1_x.shape)
            # print('shape of g3_y is ', g3_y.shape)
            # print('shape of noise is ', noise.shape)

            # here the L_1 loss part implemented as in the paper, but it will make
            # more sense if we just take HSIC of the noise and whole x, not only the max
            hsic_vals = torch.Tensor(np.zeros((x.shape[1], 1)))
            for i in range(x.shape[1]):
                val = HSIC(x[:, i].reshape((-1, 1)), noise)
                hsic_vals[i] = val

            loss = lamb * F.mse_loss(y_approx, y) + (1 - lamb) * hsic_vals.max()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_trace.append(loss.detach().item())
            if batch % log_every_batch == 0:
                print(f'Training: epoch {epoch} batch {batch} loss {loss}')

        model.eval()
        test_loss_trace = []
        for batch, (x, y) in enumerate(test_loader):
            x = x.to(device)
            x = torch.squeeze(x, -1)
            x = x.float()
            y = y.to(device)
            y = torch.squeeze(y, -1)
            y = y.float()

            g1_x, y_approx, g3_y = model.forward(x, y)
            noise = g3_y - g1_x

            hsic_vals = torch.Tensor(np.zeros((x.shape[1], 1)))
            for i in range(x.shape[1]):
                val = HSIC(x[:, i].reshape((-1, 1)), noise)
                hsic_vals[i] = val

            # as in the paper
            loss = torch.Tensor([1000])
            if F.mse_loss(y_approx, y) < 1e-3:
                loss = lamb * F.mse_loss(y_approx, y) + (1 - lamb) * hsic_vals.max()

            test_loss_trace.append(loss.detach().item())
            if batch % log_every_batch == 0:
                print(f'Test: epoch {epoch} batch {batch} loss {loss}')

        train_avg = np.mean(train_loss_trace)
        test_avg = np.mean(test_loss_trace)

        if test_avg < min_loss:
            min_loss = test_avg

        train_loss_avgs.append(train_avg)
        test_loss_avgs.append(test_avg)
        print(f'epoch {epoch} finished - avarage train loss {train_avg} ',
              f'avarage test loss {test_avg}')

    return train_loss_avgs, test_loss_avgs, min_loss

