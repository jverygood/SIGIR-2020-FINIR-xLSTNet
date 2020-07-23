import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as data
import logging
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
logging.basicConfig(level=logging.DEBUG)

window = 30  # 窗口大小
# hidRNN = 100
# hidCNN = 32
# hidSkip = 100
# CNN_kernel = 6
# skip = 10
# highway_window = 0
# dropout = 0
# hidRNN = 100
hidRNN = 200
# hidCNN = 32
hidCNN = 64
hidSkip = 100
# CNN_kernel = 6
CNN_kernel=5
skip = 10
highway_window = 0
dropout = 0

EPOCH = 20
BATCH_SIZE = 16


alum = pd.read_csv('./data/Aluminium_tr.csv', encoding='utf-8')
copper = pd.read_csv('./data/Copper_tr.csv', encoding='utf-8')
lead = pd.read_csv('./data/Lead_tr.csv', encoding='utf-8')
nickel = pd.read_csv('./data/Nickel_tr.csv', encoding='utf-8')
tin = pd.read_csv('./data/Tin_tr.csv', encoding='utf-8')
zinc = pd.read_csv('./data/Zinc_tr.csv', encoding='utf-8')

# tr = pd.read_csv('./data/tr.csv', encoding='utf-8')
start_time = alum.shape[0]-253*13  # start time
validation_ratio = 0.1

dataset = pd.concat([
    pd.DataFrame(alum.loc[start_time:], columns=['Open', 'High', 'Low', 'Close', 'Volume']),
    pd.DataFrame(copper.loc[start_time:], columns=['Open', 'High', 'Low', 'Close', 'Volume']),
    pd.DataFrame(lead.loc[start_time:], columns=['Open', 'High', 'Low', 'Close', 'Volume']),
    pd.DataFrame(nickel.loc[start_time:], columns=['Open', 'High', 'Low', 'Close', 'Volume']),
    pd.DataFrame(tin.loc[start_time:], columns=['Open', 'High', 'Low', 'Close', 'Volume']),
    pd.DataFrame(zinc.loc[start_time:], columns=['Open', 'High', 'Low', 'Close', 'Volume']),
    pd.DataFrame(alum.loc[start_time:], columns=['DOW', 'DXY', 'NKY', 'SHS300', 'SP500', 'UKX', 'VIX', 'SX5'])
], axis=1)

dataset = torch.tensor(dataset.values, dtype=torch.float).cuda()

target = pd.concat([
    pd.DataFrame(alum, columns=['Close']),
    pd.DataFrame(copper, columns=['Close']),
    pd.DataFrame(lead, columns=['Close']),
    pd.DataFrame(nickel, columns=['Close']),
    pd.DataFrame(tin, columns=['Close']),
    pd.DataFrame(zinc, columns=['Close']),
], axis=1)
target = torch.tensor(target.values, dtype=torch.float)

label = pd.concat([
    pd.DataFrame(alum, columns=['L1', 'L20', 'L60']),
    pd.DataFrame(copper, columns=['L1', 'L20', 'L60']),
    pd.DataFrame(lead, columns=['L1', 'L20', 'L60']),
    pd.DataFrame(nickel, columns=['L1', 'L20', 'L60']),
    pd.DataFrame(tin, columns=['L1', 'L20', 'L60']),
    pd.DataFrame(zinc, columns=['L1', 'L20', 'L60']),
], axis=1)
label = torch.tensor(label.values, dtype=torch.float)

train_length = int(dataset.shape[0] * (1-validation_ratio))
validate_length = int(dataset.shape[0] * validation_ratio)

train_data = dataset[:train_length, ]
train_target = target[:train_length, ]

validate_data = dataset[-(validate_length+window-1):, ]
validate_label = label[-validate_length:, ]

class LoadData(data.Dataset):

    def __init__(self, dataset, label, d):
        self.dataset = dataset
        self.label = label

        self.d = d

        self.train_length = train_length - window - self.d

    def __getitem__(self, idx):
        X = self.dataset[idx:idx + window, ]
        Y = self.label[idx + window + self.d - 1,]

        return X, Y

    def __len__(self):
        return self.train_length

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()
        self.P = window  # window, 输入窗口大小
        self.hidR = hidRNN

        self.features = 5
        self.m = 6 * self.features + 5  # 变量数

        self.hidC = hidCNN  # 卷积核数
        self.hidS = hidSkip
        self.Ck = CNN_kernel  # 卷积核大小
        self.skip = skip
        self.pt = (self.P - self.Ck) // self.skip
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout)
        self.target = 6
        self.em_size = 5

        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.target)

        else:
            self.linear1 = nn.Linear(self.hidR, self.target)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)

        self.output = nn.Tanh()

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.em_size + 5, out_features=self.features),
            nn.ReLU())
        self.em = nn.Embedding(6, self.em_size)
        self.w = nn.Linear(5, 5)
        self.w1 = nn.Linear(25, 1)
        self.w2 = nn.Linear(25, 1)
        self.w3 = nn.Linear(25, 1)
        self.w4 = nn.Linear(25, 1)
        self.w5 = nn.Linear(25, 1)
        self.fea_cross_weight = [self.w1, self.w2, self.w3, self.w4, self.w5]
        self.index_fc = nn.Linear(8, 5)
        self.W1 = nn.Linear(49, 7)

    def forward(self, x):

        batch_size = x.size(0)  # batch, window, n_val

        for i in range(self.target):
            embed = self.em(torch.tensor(i).cuda())
            embed = torch.unsqueeze(torch.unsqueeze(embed, 0), 1)  # 1, 1, size
            embed = embed.repeat(batch_size, window, 1)
            fea = x[:, :, 5 * i:5 * (i + 1)]
            # fea1=fea.unsqueeze(-1)
            # fea2=fea.unsqueeze(-2)
            # fea_cross=torch.matmul(fea1,fea2).reshape(batch_size,window,-1)
            # fea_map=[]
            # for j in range(self.em_size):
            #     fea_map.append(self.fea_cross_weight[j](fea_cross))
            # crossed_fea=torch.cat(fea_map,dim=2)
            # fc=crossed_fea
            fc = self.w(fea)
            fc += embed
            # fc = self.fc(torch.cat([embed, fea], dim=2))      # batch_size, window, size

            if i == 0:
                fc_output = fc
            else:
                fc_output = torch.cat([fc_output, fc], dim=2)
        index_fea = x[:, :, 30:]
        index_fea = self.index_fc(index_fea)

        features = torch.cat([fc_output, index_fea], dim=2)
        features_origin = features
        # lstnet_input = fc_output
        features = features.reshape(batch_size, window, -1, self.em_size)
        split_tensor1 = torch.stack(torch.split(features, self.em_size * [1], 3), 0)
        split_tensor2 = split_tensor1.permute(0, 1, 2, 4, 3)
        dot_result_m = torch.matmul(split_tensor1, split_tensor2)
        dot_result_m = dot_result_m.view(self.em_size, batch_size, window, 7 * 7)
        crossed_feas = self.W1(dot_result_m)
        crossed_feas = crossed_feas.permute(1, 0, 2, 3)
        crossed_feas = crossed_feas.permute(0, 2, 1, 3)
        crossed_feas = crossed_feas.permute(0, 1, 3, 2)
        # crossed_feas=F.relu(crossed_feas)
        lstnet_input = crossed_feas
        lstnet_input = lstnet_input.reshape(batch_size, window, -1)
        lstnet_input += features_origin
        # features2 = features.reshape(batch_size,window,self.em_size,-1)

        # CNN
        c = lstnet_input.view(-1, 1, self.P, self.m)  # batch, 1, window, n_val
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)  # batch, hidCNN, window-kernel_size+1

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))  # batch, hidRNN

        # skip-rnn
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)  # batch, skip*hidSkip + hidRNN

        res = self.linear1(r)  # batch, n_val

        # highway
        if (self.hw > 0):
            z = lstnet_input[:, -self.hw:, :]  # batch, hw, n_val
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z  # batch, n_val

        res = self.output(res)

        return res

def model_train(d):

    if d == 1:
        test_label = validate_label[:, 0::3]
    elif d == 20:
        test_label = validate_label[:, 1::3]
    elif d == 60:
        test_label = validate_label[:, 2::3]

    train_dataset = LoadData(train_data, target, d)
    train_generator = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    torch.cuda.manual_seed(train_length)
    np.random.seed(train_length)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    LSTNet = Model()
    loss = nn.MSELoss()

    LSTNet = LSTNet.cuda()
    loss = loss.cuda()

    LSTNet_optimizer = torch.optim.Adam(LSTNet.parameters(), lr=1e-4)

    acc_max = 0

    for epoch in range(EPOCH):
        for step, train in enumerate(train_generator):
            X, Y = train
            X, Y = X.cuda(), Y.cuda()

            # Train
            Y_hat = LSTNet(X)

            mse_loss = loss(Y_hat, Y)

            LSTNet_optimizer.zero_grad()
            mse_loss.backward()
            LSTNet_optimizer.step()

            if step % 10 == 0:
                logging.info("epoch:%d step:%d [loss: %f]" % (epoch, step, mse_loss.cpu()))

        count_right = 0
        count_wrong = 0

        for i in range(validate_length):
            X = validate_data[i:i + window, ]
            X = X.unsqueeze(0)

            period = validate_data[i + window - 1,].cpu()

            Y = test_label[i, ]

            X_hat = LSTNet(X).cpu()

            for j in range(6):
                if (X_hat[:, j] > period[3 + j * 5]) and (Y[j] == 1):
                    count_right += 1
                elif (X_hat[:, j] <= period[3 + j * 5]) and (Y[j] == 0):
                    count_right += 1
                else:
                    count_wrong += 1

        acc = count_right / (count_wrong + count_right)
        print("epoch %d acc:%f" % (epoch, acc))

        if acc > acc_max:
            torch.save(LSTNet, 'LSTNet_' + str(d) + 'd_' + str(acc) + '_' + str(epoch) + '.pth')
            acc_max = acc
            print("max acc:%f" % acc)

if __name__  ==  "__main__":
    for i in range(1000):
        model_train(d=1)
        model_train(d=20)
        model_train(d=60)