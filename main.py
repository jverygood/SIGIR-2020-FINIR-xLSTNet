import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
import os
import time
import datetime
start_time = time.time()        # start_time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
logging.basicConfig(level=logging.DEBUG)

window = 30

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

BATCH_SIZE = 16

# train data
alum1 = pd.read_csv('./data/Aluminium_tr.csv', encoding='utf-8')
copper1 = pd.read_csv('./data/Copper_tr.csv', encoding='utf-8')
lead1 = pd.read_csv('./data/Lead_tr.csv', encoding='utf-8')
nickel1 = pd.read_csv('./data/Nickel_tr.csv', encoding='utf-8')
tin1 = pd.read_csv('./data/Tin_tr.csv', encoding='utf-8')
zinc1 = pd.read_csv('./data/Zinc_tr.csv', encoding='utf-8')

# tr = pd.read_csv('./data/tr.csv', encoding='utf-8')

train_data = pd.concat([
                        pd.DataFrame(alum1, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(copper1, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(lead1, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(nickel1, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(tin1, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(zinc1, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(alum1, columns=['DOW', 'DXY', 'NKY', 'SHS300', 'SP500', 'UKX', 'VIX', 'SX5'])
                        ], axis=1)

# validation data
alum = pd.read_csv('./data/Aluminium_2019.csv', encoding='utf-8')
copper = pd.read_csv('./data/Copper_2019.csv', encoding='utf-8')
lead = pd.read_csv('./data/Lead_2019.csv', encoding='utf-8')
nickel = pd.read_csv('./data/Nickel_2019.csv', encoding='utf-8')
tin = pd.read_csv('./data/Tin_2019.csv', encoding='utf-8')
zinc = pd.read_csv('./data/Zinc_2019.csv', encoding='utf-8')

# va = pd.read_csv('./data/va.csv', encoding='utf-8')

test_data = pd.concat([
                        pd.DataFrame(alum, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(copper, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(lead, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(nickel, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(tin, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(zinc, columns=['Open', 'High', 'Low', 'Close', 'Volume']),
                        pd.DataFrame(alum, columns=['DOW', 'DXY', 'NKY', 'SHS300', 'SP500', 'UKX', 'VIX', 'SX5'])
                        ], axis=1)

test_length = test_data.shape[0]

test_data = pd.concat([train_data.loc[train_data.shape[0]-window+1:], test_data], axis=0)
test_data = torch.tensor(test_data.values, dtype=torch.float).cuda()

dates = pd.DataFrame(alum, columns=['Time'])

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()
        self.P = window        # window
        self.hidR = hidRNN

        self.features = 5
        self.m = 6*self.features + 5        #

        self.hidC = hidCNN      #
        self.hidS = hidSkip
        self.Ck = CNN_kernel        #
        self.skip = skip
        self.pt = (self.P - self.Ck) // self.skip
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m));
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
            nn.Linear(in_features=self.em_size+5, out_features=self.features),
            nn.ReLU())
        self.em = nn.Embedding(6, self.em_size)
        self.w=nn.Linear(5,5)
        self.w1=nn.Linear(25,1)
        self.w2=nn.Linear(25,1)
        self.w3=nn.Linear(25,1)
        self.w4=nn.Linear(25,1)
        self.w5=nn.Linear(25,1)
        self.fea_cross_weight=[self.w1,self.w2,self.w3,self.w4,self.w5]
        self.index_fc=nn.Linear(8,5)
        self.W1=nn.Linear(49,7)
    def forward(self, x):

        batch_size = x.size(0)      # batch, window, n_val

        for i in range(self.target):
            embed = self.em(torch.tensor(i).cuda())
            embed = torch.unsqueeze(torch.unsqueeze(embed, 0), 1)       # 1, 1, size
            embed = embed.repeat(batch_size, window, 1)
            fea=x[:, :, 5*i:5*(i+1)]
            # fea1=fea.unsqueeze(-1)
            # fea2=fea.unsqueeze(-2)
            # fea_cross=torch.matmul(fea1,fea2).reshape(batch_size,window,-1)
            # fea_map=[]
            # for j in range(self.em_size):
            #     fea_map.append(self.fea_cross_weight[j](fea_cross))
            # crossed_fea=torch.cat(fea_map,dim=2)
            # fc=crossed_fea
            fc=self.w(fea)
            fc+=embed
            # fc = self.fc(torch.cat([embed, fea], dim=2))      # batch_size, window, size

            if i == 0:
                fc_output = fc
            else:
                fc_output = torch.cat([fc_output, fc], dim=2)
        index_fea=x[:, :, 30:]
        index_fea=self.index_fc(index_fea)

        features = torch.cat([fc_output, index_fea], dim=2)
        features_origin1=features
        # lstnet_input = fc_output
        features = features.reshape(batch_size,window,-1,self.em_size)
        split_tensor1 = torch.stack(torch.split(features, self.em_size * [1], 3), 0)
        split_tensor2 =split_tensor1.permute(0, 1, 2, 4,3)
        dot_result_m = torch.matmul(split_tensor1, split_tensor2)
        dot_result_m = dot_result_m.view( self.em_size, batch_size,window, 7*7)
        crossed_feas=self.W1(dot_result_m)
        crossed_feas=crossed_feas.permute(1,0,2,3)
        crossed_feas=crossed_feas.permute(0,2,1,3)
        crossed_feas=crossed_feas.permute(0,1,3,2)
        # crossed_feas=F.relu(crossed_feas)
        features_origin2=crossed_feas
        # features
        lstnet_input=crossed_feas
        lstnet_input=lstnet_input.reshape(batch_size,window,-1)
        lstnet_input+=features_origin1
        # features2 = features.reshape(batch_size,window,self.em_size,-1)

        # CNN
        # lstnet_input=features
        c = lstnet_input.view(-1, 1, self.P, self.m)       # batch, 1, window, n_val
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)     # batch, hidCNN, window-kernel_size+1

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))       # batch, hidRNN

        # skip-rnn
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)        # batch, skip*hidSkip + hidRNN

        res = self.linear1(r)       # batch, n_val

        # highway
        if (self.hw > 0):
            z = lstnet_input[:, -self.hw:, :]      # batch, hw, n_val
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z       # batch, n_val

        res = self.output(res)

        return res

def model_test(d):

    LSTNet = torch.load('./model_saver/LSTNet_%dd_2019.pth' % d)

    LSTNet = LSTNet.cuda()

    label = []

    for i in range(test_length):
        X = test_data[i:i+window, ]
        X = X.unsqueeze(0)

        period = test_data[i + window - 1,]

        X_hat = LSTNet(X).cpu()

        step_label = []

        for j in range(6):
            if X_hat[:, j] > period[3 + j * 5].cpu():
                step_label.append(1)
            else:
                step_label.append(0)

        label.append(step_label)

    return label

if __name__  ==  "__main__":

    metal_names = ['Aluminium', 'Copper', 'Lead', 'Nickel', 'Tin', 'Zinc']
    ds = [1, 20, 60]

    results = pd.DataFrame(columns=['id', 'label'])

    for i in range(3):
        d = ds[i]
        labels = model_test(d=d)

        for j in range(6):
            name = metal_names[j]
            id = 'LME' + name + '-test-' + str(d) + 'd-'

            for k in range(test_length):
                date = dates.loc[k]['Time']
                year,month,day=date.split('/')
                if len(month)==1:
                    month='0'+month
                if len(day)==1:
                    day='0'+day
                date=year+'-'+month+'-'+day
                # date = date.replace('/','-')
                # date = date.replace('-1','-01')
                id_date = id + date
                label = labels[k][j]

                results = results.append({'id':id_date, 'label':label}, ignore_index=True)

    results.to_csv('results.csv', encoding='utf-8', index=False)

end_time = time.time()
print("Running Time: %f seconds" % (end_time-start_time))





