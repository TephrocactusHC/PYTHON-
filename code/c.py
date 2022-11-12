import numpy as np
from gensim.models import keyedvectors
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
#200维的词向量，内容也较为丰富，共有143612个词语和字符，包括了常见的中文字和词语，也有部分英文和特殊符号、数字等
vecmodel = keyedvectors.load_word2vec_format(r'D:\BaiduNetdiskDownload\light_Tencent_AILab_ChineseEmbedding.bin',binary=True)

train_data=pd.read_csv(r'E:\PYTHONDATA\train_data1.csv')
train_data=train_data[1:10562]
# #查看某个字词的索引：
# print(model.get_index('绝望'))
# print(model[3728])


# 得到句向量的一个函数
a = np.zeros(200,)
def get_sentense_vec(sentence):
    length = len(sentence.split())
    try:
        sen_vec = vecmodel[sentence.split()[0]]
    except KeyError:
        sen_vec = a
    if length >= 19:
        for i in range(1,19):
            try:
                sen_vec = np.vstack([sen_vec, vecmodel[sentence.split()[i]]])
            except KeyError:
                sen_vec = np.vstack([sen_vec, a])
    else:
        for i in range(1,length):
            try:
                sen_vec = np.vstack([sen_vec, vecmodel[sentence.split()[i]]])
            except KeyError:
                sen_vec = np.vstack([sen_vec, a])
        for i in range(length,19):
            try:
                sen_vec = np.vstack([sen_vec, a])
            except KeyError:
                sen_vec = np.vstack([sen_vec, a])
    sen_vec=torch.from_numpy(sen_vec)
    sen_vec=sen_vec.to(torch.float32)
    return sen_vec

device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
a = list(map(get_sentense_vec,train_data.iloc[1:,2]))
x_data=torch.stack(a)
y_data=np.array(train_data.iloc[1:,3])
y_data=torch.LongTensor(y_data)
x_train,x_val,y_train,y_val=train_test_split(x_data,y_data,test_size=0.2, random_state=0,stratify=y_data)
x_train=x_train.cuda()
x_val=x_val.cuda()
y_train=y_train.cuda()
y_val=y_val.cuda()

dataset1 = TensorDataset(x_train,y_train)
train_loader = DataLoader(dataset=dataset1,batch_size=32,shuffle=True,num_workers=0)
dataset2 = TensorDataset(x_val,y_val)
val_loader = DataLoader(dataset=dataset2,batch_size=32,shuffle=True,num_workers=0)

n_hidden = 100
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=200, hidden_size=n_hidden,
                            num_layers=2,
                            bidirectional=True,dropout=0.6)
        # fc
        self.fc = nn.Linear(n_hidden * 2, 2)
        # Relu
        self.relu=nn.ReLU(inplace=True)

    def forward(self, X):
        batch_size = X.shape[0]
        input = X.transpose(0, 1)
        hidden_state = torch.randn(2*2, batch_size, n_hidden)
        hidden_state=hidden_state.cuda()
        cell_state = torch.randn(2*2, batch_size, n_hidden)
        cell_state=cell_state.cuda()
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]
        out = self.fc(outputs)
        out=self.relu(out)
        return out

model = BiLSTM()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# #Training
TrainLoss = []
TrainAcc = []
ValLoss = []
ValAcc = []
epochs = 60
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    i=0
    num_correct = 0
    train_loss = 0
    train_losses = []

    for x, y in train_loader:
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())
        train_losses.append(train_loss)
        pred = output.argmax(dim=1)
        num_correct += torch.eq(pred, y).sum().float().item()
        i += 1
        if i ==x_train.shape[0]/32:
            print("TrainLoss: {:.6f}\t TrainAcc: {:.6f}".format(train_loss / len(train_loader),
                                                                          num_correct / len(train_loader.dataset)))
            TrainAcc.append(num_correct / len(train_loader.dataset))
            TrainLoss.append(train_loss / len(train_loader))


    num1_correct = 0
    val_loss = 0
    val_losses = []
    j=0
    with torch.no_grad():
        for x1, y1 in val_loader:
            output = model(x1)
            loss = criterion(output, y1)
            val_loss += float(loss.item())
            val_losses.append(val_loss)
            pred = output.argmax(dim=1)
            num1_correct += torch.eq(pred, y1).sum().float().item()
            j += 1
            if j == x_val.shape[0] / 32:
                print("ValLoss: {:.6f}\t VALAcc: {:.6f}".format( val_loss / len(val_loader),
                                                                            num1_correct / len(val_loader.dataset)))
                ValAcc.append(num1_correct / len(val_loader.dataset))
                ValLoss.append(val_loss / len(val_loader))

length=list(range(0,len(TrainLoss)))
plt.plot(length,TrainLoss, '.-',label='TrainLoss')
plt.plot(length,ValLoss, '.-',label='ValLoss')
plt.legend()
plt.show()
plt.plot(length,TrainAcc, '.-',label='TrainAcc')
plt.plot(length,ValAcc, '.-',label='ValAcc')
plt.legend()
plt.show()
torch.save(model, r'E:\PYTHONDATA\bilstm60.pth')

#一些对gensim的实验性操作，可以通过这个部分的内容来熟悉gensim
# print (model.key_to_index)
# print (model.key_to_index.keys())
# #查看与该词最接近的其他词汇及相似度：
# print(model.most_similar(['绝望']))
# for k in model.similar_by_word('绝望'):
#     print(k[0],k[1])
# #查看两个词之间的相似度：
# print(model.similarity('妈妈','母亲'))

