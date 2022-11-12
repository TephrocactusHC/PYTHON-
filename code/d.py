import torch
import numpy as np
from gensim.models import keyedvectors
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn import metrics
#import matplotlib.pyplot as plt
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

criterion = nn.CrossEntropyLoss()

vecmodel = keyedvectors.load_word2vec_format(r'D:\BaiduNetdiskDownload\light_Tencent_AILab_ChineseEmbedding.bin',
                                             binary=True)

test_data = pd.read_csv(r'E:\PYTHONDATA\test_data1.csv')
test_data = test_data[1:10142]

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

a = list(map(get_sentense_vec,test_data.iloc[0:,2]))
x_data = torch.stack(a)
y_data = np.array(test_data.iloc[0:,3])
y_data = torch.LongTensor(y_data)
x_data = x_data.cuda()
y_data = y_data.cuda()
dataset = TensorDataset(x_data,y_data)
test_loader = DataLoader(dataset=dataset,batch_size=1,num_workers=0)

model = torch.load(r'E:\PYTHONDATA\bilstm60.pth')

theone = torch.ones(1)
thezero = torch.zeros(1)

y=[]
predlist=[]

changshi=[]
for epoch in range(1):
    TP=0
    TN=0
    FP=0
    FN=0
    print('测试结果：')
    num1_correct = 0
    test_loss = 0
    test_losses = []
    j = 0
    for x1, y1 in test_loader:
        output = model(x1)
        loss = criterion(output, y1)
        test_loss += float(loss.item())
        test_losses.append(test_loss)
        pred = output.argmax(dim=1)
        changshi.append(pred.item())
        num1_correct += torch.eq(pred, y1).sum().float().item()
        if y1.item() == 0 and pred.item() == 0:
            TP += 1
        if y1.item() == 1 and pred.item() == 1:
            TN += 1
        if y1.item() == 0 and pred.item() == 1:
            FP += 1
        if y1.item() == 1 and pred.item() == 0:
            FN += 1

        output=torch.softmax(output,dim=1)
        output=output[0,0].cpu()
        y.append(y1.cpu().item())
        predlist.append(output.item())

        # if (FP+TN)==0:
        #     FPR=0
        # elif FP!=0 and TN==0:
        #     FPR=0
        # else:
        #     FPR=(FP)/(FP+TN)
        #
        # if (TP+TN)==0:
        #     TPR=0
        # elif TP!=0 and TN==0:
        #     TPR=0
        # else:
        #     TPR=(TP)/(TP+TN)
        # if FPR!=0 and TPR!=0:
        #     fpr.append(FPR)
        #     tpr.append(TPR)
        j += 1
        if j == x_data.shape[0] / 1:
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            print("混淆矩阵：")
            print('TP: {}\t FN: {}\nFP: {}\t TN: {}\t\n '.format(TP, FN, FP, TN))
            print("TestLoss: {:.6f}\t\n".format(test_loss / len(test_loader)))
            print('P: {:.6f}\t R: {:.6f}\t F1: {:.6f}\t TestAcc: {:.6f}'.format(P,R,F1,
                                                            num1_correct / len(test_loader.dataset)))

fpr, tpr, thresholds = metrics.roc_curve(y, predlist, pos_label=0)
print("AUC:",metrics.auc(fpr, tpr))

#输出混淆矩阵以及各个评价指标
target_names=['真新闻', '假新闻']
print(metrics.classification_report(y,changshi,target_names=target_names))

#plt.plot(fpr, tpr,color='darkorange',  label='ROC',lw=2)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.legend(loc="lower right")
#plt.plot([0,1],[0,1],"k-")
#plt.show()
