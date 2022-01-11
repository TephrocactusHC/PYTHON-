import pandas as pd
import matplotlib.pyplot as plt
train_data=pd.read_csv(r'E:\PYTHONDATA\train_data1.csv')
test_data=pd.read_csv(r'E:\PYTHONDATA\test_data1.csv')
print(train_data.info())
print(test_data.info())
train_data['Title']=train_data['Title'].astype('str')
test_data['Title']=test_data['Title'].astype('str')
train_data['text_len'] = train_data['Title'].apply(lambda x: len(x.split(' ')))
print('train_title_len\n',train_data['text_len'].describe())
test_data['text_len'] = test_data['Title'].apply(lambda x: len(x.split(' ')))
print('test_title_len\n',test_data['text_len'].describe())
_ = plt.hist(train_data['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("TRAIN:Histogram of char count")
plt.show()
_ = plt.hist(test_data['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("TEST:Histogram of char count")
plt.show()

def getlen(text):
    return len(text)

train_data['Ofiicial Account Name'] = train_data['Ofiicial Account Name'].astype('str')
test_data['Ofiicial Account Name'] = test_data['Ofiicial Account Name'].astype('str')
train_data['Ofiicial Account Name'] = list(map(getlen,train_data['Ofiicial Account Name']))
print('train_Name_len\n',train_data['Ofiicial Account Name'].describe())
test_data['Ofiicial Account Name'] = list(map(getlen,test_data['Ofiicial Account Name']))
print('test_Name_len\n',test_data['Ofiicial Account Name'].describe())

_ = plt.hist(train_data['Ofiicial Account Name'], bins=200)
plt.xlabel('Text char count')
plt.title("TRAIN:Histogram of char count")
plt.show()

_ = plt.hist(test_data['Ofiicial Account Name'], bins=200)
plt.xlabel('Text char count')
plt.title("TEST:Histogram of char count")
plt.show()

