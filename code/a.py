import xmnlp
import pandas as pd
import jieba
from string import punctuation
punctuation = r"""!"#$%&'()*+,-.:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；/《》’‘……￥·"""

#去除文本中所有的标点符号
def FilterPunctuation(thetext):
    s =thetext
    dicts={i:'' for i in punctuation}
    punc_table=str.maketrans(dicts)
    new_s=s.translate(punc_table)
    return new_s

# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

'''
应用结巴分词，将新闻Ofiicial Account Name和Title进行分词
'''
def seperate(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    # 这里加载停用词的路径，去除停用词
    stopwords = stopwordslist(r'E:\PYTHONDATA\stopword.txt')
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ' '
    outstr = str(FilterPunctuation(outstr))
    return outstr

'''
应用轻量级中文自然语言处理工具xmnlp进行['Report Content']的情感分析
如果评论是正向，返回1，否则返回0
这样所有字符串类型的评价都被转变成0或1，实现了数值化
'''
def SentimentAnalysis(text):
    text = text
    x = xmnlp.sentiment(text)
    if x[0] >= x[1]:#评价为负向情感
        thevalue = 0
    else:#评价为正向情感
        thevalue = 1
    text = thevalue
    return text

if __name__=='__main__':
    '''
    print(train_data)
    print(train_data.info())
    print(train_data.dtypes)
    '''#输出Pandas的内容和数据类型
    xmnlp.set_model(r'D:\BaiduNetdiskDownload\xmnlp-onnx-models')
    train_data=pd.read_csv(r'E:\pythonproject\WeFEND-AAAI20\data\train\news.csv')
    test_data=pd.read_csv(r'E:\pythonproject\WeFEND-AAAI20\data\test\news.csv')
    #数据清洗，丢弃无关标签
    train_data.drop(['News Url','Image Url'],axis=1,inplace=True)
    test_data.drop(['News Url','Image Url'],axis=1,inplace=True)

    '''
    #查找空数据
    print(train_data.isnull().sum())
    print(test_data.isnull().sum())
    #得到的结果显示，空标签都出现在['Official Account Name'],数量相对较少，保留与否不影响最终结果，决定保留

    # 查找可替换的内容
    print(train_data['Ofiicial Account Name'].describe())
    # 得到出现最多次数的是‘愚小九’
    print(test_data['Ofiicial Account Name'].describe())
    # 得到出现最多次数的是‘娱记说’

    # 这条数据我们希望保留，那么都选择出现次数最多的进行替换
    #数据清洗
    '''
    train_data['Ofiicial Account Name'].fillna('愚小九',inplace=True)
    test_data['Ofiicial Account Name'].fillna('娱记说',inplace=True)

    train_data['Report Content']=list(map(SentimentAnalysis,train_data['Report Content']))
    test_data['Report Content']=list(map(SentimentAnalysis,test_data['Report Content']))

    train_data['Title']=list(map(seperate,train_data['Title']))
    test_data['Title']=list(map(seperate,test_data['Title']))

    # 查看评论的情感分析结果
    print('train_data\n', train_data['Report Content'].value_counts())
    print('test_data\n', test_data['Report Content'].value_counts())

    train_data.drop(['Report Content'], axis=1, inplace=True)
    test_data.drop(['Report Content'], axis=1, inplace=True)

    train_data.to_csv(r'E:\PYTHONDATA\train_data.csv',encoding='utf-8')
    test_data.to_csv(r'E:\PYTHONDATA\test_data.csv', encoding='utf-8')

