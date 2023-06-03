# PowerShell Detect
恶意Powershell脚本检测的静态方法，ACC=95+%
## 深度学习方法
使用30w的PoweShell数据进行word2Vector预训练后，在17k有标注的数据上进行恶意样本检测
模型：BiLstm
## 机器方法
TFIDF进行数据处理后，使用XGB进行分类。数据多了之后处理起来比较麻烦，跑了个demo之后就没继续用了ACC也可以达到90+%
## 预训练数据集
### 下载链接
可能需要科x学x上x网x

https://aka.ms/PowerShellCorpus  这是30W样本量的powershell数据集

https://github.com/das-lab/mpsd  这是1W+的有标注powershell数据集
### 数据集的介绍

You can find the details behind the data science aspects of this work in the 'DataScience' subdirectory of the repository

未标注的样本数量30w
### 预训练数据集预处理方式

#### 字符串预处理
1. 数字替换：所有的数字都被替换成"*"，用以处理随机值，IP地址，随机域名，日期，版本号等
2. 使用word2vec和fasttext进行词向量的训练 
3. 只有出现在至少十个实例中的token才用于训练词向量。一共有大概81111个不同的token

#### 词向量预训练

1. 使用CBOW方式进行训练，训练速度更快，并且通常在包含许多频繁词的大型训练集上效果更好
####  预训练样本去重

实际上我没有使用，按道理来说去重后效果更好

有标注的数据要消除在未标注数据中出现过得相同（或者几乎相同）的数据，会影响交叉验证，不管是良性还是恶意数据都有可能有非常相似的两段代码

重复数据删除过程包括以下 4 个阶段：
1. 不在集合 {'a'-'z', 'A '-'Z', '*', '$', '-'} 中的任何符号都用作分隔符，只使用长度大于2的token，所有的都标准化为小写
2. 删除低频token：仅保留出现在100+段代码里面的token，收集了14216个token（只是为了重复数据删除，此类token仍用于训练嵌入层和评估模型）
3. 聚类：这段没怎么仔细讲，主要是通过低频token和保留的高频token的来进行聚类
4. 聚类成了116976，也就是说这些都是“独一无二”的
## 实际分类训练用的数据集

标注的数据，黑7k+，白1w加

## 训练

1. 输入token长度，前2000个token，Embedding后输出到BiLstm中，进行分类，详见代码
2. 损失函数和训练策略几乎没有，都是最常见的方法，个人觉得这个重点在预处理这一块
