# 这是一个测试文档 用于测试各个安装包的安装情况
# 由 @Classmateliu 编写

# 库的import部分 ------------------------------------------------------------
import nltk

from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://www.hanlp.com/api', auth='OTcxN0BiYnMuaGFubHAuY29tOjBLSVpEN05JQ2ZYSG02cnk=', language='zh') 
# auth不填则匿名，zh中文，mul多语种

import sklearn

import pandas as pd

import numpy as np
import string
import json
import re

# NLTK库测试部分 ------------------------------------------------------------
#nltk.download()  # 打开NLTK下载器
from nltk.corpus import brown    # 导入brown语料库
print(brown.words())             # 查看brown库中所有的单词
print(brown.categories())        # 通过categories()函数查看brown中包含的类别
print('brown中一共有{}个句子'.format(len(brown.sents())))  
# 通过sents()函数查看brown中包含的句子
print('brown中一共有{}个单词'.format(len(brown.words())))
# 通过words()函数查看brown中包含的单词

# HanLp库测试部分 ------------------------------------------------------------
#doc = HanLP.parse('2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。')
#doc.pretty_print()

#中文分词 细粒度分词
#HanLP('商品和服务。阿婆主来到北京立方庭参观自然语义科技公司。', tasks='tok').pretty_print()

#粗粒度分词
#HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks='tok/coarse').pretty_print()
#fine为细分，coarse为粗分

# sklearn库测试部分 ------------------------------------------------------------
print(sklearn.__version__)

# pandas库测试部分 ------------------------------------------------------------
pd.__version__  # 查看版本
mydataset = {
  'sites': ["Google", "Runoob", "Wiki"],
  'number': [1, 2, 3]
}

myvar = pd.DataFrame(mydataset)
print(myvar)

# numpy库测试部分 ------------------------------------------------------------
print(np.__version__)

