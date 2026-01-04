from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import string
import re
import pandas as pd
import numpy as np
from nltk import word_tokenize

# 你这段代码一共构建了 三类文本特征：
# 词级二元语法（word bigrams）
# 字符级二元语法（char bigrams）
# TF-IDF 词频特征（unigram）

data = pd.read_csv('cleaned_tweets.csv',encoding='utf-8')

# 数据读取与词干提取（Stemming）
# 对 清洗后的 tweet 做英文词干提取：hating, hated, hate → hate
# 减少词表规模，提升泛化能力
stemmer = SnowballStemmer("english")
data['stemmed'] = data.clean_tweet.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

# 词级二元语法（Word-level Bigrams）
cv = CountVectorizer(stop_words='english', min_df=.002, max_df=.8, ngram_range=(2,2))
cv.fit(data.stemmed)
cv_mat = cv.transform(data.stemmed)

# 构造特征矩阵
# 典型的稀疏高维文本特征空间，非常适合 MLP / 线性模型
bigrams = pd.DataFrame(cv_mat.todense(), index=data['index'], columns=cv.get_feature_names())
bigrams = bigrams.add_prefix('word_bigrams:')
bigrams.to_csv('word_bigram_features.csv')

# 统计非零元素数量 & 稀疏度
print ('Non-zero count:', cv_mat.nnz)
print ('Sparsity: %.2f%%' % (100.0 * cv_mat.nnz / (cv_mat.shape[0] * cv_mat.shape[1])))

oc = np.asarray(cv_mat.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'Term': cv.get_feature_names(), '# occurrences': oc})
counts_df.sort_values(by='# occurrences', ascending=False).head(20)

# 字符级二元语法（Char-level Bigrams）
# 先去掉数字（emoji unicode）
data['char_stem'] = data.tweet.apply(lambda x: x.translate(str.maketrans('','',string.digits)))

# 字符级 CountVectorizer
# char bigram 往往是仇恨言论检测中最有效的传统特征
cv_char = CountVectorizer(analyzer='char', stop_words='english',min_df=.002, max_df=.8,ngram_range=(2,2))
cv_char.fit(data.char_stem)
cv_char_mat = cv_char.transform(data.char_stem)


char_bigrams = pd.DataFrame(cv_char_mat.todense(), index=data['index'], columns=cv_char.get_feature_names())
char_bigrams = char_bigrams.add_prefix('char_bigrams:')

char_bigrams.to_csv('char_bigram_features.csv')

print ('Non-zero count:', cv_char_mat.nnz)
print ('Sparsity: %.2f%%' % (100.0 * cv_char_mat.nnz / (cv_char_mat.shape[0] * cv_char_mat.shape[1])))
oc2 = np.asarray(cv_char_mat.sum(axis=0)).ravel().tolist()
counts_df2 = pd.DataFrame({'Term': cv_char.get_feature_names(), '# occurrences': oc2})
counts_df2.sort_values(by='# occurrences', ascending=False).head(20)

# TF-IDF 特征（Unigram）
cv = CountVectorizer(stop_words='english', min_df=.002, max_df=.8, ngram_range=(1,1))
cv.fit(data.stemmed)
cv_mat = cv.transform(data.stemmed)

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(cv_mat)

weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cv.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(80)
transformed_weights.toarray()

tf_idf =pd.DataFrame(transformed_weights.todense(), index=data['index'], columns=cv.get_feature_names())

tf_idf = tf_idf.add_prefix('tfidf:')

tf_idf.to_csv('tfidf_features.csv')

# 特征文件拆分保存
# word_bigram_features.csv
# char_bigram_features.csv
# tfidf_features.csv
