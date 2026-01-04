import pandas as pd
import re
import string

data=pd.read_csv('labeled_data.csv',encoding = 'ISO-8859-1')

# 转小写
# 把 @提及、非字母数字字符（包括标点、表情等）和 URL 替换为空格，然后去重多余空格
clean_tweets = []
for index, row in data.iterrows():
    tweet = str(row['tweet']).lower()
    clean_tweets.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split()))

data['clean_tweet'] = clean_tweets

# 把整个表保存为 cleaned_tweets.csv
data.to_csv("cleaned_tweets.csv", index=False)
