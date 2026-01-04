from stanfordcorenlp import StanfordCoreNLP
import pandas as pd
import json

# 对 labeled_data.csv 中的每一条 tweet 做依存句法分析
# 并把结果存成一个 JSON 文件。
nlp = StanfordCoreNLP(r'/Users/tommypawelski/Desktop/stanford-corenlp-full-2018-02-27')
data=pd.read_csv('labeled_data.csv',encoding = 'ISO-8859-1')

new_dict = dict()

# tweet：文本内容 index：样本编号
for index, row in data.iterrows():
    tweet = str(row['tweet'])
    idx = str(row['index'])
    # 句子中第 2 个词是 root，第 1 个词是它的主语，第 3 个词是宾语
    new_dict[idx]=nlp.dependency_parse(tweet)
    # [
    #   ('nsubj', 2, 1),
    #   ('root', 0, 2),
    #   ('dobj', 2, 3)
    # ]

json = json.dumps(new_dict)
f = open("dependency_dict.json","w")
f.write(json)

f.close()
nlp.close()

# 依存关系能捕捉：攻击对象（who is attacked） 攻击动作（verbs）