import json
import pandas as pd

# 把「每条 tweet 的依存句法分析结果」转换成 可用于机器学习的数值特征向量（dependency count features），并保存为 CSV
# 读取依存分析结果和 tweet 数据
dependency_dict = json.loads(open("dependency_dict.json").read())
data=pd.read_csv('cleaned_tweets.csv',encoding = 'ISO-8859-1')

# 收集「所有出现过的依存关系类型」确定 特征空间维度
dependency_types=set()
for key, values in dependency_dict.items():
    for v in list(values):
        dependency_types.add(list(v)[0])

# 为每一种依存关系创建一列（全 0）
for type in dependency_types:
    data[str(type)] = 0

# 统计每条 tweet 中的依存关系数量
# 最终得到的是一个 Bag-of-Dependencies（依存关系词袋）
for index, row in data.iterrows():
    tweet = str(row['tweet'])
    clean_tweet = str(row['clean_tweet'])
    idx = str(row['index'])
    dependeny_vec = dependency_dict[idx]
    for dependency in dependeny_vec:
        data.loc[index, str(dependency[0])] += 1

# 给所有列加前缀 & 保存
data = data.add_prefix('dependecy:')
data.columns.values

data.to_csv("dependency_features.csv")
