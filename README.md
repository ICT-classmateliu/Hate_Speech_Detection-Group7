<!-- 由 @Classmateliu 创建、编写及维护 -->

<!-- 这里的内容是标题及logo部分的内容 -->
<div align="center">
<img src="image/logo_image.png" width="120"/>
</div>
<h1 align="center">仇恨言论检测</h1>

<!-- 这里对应中英文切换部分 -->
<h4 align="center">
  简体中文 | <a href="https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/blob/main/README_en.md">English</a>
</h4>
<div align="center">

<!-- 这里对应小徽章部分（使用不依赖仓库可见性的静态徽章以避免 "repo not found"） -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v6.1.3-blue.svg)](https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/releases)
<!-- MIT License
别人：可以自由使用、修改、再发布、商用。
要求：保留原作者版权声明和许可证文本。
责任：作者不对使用后造成的问题负责。 -->

<!-- 小标题部分 -->
<strong>@Classmateliu 采用 Python 开发，基于若干工具包实现，尚未更新完成</strong>
</div>

## 项目概述 ##
本项目是《自然语言处理》的课程项目，项目相关要求如下：
<br/>__研究内容：__ 仇恨言论通常被理解为任何基于个人或群体身份的口头、书面或行为表达。换句话说，是指基于他人的宗教、民族、国籍、种族、肤色、血统、性别或其他身份特征，对其进行攻击、贬低或煽动仇视的言论
<br/>__关键技术：__ 仇恨语音检测通常是情感分类的任务。 因此，对于训练，可以通过在通常用于对情绪进行分类的数据上进行训练，来实现可以从特定文本中分类仇恨言论的模型
<br/>__评价标准：__ 准确率，精确率，召回率
<br/>
<br/>__注意:__ 

## 相关功能 ##
运行主文件存放在 main 文件夹中，文件名为  `hate_speech_detection_gpu` ， `app_gradio` ， `train_final_model` 。`train_final_model` 使用 PyTorch MLP ，GPU训练，并加入相似度匹配、类别权重等；`app_gradio` 使用 gradio 实现可视化，输入句子提取特征以提升精确度，导入 artifacts 以复现模型 ； `hate_speech_detection_gpu` 训练使用 GPU（PyTorch，XGBoost）以及CPU（sklearn），总训练时长约为 20-30min，内部模型如下表：

| 模型 | 输入特征 | 框架/库 | 使用GPU | 用途 | 训练方式 | 评价指标 |
| --- | --- | ---: | ---: | --- | --- | --- |
| 基准模型 (Baseline MLP) | 加权 TF-IDF | PyTorch | 是 | 作为最简单的参考模型 | 5 折交叉验证 | F1-score, Accuracy, Precision (macro), Recall (macro) |
| Gradient Boosting (GB) | 加权 TF-IDF | scikit-learn | 否，CPU | 参考模型 | 3 折交叉验证 | F1-score, Accuracy, Precision (micro), Recall (micro) |
| Random Forest (RF) | 完整特征 | scikit-learn | 否，CPU | 参考模型 | 3 折交叉验证 | F1-score, Accuracy, Precision (micro), Recall (micro) |
| XGBoost | 完整特征 | XGBoost | 是 | 集成 | 5 折交叉验证 | F1-score, Accuracy, Precision (micro), Recall (micro) |
| PyTorch MLP | 完整特征 | PyTorch | 是 | 主模型 | 完整可视化训练 | F1-score, Accuracy, Precision (micro), Recall (micro), ROC/AUC |
| Voting 集成 | 完整特征 | scikit-learn | 否，CPU | 集成学习 | MLP + RF + XGBoost（soft voting） | F1-score, Accuracy, Precision, Recall |
| Stacking 集成 | 完整特征 | mlxtend | 否，CPU | 集成学习 | MLP + RF + XGBoost，LogisticRegression 作为 meta-classifier | F1-score, Accuracy, Precision, Recall |

## 使用说明 ##
在运行此项目之前，需要安装以下库以及软件包:
- HanLp（需要自己申请API以提高调用次数）
<br/>官方Github：https://github.com/hankcs/HanLP
<br/>RESTful API申请：https://bbs.hanlp.com/t/hanlp2-1-restful-api/53
- NLTK
<br/>教程：https://book.itheima.net/course/221/1270308811000782849/1271374274807996418
- sklearn
<br/>教程：https://www.runoob.com/sklearn/sklearn-install.html
- pandas
<br/>教程：http://runoob.com/pandas/pandas-install.html
- PyTorch
- XGBoost
- numpy
<br/>教程：https://www.runoob.com/numpy/numpy-install.html
- mlxtend.classifier
<br/>教程：https://rasbt.github.io/mlxtend/installation/
- re
<br/>使用说明：https://blog.csdn.net/shadowtalon/article/details/139107806
- string
- json

## 词典和词汇表 ##
- 仇恨言论词典 `hatebase_dict.csv`
<br/>用于对推文进行采样，来自 https://www.hatebase.org/ 的原始词典。虽然该词典可以实现较高的召回率，但由于其中包含许多通常不会以冒犯或仇恨方式使用的词语（例如 yellow、oreo、bird），因此其误报率也较高
- 精简的n元语法-仇恨言论词典 `refined_ngram_dict.csv`
<br/>包含一个精简的n元语法词典。从标注数据中提取了长度为1到4的n元语法集合，并计算了每个n元语法在被人工编码员判定为仇恨言论的推文中所占的比例,删除了不相关的术语。
- 积极词汇表 `Positive_word.csv`
<br/>与积极观点/情感相关的词汇表，来自 https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English
- 消极词汇表 `negative_word.csv`
<br/>与负面观点/情绪相关的词汇表。来自 https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English

## 数据集 ##
英文数据集存储在 `initial_dataset_en` 文件夹中，数据集来自发表的论文：《自动化仇恨言论检测与冒犯性语言问题》（“Automated Hate Speech Detection and the Problem of Offensive Language”）收录于 ICWSM
 - 内部包含一个开源的数据集，数据集包含 24,784 条推文，推文由 CrowdFlower 用户手动标记为 `hate_speech`、`offensive_language` 或 `neither`
 - __数据集定义__
<br/> `index` 推文的唯一标识符
<br/> `count` 给这条推文贴标签的 CrowdFlower 用户总数
<br/> `hate_speech` 在 CrowdFlower 上将该推文标记为包含或构成仇恨言论的用户数量
<br/> `offensive_language` 在 CrowdFlower 上将该推文标记为包含或构成冒犯性语言的用户数量
<br/> `neither` CrowdFlower 用户中，认为该推文既非仇恨言论也非冒犯性语言的人数
<br/> `class` CrowdFlower 用户给出的多数标签（0 代表仇恨言论，1 代表冒犯性语言，2 代表两者都不是）
<br/> `tweet` 推文（文本形式）
<br/> `clean_tweet` 去除标点符号并转换为小写后的推文文本

中文数据集存储在 `initial_dataset_cn` 文件夹中，数据集来自发表的论文：基于RoBERTa的中文仇恨言论侦测方法研究，收录于CCL 2023
 - 数据集中包含了17430条标注好的句子，覆盖种族，性别，地域等主题。其中，label 0 代表安全，label 1 代表仇恨言论

## 特征数据集生成脚本 ##
本项目按照“文本预处理 → 句法分析 → 特征抽取”的流程构建模型训练所需的多类特征
 -  `clean_tweets.py` 对原始标注数据 labeled_data.csv 中的推文进行清洗与规范化处理，生成包含 clean_tweet 的基础数据集 cleaned_tweets.csv
 -  `stanford_nlp.py` 调用 Stanford CoreNLP 对每条推文执行依存句法解析，并将解析结果按推文 index 保存为 dependency_dict.json
 -  `dependency_features.py` 在此基础上统计各类依存关系的出现次数，生成依存句法特征表 dependency_features.csv
 - `ngram_features.py` 基于清洗后的文本提取词级与字符级 n-gram 特征以及词级 TF-IDF 特征，分别输出 word_bigram_features.csv、char_bigram_features.csv 和 tfidf_features.csv
 - `sentiment_scores.py` 利用仇恨词典与情感词典计算每条推文中仇恨词、消极词和积极词的命中次数及其归一化比例，生成情感相关数值特征 sentiment_scores.csv
 - `tf-idf.py` 进一步基于仇恨词典计算仇恨词 TF-IDF 累加得分，用于衡量推文的仇恨强度。上述脚本共同构成完整的特征工程流程，为后续模型训练与评估提供统一、可复现的输入特征

## 引用来源 ##
- 实现方案参考及README编写 https://github.com/aman-saha/hate-speech-detection/tree/master
- 词典 https://github.com/SunYanCN/hate-speech-and-offensive-language/tree/master/lexicons
- 英文数据集 https://github.com/t-davidson/hate-speech-and-offensive-language
- 中文数据集 https://github.com/RXJ588/CHSD/tree/main