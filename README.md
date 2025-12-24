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

<!-- 这里对应小徽章部分 -->
[![License](https://img.shields.io/github/license/ICT-classmateliu/Hate_Speech_Detection-Group7)](https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/ICT-classmateliu/Hate_Speech_Detection-Group7)](https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/releases)
<!-- MIT License
别人：可以自由使用、修改、再发布、商用。
要求：保留原作者版权声明和许可证文本。
责任：作者不对使用后造成的问题负责。 -->

<!-- 小标题部分 -->
<strong>@Classmateliu 采用 Python 开发，基于若干工具包实现</strong>
</div>

## 项目概述 ##
本项目是《自然语言处理》的课程项目，项目相关要求如下：
<br/>__研究内容：__ 仇恨言论通常被理解为任何基于个人或群体身份的口头、书面或行为表达。换句话说，是指基于他人的宗教、民族、国籍、种族、肤色、血统、性别或其他身份特征，对其进行攻击、贬低或煽动仇视的言论
<br/>__关键技术：__ 仇恨语音检测通常是情感分类的任务。 因此，对于训练，可以通过在通常用于对情绪进行分类的数据上进行训练，来实现可以从特定文本中分类仇恨言论的模型
<br/>__评价标准：__ 准确率，精确率，召回率
<br/>
<br/>__注意:__ 

## 相关功能 ##
运行主文件存放在 main 文件夹中，文件名为 `hate_speech_detection_gpu` ，训练使用GPU（PyTorch，XGBoost）以及CPU（sklearn），总训练时长约为 20-30min，内部模型如下表：

| 模型 | 输入特征 | 框架/库 | GPU 加速 | 用途 | 训练方式 | 评价指标 |
| --- | --- | ---: | ---: | --- | --- | --- |
| 基准模型 (Baseline MLP) | 加权 TF-IDF (`weighted_TFIDF_scores`) | PyTorch | 可选（小模型不必要） | 作为最简单的参考模型 | 5 折交叉验证 | F1-score, Accuracy, Precision (macro), Recall (macro) |
| Gradient Boosting (GB) | 加权 TF-IDF | scikit-learn | CPU | 参考模型 | 3 折交叉验证 | F1-score, Accuracy, Precision (micro), Recall (micro) |
| Random Forest (RF) | 完整特征 | scikit-learn | CPU | 参考模型 | 3 折交叉验证 | F1-score, Accuracy, Precision (micro), Recall (micro) |
| XGBoost | 完整特征 | XGBoost | 可选 GPU | 参考模型 / 集成 | 3 或 5 折交叉验证（视是否使用 GPU） | F1-score, Accuracy, Precision (micro), Recall (micro) |
| PyTorch MLP (完整特征) | 完整特征 | PyTorch | GPU | 主要模型 | 5 折交叉验证 + 最终训练 | F1-score, Accuracy, Precision (micro), Recall (micro), ROC/AUC |
| scikit-learn MLPClassifier | 完整特征 | scikit-learn | CPU | 集成学习子模型 | 最终训练 | 参与 Voting/Stacking 集成 |
| Voting 集成 | 完整特征 | scikit-learn | CPU | 集成学习 | MLP + RF + XGBoost（soft voting） | F1-score, Accuracy, Precision, Recall |
| Stacking 集成 | 完整特征 | mlxtend | CPU | 集成学习 | MLP + RF + XGBoost，LogisticRegression 作为 meta-classifier | F1-score, Accuracy, Precision, Recall |

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
项目中使用四种特征空间，包括 TF-IDF（加权得分和矩阵）、N-gram（字符级和词级）、类型依存关系和情感得分。每种特征空间都需要使用不同的文本语料库预处理脚本
- `clean_tweets.py` 
- `stanford_nlp.py` 
- `dependency_features.py` 
- `ngram_features.py` 
- `sentiment_scores.py` 
- `tf-idf.py` 

## 引用来源 ##
- 实现方案参考及README编写 https://github.com/aman-saha/hate-speech-detection/tree/master
- 词典 https://github.com/SunYanCN/hate-speech-and-offensive-language/tree/master/lexicons
- 英文数据集 https://github.com/t-davidson/hate-speech-and-offensive-language
- 中文数据集 https://github.com/RXJ588/CHSD/tree/main