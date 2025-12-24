<!-- 由 @Classmateliu 创建、编写及维护 -->

<!-- 这里的内容是标题及logo部分的内容 -->
<div align="center">
<img src="image/logo_image.png" width="120"/>
</div>
<h1 align="center">基于HanLP的仇恨言论检测</h1>

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
<strong>@Classmateliu 采用 Python 开发，基于 HanLP 工具包实现</strong>
</div>

## 项目概述 ##
本项目是《自然语言处理》的课程项目，项目相关要求如下：
<br/>__研究内容：__ 仇恨言论通常被理解为任何基于个人或群体身份的口头、书面或行为表达。换句话说，是指基于他人的宗教、民族、国籍、种族、肤色、血统、性别或其他身份特征，对其进行攻击、贬低或煽动仇视的言论
<br/>__关键技术：__ 仇恨语音检测通常是情感分类的任务。 因此，对于训练，可以通过在通常用于对情绪进行分类的数据上进行训练，来实现可以从特定文本中分类仇恨言论的模型
<br/>__评价标准：__ 准确率，精确率，召回率
<br/>
<br/>__注意:__ 

## 相关功能 ##

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
项目中使用四种特征空间，包括 TF-IDF（加权得分和矩阵）、N-gram（字符级和词级）、类型依存关系和情感得分。每种特征空间都需要使用不同的文本语料库预处理脚本，具体如下：
- `clean_tweets.py` 对初始数据集执行常规预处理步骤，包括去除标点符号和将所有字符转换为小写
- `stanford_nlp.py` 使用斯坦福解析器的类型依存关系工具保留推文中的词语结构，该工具可以识别句子中词语之间的句法关系。此脚本在数据集上运行斯坦福解析器，并将每条推文的类型依存关系以字典的形式返回（每个键代表推文索引，每个值是该推文的类型依存关系列表）。然后，该字典以 .json 文件 dependency_dict.json 的形式输出
- `dependency_features.py` 读取包含类型化依赖关系的 .json 文件，并创建基于依赖关系的特征。斯坦福解析器识别出的每条推文之间的关系产生了 41 个特征，每个特征存储了每种不同类型依赖关系的计数
- `ngram_features.py` 使用计数向量化器在词级和字符级创建二元语法特征。计数向量化器也用于创建 TF-IDF 矩阵。对于 TF-IDF、词级二元语法和字符级二元语法，停用词被移除，剩余的词干被提取。随后，对于词级二元语法和 TF-IDF，符号被移除，以确保像“#Selfie”这样的词与“Selfie”的计数相同。然而，数字并未从文本中移除，以保留 Unicode 形式的表情符号作为词。另一方面，对于字符级二元语法，符号被保留在文本中，以捕捉通常用于审查脏话（例如“b*tch”或“a$$hole”）的印刷符号。在构建字符级二元语法时，数字被移除，因为 Unicode 表示在字符级上没有价值
- `sentiment_scores.py` 统计每条推文中出现在仇恨词汇库、负面词汇库和正面词汇库中的词汇数量。由于一些包含负面词汇的推文也可能表达正面含义（例如“真tm棒”），因此添加了正面词汇库以进行更全面的情感分析。此外，情感得分（即正面和负面词频的归一化）的计算方法是将推文长度除以词频，其依据是：在负面/正面词汇频率相同的情况下，较短的推文比较长的推文表达更强烈的情感
- `tf-idf.py` TF-IDF分数是基于词频和逆文档频率计算得出的，作为基线模型中唯一包含的特征集。对于每条推文，计算每个词的 TF-IDF 值，如果该词出现在仇恨词汇库中，则赋予其权重 1，否则赋予权重 0。权重设置为 1 和 0 是因为我们只关注出现在词汇库中的词。然后，将所有 TF-IDF 值乘以其对应的权重，并将结果相加，得到每条推文的 TF-IDF 分数

## 引用来源 ##
- 实现方案参考及README编写 https://github.com/aman-saha/hate-speech-detection/tree/master
- 词典 https://github.com/SunYanCN/hate-speech-and-offensive-language/tree/master/lexicons
- 英文数据集 https://github.com/t-davidson/hate-speech-and-offensive-language
- 中文数据集 https://github.com/RXJ588/CHSD/tree/main