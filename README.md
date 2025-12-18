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

## 概述 ##
本项目是《自然语言处理》的课程项目，项目相关要求如下：
<br/>__研究内容：__ 仇恨言论通常被理解为任何基于个人或群体身份的口头、书面或行为表达。换句话说，是指基于他人的宗教、民族、国籍、种族、肤色、血统、性别或其他身份特征，对其进行攻击、贬低或煽动仇视的言论
<br/>__关键技术：__ 仇恨语音检测通常是情感分类的任务。 因此，对于训练，可以通过在通常用于对情绪进行分类的数据上进行训练，来实现可以从特定文本中分类仇恨言论的模型
<br/>__评价标准：__ 准确率，精确率，召回率
<br/>
<br/>__注意:__ 

## 相关功能 ##

## 关于 ##
在运行此项目之前，需要安装以下库以及软件包:
- HanLp（需要自己申请API以提高调用次数）
<br/>官方Github：https://github.com/hankcs/HanLP
<br/>RESTful API申请：https://bbs.hanlp.com/t/hanlp2-1-restful-api/53
- NLTK
<br/>教程：https://book.itheima.net/course/221/1270308811000782849/1271374274807996418
- sklearn
<br/>教程：https://www.runoob.com/sklearn/sklearn-install.html
- stanfordcorenlp（配置版本出现问题 此处暂不涉及）
- pandas
<br/>教程：http://runoob.com/pandas/pandas-install.html
- numpy
<br/>教程：https://www.runoob.com/numpy/numpy-install.html
- mlxtend.classifier
<br/>教程：https://rasbt.github.io/mlxtend/installation/
- re
<br/>使用说明：https://blog.csdn.net/shadowtalon/article/details/139107806
- string
- json
