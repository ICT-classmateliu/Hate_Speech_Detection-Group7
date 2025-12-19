<div align="center">
<img src="image/logo_image.png" width="120"/>
</div>
<h1 align="center">Hate Speech Detection Based on HanLP</h1>
<h4 align="center">
  <a href="https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/blob/main/README.md">简体中文</a> | English
</h4>
<div align="center">

<!-- 这里对应小徽章部分 -->
[![License](https://img.shields.io/github/license/ICT-classmateliu/Hate_Speech_Detection-Group7)](https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/ICT-classmateliu/Hate_Speech_Detection-Group7)](https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/releases)
<!-- MIT License
别人：可以自由使用、修改、再发布、商用。
要求：保留原作者版权声明和许可证文本。
责任：作者不对使用后造成的问题负责。 -->

 <!-- Subtitle -->
 <strong>Implemented in Python by @Classmateliu, based on the HanLP toolkit</strong>
</div>

## Project Overview
This project is a course project for *Natural Language Processing*. The related requirements are as follows:  
__Research content:__ Hate speech is generally understood as any verbal, written or behavioral expression based on the identity of an individual or group. In other words, it refers to speech that attacks, demeans or incites hatred against others on the basis of their religion, ethnicity, nationality, race, skin color, ancestry, gender or other identity characteristics.  
__Key techniques:__ Hate speech detection is usually treated as a sentiment classification task. Therefore, for training, a model that can classify hate speech from specific text can be obtained by training on datasets that are commonly used for emotion classification.  
__Evaluation metrics:__ Accuracy, Precision and Recall.
<br/>
<br/>__Note:__ 

## Functions

## Usage
Before running this project, the following libraries and packages need to be installed:  
- HanLP (you need to apply for your own API key to increase the number of calls)  
Official GitHub: https://github.com/hankcs/HanLP  
RESTful API application: https://bbs.hanlp.com/t/hanlp2-1-restful-api/53  
- NLTK  
Tutorial: https://book.itheima.net/course/221/1270308811000782849/1271374274807996418  
- sklearn  
Tutorial: https://www.runoob.com/sklearn/sklearn-install.html  
- pandas  
Tutorial: http://runoob.com/pandas/pandas-install.html  
- numpy  
Tutorial: https://www.runoob.com/numpy/numpy-install.html  
- mlxtend.classifier  
Tutorial: https://rasbt.github.io/mlxtend/installation/  
- re  
Documentation: https://blog.csdn.net/shadowtalon/article/details/139107806  
- string  
- json

## Dictionaries and Lexicons
- Hate speech lexicon `hatebase_dict.csv`  
Used for sampling tweets, derived from the original lexicon on https://www.hatebase.org/. Although this lexicon can achieve a relatively high recall, it also has a high false positive rate because it contains many terms that are not usually used in an offensive or hateful way (for example, yellow, oreo, bird).  
- Refined n-gram hate speech lexicon `refined_ngram_dict.csv`  
Contains a refined n-gram lexicon. N-grams of length 1 to 4 are extracted from annotated data and the proportion of each n-gram appearing in tweets labeled as hate speech by human coders is computed, and unrelated terms are removed.  
- Positive lexicon `Positive_word.csv`  
A lexicon of words related to positive opinions / sentiments, from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English  
- Negative lexicon `negative_word.csv`  
A lexicon of words related to negative opinions / sentiments, from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English  

## References
- Lexicons: https://github.com/SunYanCN/hate-speech-and-offensive-language/tree/master/lexicons  
- Implementation reference: https://github.com/aman-saha/hate-speech-detection/tree/master

