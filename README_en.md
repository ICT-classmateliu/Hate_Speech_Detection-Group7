<!-- Created, written and maintained by @Classmateliu -->

<!-- Title and logo section -->
<div align="center">
<img src="image/logo_image.png" width="120"/>
</div>
<h1 align="center">Hate Speech Detection</h1>

<!-- Language toggle -->
<h4 align="center">
  English | <a href="https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/blob/main/README.md">简体中文</a>
:</h4>
<div align="center">

<!-- Badges -->
[![License](https://img.shields.io/github/license/ICT-classmateliu/Hate_Speech_Detection-Group7)](https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/ICT-classmateliu/Hate_Speech_Detection-Group7)](https://github.com/ICT-classmateliu/Hate_Speech_Detection-Group7/releases)
<!-- MIT License
Permissions: free to use, modify, redistribute, and commercial use.
Requirements: preserve original author copyright and license text.
Liability: author is not responsible for damages caused by use.
-->

<strong>@Classmateliu implemented in Python using several libraries</strong>
</div>

## Project Overview ##
This project is a course assignment for "Natural Language Processing". Project requirements are as follows:
<br/>__Research focus:__ Hate speech is commonly understood as any verbal, written, or behavioral expression targeting an individual or group based on identity. In other words, it refers to expressions that attack, demean, or incite hostility toward others based on religion, ethnicity, nationality, race, color, ancestry, gender, or other identity attributes.
<br/>__Key techniques:__ Hate speech detection is typically framed as a classification task. Therefore, models can be trained on datasets commonly used for sentiment/expression classification to build systems that classify hate speech in given texts.
<br/>__Evaluation metrics:__ Accuracy, Precision, Recall
<br/>
<br/>__Note:__

## Features ##
The main script is located in the `main` folder and named `hate_speech_detection_gpu`. Training uses GPU (PyTorch, XGBoost) and CPU (sklearn); total training time is about 20–30 minutes. The internal models are summarized in the table below:

| Model | Input features | Framework/Library | GPU Acceleration | Purpose | Training | Evaluation metrics |
| --- | --- | ---: | ---: | --- | --- | --- |
| Baseline MLP | Weighted TF-IDF (`weighted_TFIDF_scores`) | PyTorch | Optional (not required for small models) | Simple reference baseline | 5-fold cross-validation | F1-score, Accuracy, Precision (macro), Recall (macro) |
| Gradient Boosting (GB) | Weighted TF-IDF | scikit-learn | CPU | Reference model | 3-fold cross-validation | F1-score, Accuracy, Precision (micro), Recall (micro) |
| Random Forest (RF) | Full features | scikit-learn | CPU | Reference model | 3-fold cross-validation | F1-score, Accuracy, Precision (micro), Recall (micro) |
| XGBoost | Full features | XGBoost | Optional GPU | Reference model / ensemble | 3 or 5-fold cross-validation (depends on GPU usage) | F1-score, Accuracy, Precision (micro), Recall (micro) |
| PyTorch MLP (full features) | Full features | PyTorch | GPU | Main model | 5-fold cross-validation + final training | F1-score, Accuracy, Precision (micro), Recall (micro), ROC/AUC |
| scikit-learn MLPClassifier | Full features | scikit-learn | CPU | Ensemble sub-model | Final training | Used in Voting/Stacking ensemble |
| Voting ensemble | Full features | scikit-learn | CPU | Ensemble learning | MLP + RF + XGBoost (soft voting) | F1-score, Accuracy, Precision, Recall |
| Stacking ensemble | Full features | mlxtend | CPU | Ensemble learning | MLP + RF + XGBoost, LogisticRegression as meta-classifier | F1-score, Accuracy, Precision, Recall |

## Usage ##
Before running this project, install the following libraries and packages:
- HanLp (you need to apply for an API to increase request quota)
<br/>Official Github：https://github.com/hankcs/HanLP
<br/>RESTful API application：https://bbs.hanlp.com/t/hanlp2-1-restful-api/53
- NLTK
<br/>Tutorial：https://book.itheima.net/course/221/1270308811000782849/1271374274807996418
- sklearn
<br/>Tutorial：https://www.runoob.com/sklearn/sklearn-install.html
- pandas
<br/>Tutorial：http://runoob.com/pandas/pandas-install.html
- PyTorch
- XGBoost
- numpy
<br/>Tutorial：https://www.runoob.com/numpy/numpy-install.html
- mlxtend.classifier
<br/>Installation：https://rasbt.github.io/mlxtend/installation/
- re
<br/>Usage：https://blog.csdn.net/shadowtalon/article/details/139107806
- string
<br/>- json

## Dictionaries and Lexicons ##
- Hate speech dictionary `hatebase_dict.csv`
<br/>Used for sampling tweets, originating from https://www.hatebase.org/. Although this dictionary yields high recall, it contains many terms that are not typically used offensively or as hate speech (e.g., "yellow", "oreo", "bird"), so it also produces many false positives.
- Refined n-gram hate dictionary `refined_ngram_dict.csv`
<br/>Contains a refined n-gram dictionary. It extracts n-grams of length 1 to 4 from annotated data and computes the proportion of each n-gram appearing in tweets labeled as hate speech by human annotators, removing irrelevant terms.
- Positive lexicon `Positive_word.csv`
<br/>A lexicon associated with positive opinions/sentiments, from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English
- Negative lexicon `negative_word.csv`
<br/>A lexicon associated with negative opinions/sentiments, from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English

## Datasets ##
English datasets are stored in the `initial_dataset_en` folder. The dataset originates from the paper: "Automated Hate Speech Detection and the Problem of Offensive Language" published in ICWSM.
 - The repository includes an open dataset with 24,784 tweets manually labeled by CrowdFlower users as `hate_speech`, `offensive_language`, or `neither`.
 - __Dataset schema__
<br/> `index` unique identifier for the tweet
<br/> `count` total number of CrowdFlower users who labeled this tweet
<br/> `hate_speech` number of CrowdFlower users who labeled the tweet as containing or constituting hate speech
<br/> `offensive_language` number of CrowdFlower users who labeled the tweet as containing offensive language
<br/> `neither` number of CrowdFlower users who labeled the tweet as neither hate speech nor offensive language
<br/> `class` majority label assigned by CrowdFlower users (0 = hate_speech, 1 = offensive_language, 2 = neither)
<br/> `tweet` the tweet text
<br/> `clean_tweet` tweet text after removing punctuation and lowercasing

Chinese datasets are stored in the `initial_dataset_cn` folder. The dataset originates from the paper: "Research on Chinese Hate Speech Detection based on RoBERTa", published in CCL 2023.
 - The dataset contains 17,430 labeled sentences covering topics such as race, gender, and region. Label 0 indicates safe, label 1 indicates hate speech.

## Feature dataset generation scripts ##
The project uses four feature spaces: TF-IDF (weighted score and matrix), N-gram (character-level and word-level), typed dependency, and sentiment scores. Each feature space requires different preprocessing scripts as follows:
- `clean_tweets.py` 
- `stanford_nlp.py`
- `dependency_features.py` 
- `ngram_features.py` 
- `sentiment_scores.py` 
- `tf-idf.py` 

## References ##
- Implementation and README reference: https://github.com/aman-saha/hate-speech-detection/tree/master
- Lexicons: https://github.com/SunYanCN/hate-speech-and-offensive-language/tree/master/lexicons
- English dataset: https://github.com/t-davidson/hate-speech-and-offensive-language
- Chinese dataset: https://github.com/RXJ588/CHSD/tree/main


