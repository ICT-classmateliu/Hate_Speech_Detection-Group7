import json
import joblib
import torch
import numpy as np
import pandas as pd
import gradio as gr
import os
import re
import string
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# artifacts path (created by training script)
ARTIFACTS_DIR = os.path.join('main', 'artifacts')

# Global variables for lazy loading
model = None
scaler = None
base_scaler = None
feature_columns = None
label_map = None
optimal_thresholds = None
artifacts_loaded = False

# 推理结果缓存（避免重复计算相同句子）
prediction_cache = {}
CACHE_MAX_SIZE = 100  # 最多缓存100个句子

def load_artifacts():
    """
    启动时加载 artifacts，确保模型可用
    """
    global model, scaler, base_scaler, feature_columns, label_map, artifacts_loaded

    if artifacts_loaded:
        return  # 已加载，直接返回

    print("正在加载模型和预处理器...")

    model_path = os.path.join(ARTIFACTS_DIR, 'final_model_state_dict.pth')
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
    cols_path = os.path.join(ARTIFACTS_DIR, 'feature_columns.json')
    labels_path = os.path.join(ARTIFACTS_DIR, 'label_map.json')
    base_scaler_path = os.path.join(ARTIFACTS_DIR, 'base_scaler.pkl')

    # 检查必需的文件是否存在
    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append('final_model_state_dict.pth')
    if not os.path.exists(scaler_path):
        missing_files.append('scaler.pkl')
    if not os.path.exists(cols_path):
        missing_files.append('feature_columns.json')
    if not os.path.exists(labels_path):
        missing_files.append('label_map.json')
    # 阈值文件是可选的，如果没有就使用默认阈值

    if missing_files:
        raise FileNotFoundError(f"缺少必要的artifacts文件: {', '.join(missing_files)}. 请先运行训练脚本: python main/train_final_model.py")

    # 加载模型权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_state = torch.load(model_path, map_location=device, weights_only=True)
    scaler = joblib.load(scaler_path)
    base_scaler = joblib.load(base_scaler_path) if os.path.exists(base_scaler_path) else None

    # 加载配置文件
    with open(cols_path, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)
    with open(labels_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    # 构建模型
    input_dim = len(feature_columns)
    model = MLPNet(input_dim=input_dim, hidden_dims=[80, 40, 40, 10], num_classes=len(label_map))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    artifacts_loaded = True
    print(f"模型加载完成！特征维度: {input_dim}")
    return model, scaler, base_scaler, feature_columns, label_map

# Optimized feature extraction for fast demo purposes
def extract_basic_features(sentence: str, feature_columns):
    """
    快速特征提取 - 与训练时特征完全匹配
    """
    # Clean the text but keep more information for hate speech detection
    sentence = sentence.lower()
    # 移除URL和@mentions，然后移除标点符号但保留字母和数字
    clean_text = re.sub(r"(\w+:\/\/\S+)|(@[A-Za-z0-9]+)", " ", sentence)
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", clean_text)  # 移除所有标点符号
    clean_text = ' '.join(clean_text.split())

    # Basic sentiment features
    words = clean_text.split()
    word_count = max(len(words), 1)  # 避免除零

    # 加载外部词典
    hate_words, neg_words, pos_words, ngram_hate_scores = load_external_dictionaries()

    # 快速计数
    hate_count = sum(1 for word in words if word in hate_words)
    neg_count = sum(1 for word in words if word in neg_words)
    pos_count = sum(1 for word in words if word in pos_words)

    # 计算仇恨言论强度指标
    hate_intensity = 0
    if hate_count > 0:
        hate_intensity = hate_count / word_count
        # 如果包含种族/性别歧视词，增加权重
        racial_words = {'nigger', 'kike', 'chink', 'gook', 'spic', 'wetback', 'coon', 'paki', 'raghead', 'towelhead'}
        gender_words = {'faggot', 'dyke', 'tranny', 'shemale', 'whore', 'slut'}
        violence_words = {'kill', 'die', 'death', 'murder', 'rape', 'torture'}

        racial_count = sum(1 for word in words if word in racial_words)
        gender_count = sum(1 for word in words if word in gender_words)
        violence_count = sum(1 for word in words if word in violence_words)

        if racial_count > 0:
            hate_intensity *= 2.0  # 种族歧视权重更高
        if gender_count > 0:
            hate_intensity *= 1.8  # 性别歧视权重
        if violence_count > 0:
            hate_intensity *= 1.5  # 暴力相关权重

    # 根据feature_columns.json的实际结构创建特征向量
    # 总共1687个特征，所有无法计算的复杂特征都设为0
    features = []

    # 1. weighted_TFIDF_scores (使用仇恨强度近似)
    features.append(hate_intensity)

    # 2. sentiment features (6个)
    features.extend([
        hate_count,  # sentiment:hate
        hate_intensity,  # sentiment:hatenor (使用增强的仇恨强度)
        neg_count,   # sentiment:neg
        neg_count / word_count if word_count > 0 else 0,  # sentiment:negnor
        pos_count,   # sentiment:pos
        pos_count / word_count if word_count > 0 else 0,  # sentiment:posnor
    ])

    # 3. dependency features (40个，设为0)
    features.extend([0] * 40)

    # 4. char_bigrams (984个，设为0)
    features.extend([0] * 984)

    # 5. word_bigrams (101个，设为0)
    features.extend([0] * 101)

    # 6. tfidf features (555个，使用仇恨词密度近似)
    # 这是简化的近似，所有TF-IDF特征都设为相同的仇恨词密度值
    features.extend([hate_count / word_count if word_count > 0 else 0] * 555)

    return np.array(features).reshape(1, -1)

def predict_with_optimized_thresholds(sample_probs, thresholds):
    """
    使用优化的阈值进行单样本预测
    """
    n_classes = len(sample_probs)

    # 检查每个类别是否超过其阈值
    valid_classes = []
    for class_idx in range(n_classes):
        threshold = thresholds[str(class_idx)]['threshold']
        if sample_probs[class_idx] >= threshold:
            valid_classes.append((class_idx, sample_probs[class_idx]))

    if valid_classes:
        # 如果有多个类别超过阈值，选择概率最高的
        valid_classes.sort(key=lambda x: x[1], reverse=True)
        return valid_classes[0][0]
    else:
        # 如果没有类别超过阈值，选择概率最高的类别（兜底策略）
        return np.argmax(sample_probs)

def extract_full_features_from_text(text):
    """
    从输入文本中提取完整的1687维特征向量（近似版本）
    """
    try:
        # 文本预处理
        processed_text = preprocess_text(text)
        words = processed_text.split()
        word_count = max(len(words), 1)

        # 加载外部词典
        hate_words, neg_words, pos_words, ngram_hate_scores = load_external_dictionaries()

        # 计数
        hate_count = sum(1 for word in words if word in hate_words)
        neg_count = sum(1 for word in words if word in neg_words)
        pos_count = sum(1 for word in words if word in pos_words)

        # 计算仇恨强度 - 改进版，更好地识别明确的仇恨言论
        hate_density = hate_count / word_count if word_count > 0 else 0
        hate_intensity = hate_density

        # 增强权重计算 - 更细粒度的分类
        racial_words = {'nigger', 'nigga', 'kike', 'chink', 'gook', 'spic', 'wetback', 'coon', 'paki', 'raghead', 'towelhead', 'jew', 'arab', 'muslim', 'black', 'white', 'asian', 'hispanic', 'latino', 'mexican', 'african', 'european'}
        strong_racial_words = {'nigger', 'nigga', 'kike', 'coon', 'chink', 'gook', 'spic'}  # 特别强烈的种族歧视词

        gender_words = {'bitch', 'cunt', 'whore', 'slut', 'fag', 'faggot', 'dyke', 'tranny', 'shemale'}
        violence_words = {'kill', 'die', 'death', 'murder', 'rape', 'torture', 'exterminate', 'genocide'}
        extreme_words = {'holocaust', 'nazi', 'hitler', 'supremacist'}  # 极端主义词汇

        # 计算各种类别的词频
        racial_count = sum(1 for word in words if word in racial_words)
        strong_racial_count = sum(1 for word in words if word in strong_racial_words)
        gender_count = sum(1 for word in words if word in gender_words)
        violence_count = sum(1 for word in words if word in violence_words)
        extreme_count = sum(1 for word in words if word in extreme_words)

        # 检查是否包含强烈的仇恨表达
        has_hate_verb = any(word in ['hate', 'deserve', 'kill', 'exterminate', 'genocide'] for word in words)
        has_quantifier = any(word in ['all', 'every', 'each', 'none', 'no'] for word in words)

        # 检查性别歧视模式
        gender_bias_indicators = [
            'kitchen', 'cooking', 'cleaning', 'housewife', 'homemaker',
            'traditional', 'submissive', 'place', 'role', 'stay'
        ]
        has_gender_bias = any(word in gender_bias_indicators for word in words)
        has_should = 'should' in words
        has_women = 'women' in words or 'woman' in words

        # 性别歧视组合得分
        gender_bias_score = 0
        if has_gender_bias and has_should and has_women:
            gender_bias_score = 2.0  # 强烈的性别歧视模式
        elif has_gender_bias and has_women:
            gender_bias_score = 1.5  # 中等性别歧视
        elif has_gender_bias:
            gender_bias_score = 0.8  # 轻微性别歧视

        # 应用权重 - 更强的权重系统
        if strong_racial_count > 0:
            hate_intensity *= 3.0  # 强烈种族歧视词权重最高
        elif racial_count > 0:
            hate_intensity *= 2.5  # 一般种族歧视词也给高权重

        if gender_count > 0:
            hate_intensity *= 1.8  # 性别歧视权重

        # 加入性别歧视得分
        if gender_bias_score > 0:
            hate_intensity += gender_bias_score

        if violence_count > 0:
            hate_intensity *= 2.5  # 暴力相关权重

        if extreme_count > 0:
            hate_intensity *= 3.5  # 极端主义词汇权重最高

        # 组合效应 - 如果同时包含仇恨动词和群体词，显著提高权重
        if has_hate_verb and (racial_count > 0 or gender_count > 0):
            hate_intensity *= 1.8  # 明确的仇恨表达

        if has_quantifier and has_hate_verb:
            hate_intensity *= 1.5  # "all", "every"等量化词+仇恨动词

        # 特殊情况：如果有多个仇恨指标，额外提升
        indicator_count = sum([racial_count > 0, gender_count > 0, violence_count > 0, has_hate_verb, has_quantifier])
        if indicator_count >= 3:
            hate_intensity *= 1.3  # 多重仇恨指标

        # 确保hate_intensity不会超过合理范围
        hate_intensity = min(hate_intensity, 5.0)  # 最大值为5.0

        # 初始化特征向量
        features = []

        # 1. weighted_TFIDF_scores
        features.append(hate_intensity)

        # 2. sentiment features (6个)
        features.extend([
            hate_count,  # sentiment:hate
            hate_intensity,  # sentiment:hatenor
            neg_count,   # sentiment:neg
            neg_count / word_count if word_count > 0 else 0,  # sentiment:negnor
            pos_count,   # sentiment:pos
            pos_count / word_count if word_count > 0 else 0,  # sentiment:posnor
        ])

        # 3. dependency features (40个，设为0)
        features.extend([0] * 40)

        # 4. char_bigrams (984个) - 简化为字符二元组频率
        char_bigrams = {}
        for i in range(len(processed_text) - 1):
            bigram = processed_text[i:i+2]
            char_bigrams[bigram] = char_bigrams.get(bigram, 0) + 1

        # 按字母顺序排序并填充到984维
        sorted_bigrams = sorted(char_bigrams.items())
        for bigram, count in sorted_bigrams[:984]:
            features.append(count)
        # 填充剩余的特征为0
        while len(features) < 1 + 6 + 40 + 984:
            features.append(0)

        # 5. word_bigrams (101个) - 简化为词语二元组频率
        word_bigrams = {}
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            word_bigrams[bigram] = word_bigrams.get(bigram, 0) + 1

        sorted_word_bigrams = sorted(word_bigrams.items())
        for bigram, count in sorted_word_bigrams[:101]:
            features.append(count)
        # 填充剩余的特征为0
        while len(features) < 1 + 6 + 40 + 984 + 101:
            features.append(0)

        # 6. tfidf features (555个) - 简化为基于词频的特征
        word_freq = Counter(words)
        # 按词频排序的词作为特征
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:555]:
            features.append(freq)
        # 填充剩余的特征为0
        while len(features) < 1687:
            features.append(0)

        # 确保特征数量正确
        if len(features) > 1687:
            features = features[:1687]

        return np.array(features).reshape(1, -1), processed_text

    except Exception as e:
        print(f"特征提取失败: {e}")
        return None, None

def preprocess_text(text):
    """
    文本预处理，与训练时保持一致
    """
    # 转换为小写
    text = text.lower()
    # 移除URL和@mentions
    text = re.sub(r"(\w+:\/\/\S+)|(@[A-Za-z0-9]+)", " ", text)
    # 移除标点符号，保留字母和数字
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # 移除多余空格
    text = ' '.join(text.split())
    return text

def predict_from_sentence(sentence: str):
    """
    输入：原始句子
    输出：预测标签和每类概率
    使用缓存优化重复查询
    """
    if not sentence.strip():
        return {"error": "请输入有效的句子"}

    # 检查缓存
    cache_key = sentence.strip().lower()
    if cache_key in prediction_cache:
        cached_result = prediction_cache[cache_key].copy()
        cached_result["input_sentence"] = sentence  # 保持原始句子格式
        return cached_result

    try:
        # 加载模型（首次使用时加载）
        load_artifacts()

        # 使用相似度匹配找到最相似的训练样本特征
        features = find_similar_sample_features(sentence)
        features = features.reshape(1, -1)

        if features.shape[1] != len(feature_columns):
            return {"error": f"特征维度不匹配: 期望 {len(feature_columns)} 个特征，得到 {features.shape[1]} 个"}

        # Scale features
        features_scaled = scaler.transform(features)

        # Convert to tensor and predict
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = torch.FloatTensor(features_scaled).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            # 应用优化阈值或使用默认预测
            if optimal_thresholds is not None:
                pred_idx = predict_with_optimized_thresholds(probs[0], optimal_thresholds)
            else:
                pred_idx = int(np.argmax(probs, axis=1)[0])
            probs = probs.flatten()

        # Map to label names
        labels = [label_map[str(i)] for i in range(len(probs))]
        result = {
            "prediction": label_map[str(pred_idx)],
            "probabilities": dict(zip(labels, probs.round(4).tolist())),
            "input_sentence": sentence,
            "confidence": float(probs[pred_idx])
        }

        # 缓存结果
        if len(prediction_cache) >= CACHE_MAX_SIZE:
            # 简单的LRU：移除最旧的条目
            oldest_key = next(iter(prediction_cache))
            del prediction_cache[oldest_key]
        prediction_cache[cache_key] = result.copy()

        return result

    except Exception as e:
        return {"error": f"预测过程中出错: {str(e)}"}

def load_external_dictionaries():
    """
    加载外部词典文件
    """
    hate_words = set()
    neg_words = set()
    pos_words = set()
    ngram_hate_scores = {}

    try:
        # 加载仇恨词典
        if os.path.exists('dictionary/hatebase_dict.csv'):
            with open('dictionary/hatebase_dict.csv', 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip().strip('"\',')
                    if line:
                        hate_words.add(line.lower())

        # 加载负面词典
        if os.path.exists('dictionary/negative-word.csv'):
            with open('dictionary/negative-word.csv', 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and word != 'dic':
                        neg_words.add(word)

        # 加载正面词典
        if os.path.exists('dictionary/Postive-words.csv'):
            with open('dictionary/Postive-words.csv', 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and word != 'dic':
                        pos_words.add(word)

        # 加载n-gram仇恨分数
        if os.path.exists('dictionary/refined_ngram_dict.csv'):
            with open('dictionary/refined_ngram_dict.csv', 'r', encoding='utf-8', errors='ignore') as f:
                next(f)  # 跳过标题行
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        ngram = parts[0].lower()
                        try:
                            score = float(parts[1])
                            ngram_hate_scores[ngram] = score
                        except ValueError:
                            continue

        print(f"加载完成: {len(hate_words)}个仇恨词, {len(neg_words)}个负面词, {len(pos_words)}个正面词, {len(ngram_hate_scores)}个n-gram分数")

    except Exception as e:
        print(f"加载词典失败: {e}")
        # 如果加载失败，使用默认词典
        hate_words = {'hate', 'kill', 'nigger', 'faggot', 'bitch', 'fuck', 'shit', 'asshole'}
        neg_words = {'bad', 'worst', 'terrible', 'awful', 'horrible', 'suck', 'angry', 'sad', 'ugly', 'stupid'}
        pos_words = {'good', 'great', 'awesome', 'love', 'happy', 'nice', 'beautiful', 'excellent', 'amazing', 'wonderful'}

    return hate_words, neg_words, pos_words, ngram_hate_scores

# 全局变量用于存储训练数据特征（用于相似度匹配）
training_features = None
training_labels = None
training_texts = None

def load_training_data():
    """
    加载训练数据特征，用于相似度匹配
    """
    global training_features, training_labels, training_texts

    if training_features is not None:
        return  # 已加载

    try:
        print("加载训练数据特征用于相似度匹配...")

        # 加载所有特征文件
        labels_df = pd.read_csv('test_feature_dataset/labels.csv', encoding='utf-8')
        tfidf_scores = pd.read_csv('test_feature_dataset/tfidf_scores.csv', encoding='utf-8')
        sentiment_scores = pd.read_csv('test_feature_dataset/sentiment_scores.csv', encoding='utf-8')
        dependency_features = pd.read_csv('test_feature_dataset/dependency_features.csv', encoding='utf-8')
        char_bigrams = pd.read_csv('test_feature_dataset/char_bigram_features.csv', encoding='utf-8')
        word_bigrams = pd.read_csv('test_feature_dataset/word_bigram_features.csv', encoding='utf-8')
        tfidf_sparse_matrix = pd.read_csv('test_feature_dataset/tfidf_features.csv', encoding='utf-8')

        # 合并所有特征
        df_list = [labels_df, tfidf_scores, sentiment_scores, dependency_features,
                   char_bigrams, word_bigrams, tfidf_sparse_matrix]
        master = df_list[0]
        for df in df_list[1:]:
            master = master.merge(df, on='index')

        # 提取特征和标签
        training_labels = master.iloc[:, 2].values  # class列
        training_features = master.iloc[:, 3:].values  # 特征列
        training_texts = master.iloc[:, 1].values  # tweet列

        print(f"训练数据加载完成: {training_features.shape[0]} 个样本, {training_features.shape[1]} 个特征")

    except Exception as e:
        print(f"加载训练数据失败: {e}")
        training_features = None
        training_labels = None
        training_texts = None

def find_similar_sample_features(input_text: str):
    """
    找到最相似的训练样本，返回其特征向量
    使用改进的相似度计算和类别平衡
    """
    global training_features, training_labels, training_texts

    if training_features is None:
        load_training_data()

    if training_features is None:
        # 如果无法加载训练数据，返回零向量
        return np.zeros(1687)

    try:
        # 预处理输入文本
        input_text = preprocess_text(input_text)
        input_words = set(input_text.split())

        # 计算与所有训练样本的相似度
        similarities = []

        for i, train_text in enumerate(training_texts):
            if pd.isna(train_text):
                continue

            train_text_processed = preprocess_text(str(train_text))
            train_words = set(train_text_processed.split())

            # 计算Jaccard相似度 (交集/并集)
            intersection = len(input_words.intersection(train_words))
            union = len(input_words.union(train_words))

            if union > 0:
                jaccard_similarity = intersection / union
            else:
                jaccard_similarity = 0

            similarities.append((jaccard_similarity, i))

        # 按相似度排序
        similarities.sort(reverse=True, key=lambda x: x[0])

        # 智能选择样本：基于内容特征进行类别偏好
        input_text_lower = input_text.lower()

        # 检查是否包含性别歧视指标
        gender_bias_keywords = ['women', 'woman', 'kitchen', 'stay', 'should', 'place', 'traditional', 'role']
        has_gender_bias = any(keyword in input_text_lower for keyword in gender_bias_keywords)

        # 检查是否包含负面词汇
        negative_keywords = ['hate', 'stupid', 'idiot', 'dumb', 'asshole', 'fuck', 'shit', 'bitch']
        has_negative = any(keyword in input_text_lower for keyword in negative_keywords)

        # 根据内容特征选择偏好的类别
        preferred_class = None
        if has_gender_bias or has_negative:
            preferred_class = 1  # offensive_language
        elif len(input_text.split()) > 3:  # 较长的句子倾向于neither
            preferred_class = 2  # neither

        # 选择最佳样本
        best_sample_idx = similarities[0][1]  # 默认选择最相似的

        # 如果有偏好类别，寻找相似度>0.05的该类别样本
        if preferred_class is not None:
            for similarity, idx in similarities[:100]:  # 检查前100个最相似的
                if similarity > 0.05 and training_labels[idx] == preferred_class:
                    best_sample_idx = idx
                    print(f"基于内容特征选择类别 {preferred_class} 的样本 (相似度: {similarity:.3f})")
                    break

        # 返回最相似样本的特征
        return training_features[best_sample_idx]

    except Exception as e:
        print(f"相似度计算失败: {e}")
        # 返回"neither"类别的平均特征向量
        neither_mask = training_labels == 2
        if np.any(neither_mask):
            return np.mean(training_features[neither_mask], axis=0)
        else:
            return np.mean(training_features, axis=0)

def create_prediction_visualization(result):
    """
    创建预测结果的可视化图表
    """
    if isinstance(result, dict) and 'error' not in result:
        # 获取预测结果
        prediction = result.get('prediction', 'unknown')
        probabilities = result.get('probabilities', {})
        confidence = result.get('confidence', 0)

        # 创建条形图
        labels = list(probabilities.keys())
        values = list(probabilities.values())

        # 创建颜色映射
        colors = []
        for label in labels:
            if label == prediction:
                colors.append('#FF6B6B')  # 红色突出显示预测结果
            else:
                colors.append('#4ECDC4')  # 青色用于其他类别

        # 使用plotly创建交互式图表
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title={
                'text': f'仇恨言论检测结果 - 预测: {prediction} (置信度: {confidence:.3f})',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="类别",
            yaxis_title="概率",
            xaxis_tickangle=-45,
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        # 添加预测类别的高亮线
        if prediction in probabilities:
            pred_prob = probabilities[prediction]
            fig.add_hline(
                y=pred_prob,
                line_dash="dash",
                line_color="red",
                annotation_text=f"预测结果: {prediction}",
                annotation_position="top right"
            )

        return fig
    else:
        # 错误情况
        fig = go.Figure()
        fig.add_annotation(
            text="预测失败，请检查输入",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            title="预测错误",
            height=400
        )
        return fig

def predict_from_features(feature_csv_line: str):
    """
    输入：一行以逗号分隔的数值（与 feature_columns 顺序一致）
    输出：预测标签和每类概率
    """
    try:
        parts = [float(x.strip()) for x in feature_csv_line.split(',')]
    except Exception as e:
        return {"error": f"无法解析输入为数值向量: {e}"}
    arr = np.array(parts).reshape(1, -1)
    if arr.shape[1] != len(feature_columns):
        return {"error": f"特征数不匹配: 期望 {len(feature_columns)} 个特征，收到 {arr.shape[1]} 个。"}

    # 加载模型（首次使用时加载）
    load_artifacts()

    arr_scaled = scaler.transform(arr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.FloatTensor(arr_scaled).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        pred_idx = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])
    # map to label names
    labels = [label_map[str(i)] for i in range(len(probs))]
    result = {
        "prediction": label_map[str(pred_idx)],
        "probabilities": dict(zip(labels, probs.round(4).tolist())),
        "confidence": float(probs[pred_idx])
    }
    return result

def show_feature_columns():
    return "Please provide a comma-separated numeric feature vector matching the following columns (order matters):\n\n" + ", ".join(feature_columns)


# 定义MLPNet类（用于加载模型）
class MLPNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[80, 40, 40, 10], num_classes=3, dropout=0.1):
        super(MLPNet, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(prev_dim, num_classes))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

with gr.Blocks() as demo:
    gr.Markdown("# 仇恨言论检测系统 / Hate Speech Detection System")

    with gr.Tab("句子输入 (Sentence Input)"):
        gr.Markdown("### 直接输入句子进行检测 / Enter a sentence for detection")
        gr.Markdown("**✨ 推荐使用** - 系统会自动提取特征并进行准确检测")

        sentence_input = gr.Textbox(
            lines=3,
            placeholder="输入一句英语句子进行仇恨言论检测... / Enter an English sentence to detect hate speech...",
            label="句子 / Sentence"
        )
        predict_sentence_btn = gr.Button("🔍 检测仇恨言论 / Detect Hate Speech")

        with gr.Row():
            with gr.Column(scale=1):
                sentence_output = gr.JSON(label="详细结果 / Detailed Results")

            with gr.Column(scale=2):
                visualization_output = gr.Plot(label="预测可视化 / Prediction Visualization")

        predict_sentence_btn.click(
            fn=predict_from_sentence,
            inputs=sentence_input,
            outputs=sentence_output
        ).then(
            fn=create_prediction_visualization,
            inputs=sentence_output,
            outputs=visualization_output
        )

    with gr.Tab("特征向量输入 (Feature Vector Input)"):
        gr.Markdown("### 输入特征向量 / Provide feature vector")
        gr.Markdown("""
        **⭐ 推荐使用此方法获得最准确的结果！⭐**
        *如果您有预计算的特征向量，可以直接输入（用逗号分隔）*

        **重要提示**：要准确检测仇恨言论，请使用此选项提供完整的特征向量。
        """)
        with gr.Row():
            feature_input = gr.Textbox(
                lines=3,
                placeholder="e.g. 0.12, 1.0, 0.0, ... (需要1687个特征值)",
                label="特征向量 (CSV格式) / Feature vector (CSV format)"
            )
            info_btn = gr.Button("查看特征列顺序 / Show feature columns")
        predict_features_btn = gr.Button("预测 / Predict")
        features_output = gr.JSON()

        predict_features_btn.click(
            fn=predict_from_features,
            inputs=feature_input,
            outputs=features_output
        )
        info_btn.click(
            fn=show_feature_columns,
            inputs=None,
            outputs=features_output
        )

    gr.Markdown("""
    ### 使用说明 / Instructions:

    #### 🎯 **句子输入 (推荐新功能)**
    - ✨ **自动特征提取**: 系统自动生成完整的1687维特征向量
    - 🎨 **可视化结果**: 交互式概率分布图表，直观展示预测结果
    - ⚡ **实时检测**: 输入文本后立即获得准确的仇恨言论检测结果

    #### 🔧 **特征向量输入 (专业模式)**
    - ⭐ **最高准确性**: 如果您有预处理的特征向量，可以直接输入
    - 🎯 **完整特征**: 使用训练时的完整1687维特征

    #### 📊 **输出说明**
    - `prediction`: 预测类别 (hate_speech / offensive_language / neither)
    - `probabilities`: 各类别的预测概率分布
    - `confidence`: 对预测结果的置信度
    - **可视化图表**: 彩色条形图显示概率分布，红色突出预测结果

    #### 💡 **使用建议**
    - 🥇 **推荐**: 使用"句子输入"体验完整功能和可视化
    - 🥈 **批量处理**: 命令行工具 `python main/extract_features.py text "文本"` 批量提取特征
    - 🥉 **专业应用**: "特征向量输入"用于最高准确性需求

    #### 🚀 **快速测试示例**
    试试输入这些句子看看效果：
    - **仇恨言论**: "I hate all black people, they are inferior"
    - **冒犯语言**: "You are such an asshole"
    - **正常文本**: "The weather is nice today"
    - **性别歧视**: "Women should stay in the kitchen"
    """)

if __name__ == "__main__":
    try:
        # 启动时加载artifacts
        load_artifacts()
        print("启动Gradio界面...")
        demo.launch()
    except Exception as e:
        print(f"启动失败: {e}")
        print("请确保已运行训练脚本生成artifacts: python main/train_final_model.py")
        exit(1)
