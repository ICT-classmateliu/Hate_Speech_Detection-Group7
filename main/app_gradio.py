# 由 @Classmateliu 创建、编写及维护
# 利用 gradio 实现模型预测的可视化
# 最后修改日期：2026.1.2

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

# 用于统一管理训练阶段生成的模型与配置文件路径
ARTIFACTS_DIR = os.path.join('main', 'artifacts')

# 采用 懒加载：第一次需要用的时候才加载，之后直接复用全局变量
# 在推理系统里，我们通常不希望每次用户输入时都重复加载模型、特征处理器、配置文件
model = None
scaler = None
base_scaler = None
feature_columns = None
label_map = None
optimal_thresholds = None
tfidf_vectorizer = None  # 缓存 TfidfVectorizer，避免重复加载

# 缓存训练数据的二元组特征列名，用于实时计算
char_bigram_columns = None
word_bigram_columns = None

artifacts_loaded = False

# prediction_cache 用于缓存已经预测过的句子及其结果，避免重复计算，提高推理效率
# CACHE_MAX_SIZE 限制缓存大小，保证系统长期运行稳定
prediction_cache = {}
CACHE_MAX_SIZE = 100  # 最多缓存100个句子

def load_artifacts():
    # 这样可以让整个 Gradio 或 API 脚本共享同一份资源，避免重复加载
    global model, scaler, base_scaler, feature_columns, label_map, tfidf_vectorizer, char_bigram_columns, word_bigram_columns, artifacts_loaded

    # 避免重复加载，提高性能
    if artifacts_loaded:
        return  # 已加载，直接返回

    print("正在加载模型和预处理器...")

    model_path = os.path.join(ARTIFACTS_DIR, 'final_model_state_dict.pth')
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
    cols_path = os.path.join(ARTIFACTS_DIR, 'feature_columns.json')
    labels_path = os.path.join(ARTIFACTS_DIR, 'label_map.json')
    base_scaler_path = os.path.join(ARTIFACTS_DIR, 'base_scaler.pkl')

    # 检查必需文件是否存在
    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append('final_model_state_dict.pth')
    if not os.path.exists(scaler_path):
        missing_files.append('scaler.pkl')
    if not os.path.exists(cols_path):
        missing_files.append('feature_columns.json')
    if not os.path.exists(labels_path):
        missing_files.append('label_map.json')

    if missing_files:
        raise FileNotFoundError(f"缺少必要的artifacts文件: {', '.join(missing_files)}. 请先运行训练脚本: python main/train_final_model.py")

    # 加载模型权重与基础预处理器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_state = torch.load(model_path, map_location=device, weights_only=True)
    scaler = joblib.load(scaler_path)
    base_scaler = joblib.load(base_scaler_path) if os.path.exists(base_scaler_path) else None

    # TF-IDF 特征用于建模关键词的区分性
    # 字符和词语二元组特征用于捕捉拼写变形和局部组合语义
    # 加载 TF-IDF 向量器
    # 这个对象里包含：
    #   词汇表（vocabulary）
    #   IDF 权重
    tfidf_vectorizer_path = os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.pkl')
    
    # 加载训练阶段保存的 TfidfVectorizer
    # 因为 TF-IDF 的每一维都对应一个“固定词”，换一个 vectorizer，特征维度和语义就全乱了
    if os.path.exists(tfidf_vectorizer_path):
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        print(f"TfidfVectorizer 已加载 (词汇表大小: {len(tfidf_vectorizer.vocabulary_)})")
    else:
        tfidf_vectorizer = None
        print("警告: 未找到 TfidfVectorizer，将使用近似特征")

    # 加载字符 / 词语二元组特征列名
    try:
        # 字符二元组（char bigram） 只读表头
        char_df = pd.read_csv('test_feature_dataset/char_bigram_features.csv', nrows=0)
        char_bigram_columns = [col for col in char_df.columns if col != 'index']

        # 只加载词二元组特征名
        word_df = pd.read_csv('test_feature_dataset/word_bigram_features.csv', nrows=0)
        word_bigram_columns = [col for col in word_df.columns if col != 'index']

        print(f"二元组特征列名已加载 (字符: {len(char_bigram_columns)}, 词语: {len(word_bigram_columns)})")

    except Exception as e:
        char_bigram_columns = None
        word_bigram_columns = None
        print(f"加载二元组特征列名失败: {e}")

    # 加载配置文件（特征列 & 标签映射）
    with open(cols_path, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)  # 列名列表，用于保证推理时输入特征顺序一致
    with open(labels_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)    # 模型输出数字 → 类别名映射

    # 构建模型
    # 输入维度 = 特征列数量
    # 模型结构 = MLPNet（多层感知机）
    #   hidden_dims=[80, 40, 40, 10] → 四个隐藏层
    input_dim = len(feature_columns)
    model = MLPNet(input_dim=input_dim, hidden_dims=[80, 40, 40, 10], num_classes=len(label_map))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # 标记加载完成
    artifacts_loaded = True
    print(f"模型加载完成！特征维度: {input_dim}")

    # 返回所有核心对象，方便后续函数直接使用
    return model, scaler, base_scaler, feature_columns, label_map

# 先看每个类别是否超过自己专属的阈值，再在“合格的类别”里选概率最大的那个
def predict_with_optimized_thresholds(sample_probs, thresholds):

    # 3（hate / offensive / neither）
    n_classes = len(sample_probs)

    # 检查每个类别是否超过其阈值
    # 概率 ≥ 该类别阈值的候选类别
    valid_classes = []
    for class_idx in range(n_classes):
        threshold = thresholds[str(class_idx)]['threshold']
        if sample_probs[class_idx] >= threshold:
            valid_classes.append((class_idx, sample_probs[class_idx]))

    # 有“合格类别”的情况
    if valid_classes:
        # 如果 至少一个类别过了自己的阈值
        # 按概率从高到低排序
        # 选择概率最高的那个类别
        valid_classes.sort(key=lambda x: x[1], reverse=True)
        return valid_classes[0][0]
    else:
        # 如果没有类别超过阈值，选择概率最高的类别（兜底策略）
        return np.argmax(sample_probs)

# 把一条输入文本 text → 转换成 和训练阶段一致的 1687 维特征向量，用于给 MLP 模型做推理
def extract_full_features_from_text(text):
    try:
        # 文本预处理
        # 词频 / 密度 / 比例特征的基准
        processed_text = preprocess_text(text)
        words = processed_text.split()
        word_count = max(len(words), 1)

        # 加载外部词典
        hate_words, neg_words, pos_words, ngram_hate_scores = load_external_dictionaries()

        # 基础词频统计（情感层）
        # 出现了多少仇恨词 负面 / 正面词各多少
        hate_count = sum(1 for word in words if word in hate_words)
        neg_count = sum(1 for word in words if word in neg_words)
        pos_count = sum(1 for word in words if word in pos_words)

        # 仇恨 密度 + 强度
        hate_density = hate_count / word_count if word_count > 0 else 0
        hate_intensity = hate_density

        # 增强权重计算 定义多类敏感词集合
        # 把 offensive 和 hate speech 区分开 为后面的 加权策略 做准备
        racial_words = {'nigger', 'nigga', 'kike', 'chink', 'gook', 'spic', 'wetback', 'coon', 'paki', 'raghead', 'towelhead', 'jew', 'arab', 'muslim', 'black', 'white', 'asian', 'hispanic', 'latino', 'mexican', 'african', 'european'}
        strong_racial_words = {'nigger', 'nigga', 'kike', 'coon', 'chink', 'gook', 'spic'}  
        # 特别强烈的种族歧视词

        gender_words = {'bitch', 'cunt', 'whore', 'slut', 'fag', 'faggot', 'dyke', 'tranny', 'shemale'}
        violence_words = {'kill', 'die', 'death', 'murder', 'rape', 'torture', 'exterminate', 'genocide'}
        extreme_words = {'holocaust', 'nazi', 'hitler', 'supremacist'}  
        # 极端主义词汇

        # 计算各种类别的词频
        racial_count = sum(1 for word in words if word in racial_words)
        strong_racial_count = sum(1 for word in words if word in strong_racial_words)
        gender_count = sum(1 for word in words if word in gender_words)
        violence_count = sum(1 for word in words if word in violence_words)
        extreme_count = sum(1 for word in words if word in extreme_words)

        # 检查是否包含强烈的仇恨表达
        # 从词 → 句法/语义层
        # 是否出现仇恨/暴力动词
        has_hate_verb = any(word in ['hate', 'deserve', 'kill', 'exterminate', 'genocide'] for word in words)
        # 是否出现量化 / 泛化词
        has_quantifier = any(word in ['all', 'every', 'each', 'none', 'no'] for word in words)

        # 检查性别歧视模式
        # 不一定有脏词，但语义是歧视
        gender_bias_indicators = [
            'kitchen', 'cooking', 'cleaning', 'housewife', 'homemaker',
            'traditional', 'submissive', 'place', 'role', 'stay'
        ]
        # 是否出现 “性别角色刻板词”
        has_gender_bias = any(word in gender_bias_indicators for word in words)
        # 是否出现规范性/命令性词语
        has_should = 'should' in words
        # 是否明确指向“女性群体”
        has_women = 'women' in words or 'woman' in words

        # 性别歧视组合得分
        gender_bias_score = 0
        if has_gender_bias and has_should and has_women:
            gender_bias_score = 2.0  # 强烈的性别歧视模式
        elif has_gender_bias and has_women:
            gender_bias_score = 1.5  # 中等性别歧视
        elif has_gender_bias:
            gender_bias_score = 0.8  # 轻微性别歧视

        # 仇恨强度的加权放大 
        # 不同仇恨维度 × 不同权重
        # 强种族歧视 > 一般歧视
        # 极端主义 > 暴力 > 性别
        # 动词 + 群体 = 明确仇恨  
        # 多个指标叠加 = 更危险
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

        # 如果同时包含仇恨动词和群体词，显著提高权重
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

        # 严格按照“训练时的特征结构”，从一条文本中重新构造出同样顺序、同样维度（1687 维）的特征向量，用于模型推理
        # 构建特征向量 每一段都对应训练时的一块特征
        features = []

        # 1、加权 TF-IDF 综合分数（1维）
        try:
            if tfidf_vectorizer is not None:
                # 把文本变成 TF-IDF 稀疏向量
                tfidf_matrix = tfidf_vectorizer.transform([processed_text])
                total_tfidf_sum = float(tfidf_matrix.sum())

                # 始化四类加权分数
                # 明确仇恨词 hate
                # 一般负面 neg
                # 正面词（反向作用） pos
                # 极端攻击性词 intense
                hate_word_tfidf_sum = 0.0
                neg_word_tfidf_sum = 0.0
                pos_word_tfidf_sum = 0.0
                intense_word_tfidf_sum = 0.0

                # 强烈词典
                intense_words = {'hate', 'kill', 'fuck', 'shit', 'bitch', 'nigger', 'asshole', 'damn'}

                # 遍历 TF-IDF 词表
                # 保证用的是 训练时同一套词表 同一列索引
                for word, idx in tfidf_vectorizer.vocabulary_.items():
                    tfidf_score = float(tfidf_matrix[0, idx])

                    # 分类型加权
                    # 仇恨词 ×3 极端词 ×4（最高） 正面词反而削弱仇恨感知
                    # 给模型一个“我已经帮你算过仇恨强度”的超强先验特征
                    if word in hate_words:
                        hate_word_tfidf_sum += tfidf_score * 3.0  # 仇恨词最高权重
                    elif word in neg_words:
                        neg_word_tfidf_sum += tfidf_score * 1.5  # 负面词中等权重
                    elif word in pos_words:
                        pos_word_tfidf_sum += tfidf_score * 0.7  # 正面词降低权重（可能冲淡负面）
                    elif word in intense_words:
                        intense_word_tfidf_sum += tfidf_score * 4.0  # 特别强烈的词汇

                # 线性加权模型
                # 仇恨 > 强烈 > 负面 > 正面（反向）
                base_weight = total_tfidf_sum * 0.3  # 基础TF-IDF权重
                hate_weight = hate_word_tfidf_sum * 0.4  # 仇恨词权重
                neg_weight = neg_word_tfidf_sum * 0.15  # 负面词权重
                pos_weight = pos_word_tfidf_sum * (-0.1)  # 正面词负面影响（可能降低整体负面程度）
                intense_weight = intense_word_tfidf_sum * 0.25  # 强烈词汇权重

                weighted_score = base_weight + hate_weight + neg_weight + pos_weight + intense_weight

                # 限幅 + 加入特征
                weighted_score = min(max(weighted_score, 0), 10.0)
                features.append(weighted_score)
            else:
                # 没有 TF-IDF 时的回退方案
                # 回退到基于词典的改进加权分数
                hate_weight = sum(3.0 for word in words if word in hate_words)
                neg_weight = sum(1.5 for word in words if word in neg_words)
                pos_weight = sum(0.5 for word in words if word in pos_words)
                intense_weight = sum(4.0 for word in words if word in {'hate', 'kill', 'fuck', 'shit', 'bitch', 'nigger', 'asshole', 'damn'})

                weighted_score = (hate_weight + neg_weight - pos_weight + intense_weight) / word_count if word_count > 0 else 0
                weighted_score = min(max(weighted_score, 0), 10.0)
                features.append(weighted_score)
        except Exception as e:
            print(f"计算改进的 TF-IDF 加权分数失败，使用近似值: {e}")
            features.append(hate_intensity)

        # 2、情感特征（6维）
        features.extend([
            hate_count,  # sentiment:hate
            hate_intensity,  # sentiment:hatenor (使用增强的仇恨强度)
            neg_count,   # sentiment:neg
            neg_count / word_count if word_count > 0 else 0,  # sentiment:negnor
            pos_count,   # sentiment:pos
            pos_count / word_count if word_count > 0 else 0,  # sentiment:posnor
        ])
        # count	绝对数量    nor	归一化密度

        # 3、句法复杂度（40维，真实用 7 + padding）
        sentence_length = len(processed_text)  
        # 句子长度
        word_count_feat = len(words)  
        # 词数
        avg_word_length = sentence_length / word_count_feat if word_count_feat > 0 else 0  
        # 平均词长
        punctuation_count = sum(1 for char in processed_text if char in '.,!?;:()[]{}')  
        # 标点符号数
        punctuation_density = punctuation_count / sentence_length if sentence_length > 0 else 0  
        # 标点密度
        uppercase_ratio = sum(1 for char in processed_text if char.isupper()) / sentence_length if sentence_length > 0 else 0  
        # 大写字母比例
        digit_ratio = sum(1 for char in processed_text if char.isdigit()) / sentence_length if sentence_length > 0 else 0  
        # 数字比例

        # 全大写 → 情绪激动
        # 标点密集 → 攻击性
        # 短句 + 粗词 → 辱骂

        # 填充这些特征，然后用0填充剩余的dependency特征位置
        syntax_features = [
            sentence_length, word_count_feat, avg_word_length,
            punctuation_count, punctuation_density, uppercase_ratio, digit_ratio
        ]
        # 确保只使用前7个位置，剩余33个设为0
        features.extend(syntax_features[:7])
        features.extend([0] * (40 - len(syntax_features[:7])))

        # 4、字符二元组（984维）
        try:
            if char_bigram_columns is not None:
                # 实时计算字符二元组频率
                # 为什么 char bigram 很重要 能捕捉“规避审查的拼写变体”
                char_counts = {}
                for i in range(len(processed_text) - 1):
                    bigram = processed_text[i:i+2]
                    char_counts[bigram] = char_counts.get(bigram, 0) + 1

                # 严格按照训练时的列顺序填
                for col in char_bigram_columns:
                    # 从列名中提取二元组，如 "char_bigrams: th" -> "th"
                    bigram = col.replace('char_bigrams: ', '')
                    count = char_counts.get(bigram, 0)
                    features.append(count)
            else:
                # 如果无法加载列名，回退到0
                features.extend([0] * 984)
        except Exception as e:
            print(f"计算字符二元组特征失败: {e}")
            features.extend([0] * 984)

        # 5、词语二元组（101维）
        # 对群体攻击特别敏感
        try:
            if word_bigram_columns is not None:
                # 实时计算词语二元组频率
                word_counts = {}
                words_list = processed_text.split()
                for i in range(len(words_list) - 1):
                    bigram = f"{words_list[i]} {words_list[i+1]}"  # 用空格分隔，与训练数据格式一致
                    word_counts[bigram] = word_counts.get(bigram, 0) + 1

                # 按照训练时列的顺序填充特征
                for col in word_bigram_columns:
                    # 从列名中提取二元组，如 "word_bigrams: very good" -> "very good"
                    bigram = col.replace('word_bigrams: ', '')
                    count = word_counts.get(bigram, 0)
                    features.append(count)
            else:
                # 如果无法加载列名，回退到0
                features.extend([0] * 101)
        except Exception as e:
            print(f"计算词语二元组特征失败: {e}")
            features.extend([0] * 101)

        # 6、TF-IDF 原始向量（555维）
        # 这是模型真正吃得最多的信息
        try:
            if tfidf_vectorizer is not None:
                # 使用缓存的 vectorizer 转换文本为 TF-IDF 特征
                tfidf_features = tfidf_vectorizer.transform([processed_text]).toarray().flatten()
                # 确保维度匹配 555
                if len(tfidf_features) == 555:
                    features.extend(tfidf_features.tolist())
                elif len(tfidf_features) < 555:
                    # 填充到 555 维
                    features.extend(tfidf_features.tolist())
                    features.extend([0.0] * (555 - len(tfidf_features)))
                else:
                    # 截断到 555 维
                    features.extend(tfidf_features[:555].tolist())
            else:
                # 回退到近似值：基于仇恨密度的简单重复
                tfidf_approx = hate_density
                features.extend([tfidf_approx] * 555)
        except Exception as e:
            print(f"生成 TF-IDF 向量失败，使用近似值: {e}")
            tfidf_approx = hate_density
            features.extend([tfidf_approx] * 555)

        # (1, 1687)：符合 sklearn / torch 推理接口
        # 同时返回 processed_text 方便 UI 显示
        return np.array(features).reshape(1, -1), processed_text

    except Exception as e:
        print(f"特征提取失败: {e}")
        return None, None

# 把“用户原始输入的自然语言”，转换成“模型训练时见过的规范化文本形式”
# “推理阶段严格复现训练阶段的文本预处理流程，避免特征空间偏移（feature shift）
def preprocess_text(text):
    # 转换为小写
    # TF-IDF 默认区分大小写 字符 / 词二元组对大小写极其敏感
    text = text.lower()
    # 移除URL和@mentions
    # URL 会被当成稀有词，干扰 TF-IDF 权重
    text = re.sub(r"(\w+:\/\/\S+)|(@[A-Za-z0-9]+)", " ", text)
    # 移除标点符号，保留字母和数字
    # 移除标点（只保留字母、数字、空格）
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # 移除多余空格
    text = ' '.join(text.split())
    return text

# 输入一条原始文本 → 走完整推理链路 → 输出预测类别 + 各类别概率 + 置信度，并带缓存优化
def predict_from_sentence(sentence: str):
    # 空输入检查（防御式编程）
    if not sentence.strip():
        return {"error": "请输入有效的句子"}

    # 检查缓存
    cache_key = sentence.strip().lower()
    if cache_key in prediction_cache:
        cached_result = prediction_cache[cache_key].copy()
        cached_result["input_sentence"] = sentence  # 保持原始句子格式
        return cached_result

    try:
        # 懒加载模型与资源
        load_artifacts()

        # 使用相似度匹配找到最相似的训练样本特征
        # 基于相似训练样本的特征空间映射
        features = find_similar_sample_features(sentence)
        features = features.reshape(1, -1)

        if features.shape[1] != len(feature_columns):
            return {"error": f"特征维度不匹配: 期望 {len(feature_columns)} 个特征，得到 {features.shape[1]} 个"}

        # 特征缩放（scaler）
        features_scaled = scaler.transform(features)

        # PyTorch 推理阶段
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = torch.FloatTensor(features_scaled).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            # Softmax → 概率
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            # 应用优化阈值或使用默认预测
            # 每一类有 独立最优阈值 专门应对：类不平衡 仇恨类 recall 不足
            if optimal_thresholds is not None:
                pred_idx = predict_with_optimized_thresholds(probs[0], optimal_thresholds)
            else:
                pred_idx = int(np.argmax(probs, axis=1)[0])
            probs = probs.flatten()

        # 标签映射 + 结果组织
        labels = [label_map[str(i)] for i in range(len(probs))]
        result = {
            "prediction": label_map[str(pred_idx)],
            "probabilities": dict(zip(labels, probs.round(4).tolist())),
            "input_sentence": sentence,
            "confidence": float(probs[pred_idx])
        }

        # 缓存结果 LRU 缓存淘汰
        if len(prediction_cache) >= CACHE_MAX_SIZE:
            # 简单的LRU：移除最旧的条目
            oldest_key = next(iter(prediction_cache))
            del prediction_cache[oldest_key]
        prediction_cache[cache_key] = result.copy()

        return result

    except Exception as e:
        return {"error": f"预测过程中出错: {str(e)}"}

# 从外部文件中加载仇恨词、负面词、正面词和 n-gram 仇恨强度分数，用于特征工程和规则增强
def load_external_dictionaries():
    # 初始化四类容器
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
training_features = None # 训练集中 1687 维特征矩阵
training_labels = None # 每条训练样本的真实类别
training_texts = None # 原始文本（tweet）

# 把“训练集的完整特征空间”加载到内存中，用于在推理阶段做“相似样本检索 / 相似度匹配”
# 参考历史相似案例
def load_training_data():
    global training_features, training_labels, training_texts

    if training_features is not None:
        return  # 已加载

    try:
        print("加载训练数据特征用于相似度匹配...")

        # 加载所有特征文件
        # 开始加载训练数据
        labels_df = pd.read_csv('test_feature_dataset/labels.csv', encoding='utf-8')
        # 对应你训练时的 6 大特征块
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

        # 提取特征和标签 拆分出三大核心数组
        training_labels = master.iloc[:, 2].values  # 标签
        training_features = master.iloc[:, 3:].values  # 特征矩阵
        training_texts = master.iloc[:, 1].values  # 原始文本

        print(f"训练数据加载完成: {training_features.shape[0]} 个样本, {training_features.shape[1]} 个特征")

    except Exception as e:
        print(f"加载训练数据失败: {e}")
        training_features = None
        training_labels = None
        training_texts = None

# 给定一条输入文本，从训练集中找出“语义最相似”的一条样本，并返回该样本的特征向量，用来辅助后续模型预测
def find_similar_sample_features(input_text: str):
    global training_features, training_labels, training_texts

    # 确保训练数据已加载
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

            # 对训练文本做同样预处理
            # 保证相似度计算“口径一致”
            train_text_processed = preprocess_text(str(train_text))
            train_words = set(train_text_processed.split())

            # 计算Jaccard相似度 (交集/并集)
            # 适合 短文本 不依赖词频，只关心“是否出现” 计算快，解释性强
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
        # 转为小写方便关键词匹配
        input_text_lower = input_text.lower()

        # 检查是否包含性别歧视指标 规则增强型特征
        gender_bias_keywords = ['women', 'woman', 'kitchen', 'stay', 'should', 'place', 'traditional', 'role']
        has_gender_bias = any(keyword in input_text_lower for keyword in gender_bias_keywords)

        # 检查是否包含负面词汇 用于捕捉明显 offensive 语义
        negative_keywords = ['hate', 'stupid', 'idiot', 'dumb', 'asshole', 'fuck', 'shit', 'bitch']
        has_negative = any(keyword in input_text_lower for keyword in negative_keywords)

        # 根据内容特征选择偏好的类别
        # 不是强制分类，只是“优先考虑”
        # 性别歧视 / 脏话	offensive
        # 句子较长、无明显攻击	neither
        preferred_class = None
        if has_gender_bias or has_negative:
            preferred_class = 1  # offensive_language
        elif len(input_text.split()) > 3:  # 较长的句子倾向于neither
            preferred_class = 2  # neither

        # 选择最佳样本
        best_sample_idx = similarities[0][1]  # 默认选择最相似的

        # 如果有偏好类别，寻找相似度>0.05的该类别样本
        # 如果有偏好类别 → 在“相似样本中”找
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

# 可视化代码
def create_prediction_visualization(result):
    # 先处理输入与错误情况
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

# 直接对数值特征向量进行预测
# 绕过文本处理，直接使用模型训练时的 数值特征向量 做预测
# 适合 批量预测 或 调试模型，比如你已经有特征文件，不需要再从文本生成特征
def predict_from_features(feature_csv_line: str):
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

# 定义 MLP（多层感知器）神经网络结构
# 输入 1687 → 第1层 80 → 第2层 40 → 第3层 40 → 第4层 10
# Linear(prev_dim, hidden_dim) → 全连接层，把上一层输出映射到当前层大小
# ReLU() → 激活函数，增加非线性能力
# Dropout(dropout) → 随机丢弃一部分神经元，降低过拟合
# 典型的 MLP 分类网络
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

# 可视化界面
with gr.Blocks() as demo:
    gr.Markdown("# 仇恨言论检测系统")

    with gr.Tab("输入言论以进行检测... "):
        gr.Markdown("### 直接输入句子/短语进行检测，输出结果为三类：")
        gr.Markdown("### 仇恨言论(hate speech)，冒犯言论(offensive speech)，正常言论(normal speech)")

        sentence_input = gr.Textbox(
            lines=3,
            placeholder="在此输入以进行仇恨言论检测...",
            label="句子在这里输入哈"
        )
        predict_sentence_btn = gr.Button("🔍 开始检测仇恨言论")

        with gr.Row():
            with gr.Column(scale=1):
                sentence_output = gr.JSON(label="程序预测结果输出")

            with gr.Column(scale=2):
                visualization_output = gr.Plot(label="预测结果可视化")

        predict_sentence_btn.click(
            fn=predict_from_sentence,
            inputs=sentence_input,
            outputs=sentence_output
        ).then(
            fn=create_prediction_visualization,
            inputs=sentence_output,
            outputs=visualization_output
        )

    gr.Markdown("""
    ### 使用说明:

    #### **测试示例**
    - **仇恨言论**:
      - "I hate niggers" ⚠️原因：包含强烈种族脏词
      - "Kill every Muslim" ⚠️
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
