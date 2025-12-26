#!/usr/bin/env python3
"""
特征向量提取工具
从训练数据中提取特征向量，或从新文本生成特征向量
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import re
from collections import Counter

def extract_features_from_text(text):
    """
    从输入文本中提取特征向量（近似版本）
    """
    print("正在从文本中提取特征向量...")

    try:
        # 文本预处理
        processed_text = preprocess_text(text)
        words = processed_text.split()
        word_count = max(len(words), 1)

        # 读取feature_columns来了解特征结构
        if os.path.exists('main/artifacts/feature_columns.json'):
            with open('main/artifacts/feature_columns.json', 'r', encoding='utf-8') as f:
                feature_columns = json.load(f)
        else:
            print("找不到feature_columns.json文件")
            return None

        # 初始化特征向量
        feature_vector = []

        # 1. weighted_TFIDF_scores - 简化为基于仇恨词的密度
        hate_words = {'hate', 'stupid', 'idiot', 'dumb', 'asshole', 'fuck', 'shit', 'bitch', 'nigger', 'cunt', 'fag', 'retard'}
        hate_count = sum(1 for word in words if word in hate_words)
        tfidf_score = hate_count / word_count if word_count > 0 else 0
        feature_vector.append(tfidf_score)

        # 2. sentiment features (6个)
        neg_words = {'bad', 'worst', 'terrible', 'awful', 'horrible', 'suck', 'angry', 'sad', 'ugly'}
        pos_words = {'good', 'great', 'awesome', 'love', 'happy', 'nice', 'beautiful', 'excellent', 'amazing', 'wonderful'}

        hate_sentiment = hate_count
        hate_density = hate_count / word_count if word_count > 0 else 0
        neg_count = sum(1 for word in words if word in neg_words)
        neg_density = neg_count / word_count if word_count > 0 else 0
        pos_count = sum(1 for word in words if word in pos_words)
        pos_density = pos_count / word_count if word_count > 0 else 0

        feature_vector.extend([hate_sentiment, hate_density, neg_count, neg_density, pos_count, pos_density])

        # 3. dependency features (40个) - 设为0（无法在简单版本中计算）
        feature_vector.extend([0] * 40)

        # 4. char_bigrams (984个) - 简化为字符二元组频率
        char_bigrams = {}
        for i in range(len(processed_text) - 1):
            bigram = processed_text[i:i+2]
            char_bigrams[bigram] = char_bigrams.get(bigram, 0) + 1

        # 按字母顺序排序并填充到984维
        sorted_bigrams = sorted(char_bigrams.items())
        for bigram, count in sorted_bigrams[:984]:
            feature_vector.append(count)
        # 填充剩余的特征为0
        while len(feature_vector) < 1 + 6 + 40 + 984:
            feature_vector.append(0)

        # 5. word_bigrams (101个) - 简化为词语二元组频率
        word_bigrams = {}
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            word_bigrams[bigram] = word_bigrams.get(bigram, 0) + 1

        sorted_word_bigrams = sorted(word_bigrams.items())
        for bigram, count in sorted_word_bigrams[:101]:
            feature_vector.append(count)
        # 填充剩余的特征为0
        while len(feature_vector) < 1 + 6 + 40 + 984 + 101:
            feature_vector.append(0)

        # 6. tfidf features (555个) - 简化为基于词频的特征
        word_freq = Counter(words)
        # 按词频排序的词作为特征
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:555]:
            feature_vector.append(freq)
        # 填充剩余的特征为0
        while len(feature_vector) < 1687:
            feature_vector.append(0)

        # 确保特征数量正确
        if len(feature_vector) > 1687:
            feature_vector = feature_vector[:1687]
        elif len(feature_vector) < 1687:
            feature_vector.extend([0] * (1687 - len(feature_vector)))

        print(f"成功提取 {len(feature_vector)} 个特征")
        return feature_vector, processed_text

    except Exception as e:
        print(f"特征提取失败: {e}")
        return None

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

def extract_feature_vector_from_training_data(sample_index=None, class_type=None):
    """
    从训练数据中提取特征向量

    参数:
    sample_index: 特定的样本索引
    class_type: 类别类型 (0=hate_speech, 1=offensive_language, 2=neither)
    """
    print("从训练数据中提取特征向量...")

    try:
        # 读取标签数据
        labels_df = pd.read_csv('test_feature_dataset/labels.csv', encoding='utf-8')

        # 选择样本
        if sample_index is not None:
            sample = labels_df[labels_df['index'] == sample_index]
            if sample.empty:
                print(f"未找到索引为 {sample_index} 的样本")
                return None
        elif class_type is not None:
            samples = labels_df[labels_df['class'] == class_type]
            if samples.empty:
                print(f"未找到类别为 {class_type} 的样本")
                return None
            sample = samples.iloc[0]  # 取第一个
        else:
            # 随机选择一个样本
            sample = labels_df.sample(1).iloc[0]

        sample_index = sample['index'].iloc[0]
        tweet_text = sample['tweet'].iloc[0]
        class_label = sample['class'].iloc[0]
        print(f"选择样本 {sample_index}")
        print(f"文本: {tweet_text[:100]}...")
        print(f"类别: {class_label} ({'hate_speech' if class_label == 0 else 'offensive_language' if class_label == 1 else 'neither'})")

        # 合并所有特征文件
        feature_files = [
            'test_feature_dataset/tfidf_scores.csv',
            'test_feature_dataset/sentiment_scores.csv',
            'test_feature_dataset/dependency_features.csv',
            'test_feature_dataset/char_bigram_features.csv',
            'test_feature_dataset/word_bigram_features.csv',
            'test_feature_dataset/tfidf_features.csv'
        ]

        feature_vector = []

        for file_path in feature_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8')
                sample_row = df[df['index'] == sample_index]

                if not sample_row.empty:
                    # 添加除index外的所有数值列
                    for col in df.columns:
                        if col != 'index':
                            try:
                                val = float(sample_row[col].iloc[0])
                                feature_vector.append(val)
                            except (ValueError, TypeError):
                                continue

        print(f"提取的特征数量: {len(feature_vector)}")

        # 验证特征数量
        if os.path.exists('main/artifacts/feature_columns.json'):
            with open('main/artifacts/feature_columns.json', 'r', encoding='utf-8') as f:
                expected_features = json.load(f)
            print(f"期望的特征数量: {len(expected_features)}")

            if len(feature_vector) == len(expected_features):
                print("特征数量匹配")
            else:
                print(f"特征数量不匹配: 得到 {len(feature_vector)}, 期望 {len(expected_features)}")
                return None

        return feature_vector, sample

    except Exception as e:
        print(f"提取失败: {e}")
        return None

def save_feature_vector_to_file(feature_vector, sample_info=None, filename="feature_vector.txt"):
    """
    将特征向量保存到文件
    """
    try:
        # 转换为逗号分隔的字符串
        feature_string = ", ".join([str(x) for x in feature_vector])

        with open(filename, 'w', encoding='utf-8') as f:
            if sample_info is not None:
                f.write(f"# 样本信息: 索引={sample_info['index']}, 类别={sample_info['class']}, 文本={sample_info['tweet'][:100]}...\n")
            f.write("# 特征向量 (1687个值，用逗号分隔)\n")
            f.write(feature_string)

        print(f"特征向量已保存到: {filename}")
        print(f"文件大小: {os.path.getsize(filename)} 字节")
        return filename

    except Exception as e:
        print(f"保存失败: {e}")
        return None

def list_available_samples():
    """
    列出可用的样本
    """
    print("可用样本统计:")
    try:
        labels_df = pd.read_csv('test_feature_dataset/labels.csv', encoding='utf-8')
        total = len(labels_df)
        print(f"总样本数: {total}")

        class_counts = labels_df['class'].value_counts().sort_index()
        label_names = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}

        for class_id, count in class_counts.items():
            print(f"  {label_names[class_id]} (类别{class_id}): {count} 个样本 ({count/total*100:.1f}%)")

        print("\n前10个仇恨言论样本:")
        hate_samples = labels_df[labels_df['class'] == 0].head(10)
        for _, row in hate_samples.iterrows():
            print(f"  索引 {row['index']}: {row['tweet'][:80]}...")

    except Exception as e:
        print(f"读取失败: {e}")

def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("1. 列出可用样本: python main/extract_features.py list")
        print("2. 提取特定样本: python main/extract_features.py extract <sample_index>")
        print("3. 提取特定类别: python main/extract_features.py extract_class <class_type>")
        print("4. 随机提取样本: python main/extract_features.py random")
        print("5. 从文本提取特征: python main/extract_features.py text '输入要检测的文本'")
        print("\n类别说明:")
        print("  0 = hate_speech (仇恨言论)")
        print("  1 = offensive_language (冒犯性语言)")
        print("  2 = neither (正常)")
        return

    command = sys.argv[1]

    if command == "list":
        list_available_samples()

    elif command == "extract" and len(sys.argv) >= 3:
        sample_index = int(sys.argv[2])
        result = extract_feature_vector_from_training_data(sample_index=sample_index)
        if result:
            feature_vector, sample_info = result
            filename = save_feature_vector_to_file(feature_vector, sample_info, f"feature_vector_{sample_index}.txt")
            if filename:
                print(f"\n特征向量已保存，可以直接复制到GUI中使用")

    elif command == "extract_class" and len(sys.argv) >= 3:
        class_type = int(sys.argv[2])
        result = extract_feature_vector_from_training_data(class_type=class_type)
        if result:
            feature_vector, sample_info = result
            filename = save_feature_vector_to_file(feature_vector, sample_info, f"feature_vector_class_{class_type}.txt")
            if filename:
                print(f"\n特征向量已保存，可以直接复制到GUI中使用")

    elif command == "random":
        result = extract_feature_vector_from_training_data()
        if result:
            feature_vector, sample_info = result
            filename = save_feature_vector_to_file(feature_vector, sample_info, "feature_vector_random.txt")
            if filename:
                print(f"\n特征向量已保存，可以直接复制到GUI中使用")

    elif command == "text":
        if len(sys.argv) >= 3:
            text = " ".join(sys.argv[2:])
            result = extract_features_from_text(text)
            if result:
                feature_vector, processed_text = result
                filename = save_feature_vector_to_file(feature_vector, {"tweet": processed_text, "class": "unknown", "index": "from_text"}, "feature_vector_from_text.txt")
                if filename:
                    print(f"\n特征向量已保存，可以直接复制到GUI中使用")
                    print(f"文本: {processed_text}")
        else:
            print("请提供要处理的文本，例如: python extract_features.py text 'I hate black people'")
    else:
        print("无效命令")

if __name__ == "__main__":
    main()
