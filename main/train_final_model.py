# 由 @Classmateliu 创建、编写及维护
# 仅训练最终版模型 PyTorch MLP (完整特征) 作为最终使用的模型
# 最后修改日期：2025.12.25

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os, joblib, json
import warnings
warnings.filterwarnings('ignore')

# 检查 GPU 是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    try:
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
    except Exception:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "Unknown GPU"
    print(f"GPU 名称: {gpu_name}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
else:
    device = torch.device('cpu')
    print(f"使用设备: {device} (未检测到 CUDA GPU)")

# 读取特征 CSV 文件
print("正在加载特征数据...")
class_labels = pd.read_csv('test_feature_dataset/labels.csv', encoding='utf-8')
weighted_tfidf_score = pd.read_csv('test_feature_dataset/tfidf_scores.csv', encoding='utf-8')
sentiment_scores = pd.read_csv('test_feature_dataset/sentiment_scores.csv', encoding='utf-8')
dependency_features = pd.read_csv('test_feature_dataset/dependency_features.csv', encoding='utf-8')
char_bigrams = pd.read_csv('test_feature_dataset/char_bigram_features.csv', encoding='utf-8')
word_bigrams = pd.read_csv('test_feature_dataset/word_bigram_features.csv', encoding='utf-8')
tfidf_sparse_matrix = pd.read_csv('test_feature_dataset/tfidf_features.csv', encoding='utf-8')

# 合并所有特征
df_list = [class_labels, weighted_tfidf_score, sentiment_scores, dependency_features,
           char_bigrams, word_bigrams, tfidf_sparse_matrix]
master = df_list[0]
for df in df_list[1:]:
    master = master.merge(df, on='index')

# 提取特征和标签
y = master.iloc[:, 2].values  # y：类别标签
X = master.iloc[:, 3:].values  # X：输入特征
print(f"特征维度: {X.shape}, 标签维度: {y.shape}")

# 创建训练集和测试集 (80% train, 20% test)
X_train_df, X_test_df, y_train, y_test = train_test_split(
    master.iloc[:, 3:], y, test_size=0.2, random_state=42
)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df.values)
X_test_scaled = scaler.transform(X_test_df.values)

# 构建基准特征集
try:
    x_base_train = X_train_df[['weighted_TFIDF_scores']].values
    x_base_test = X_test_df[['weighted_TFIDF_scores']].values
except KeyError:
    tfidf_cols = [col for col in X_train_df.columns if 'TFIDF' in col.upper() or 'tfidf' in col.lower()]
    if tfidf_cols:
        x_base_train = X_train_df[[tfidf_cols[0]]].values
        x_base_test = X_test_df[[tfidf_cols[0]]].values
        print(f"使用列名: {tfidf_cols[0]}")
    else:
        x_base_train = X_train_df.iloc[:, 0:1].values
        x_base_test = X_test_df.iloc[:, 0:1].values
        print("警告: 未找到 weighted_TFIDF_scores 列，使用第一列作为基准特征")

# 标准化基准特征
base_scaler = StandardScaler()
x_base_train = base_scaler.fit_transform(x_base_train)
x_base_test = base_scaler.transform(x_base_test)

# PyTorch Dataset
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# MLP 模型
class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[80, 40, 40, 10], num_classes=3, dropout=0.1):
        super(MLPNet, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 训练函数 (与gpu文件中的train_model函数保持一致)
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6) # 优化器（Adam） L2 正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break

    return model

# 评估函数
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_and_save_final_model():
    print("\n开始训练最终模型，并保存必要的artifacts...")

    # 创建训练和测试数据集 (与gpu文件中的代码完全一致)
    train_dataset = FeatureDataset(X_train_scaled, y_train)
    test_dataset = FeatureDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 计算类别权重（处理严重类别不平衡）
    # 注意：虽然gpu文件中没有使用类别权重，但由于数据严重不平衡(13.42倍)，
    # 我们需要类别权重来确保模型不会偏向多数类别(offensive_language)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    print(f"训练集类别分布: {np.bincount(y_train)}")
    print(f"计算得到的类别权重: {class_weights}")
    print("注意: hate_speech权重最高，因为样本最少")

    # 初始化最终模型
    final_model = MLPNet(input_dim=X_train_scaled.shape[1], hidden_dims=[80, 40, 40, 10], num_classes=3).to(device)

    # 训练函数需要支持类别权重
    def train_model_with_weights(model, train_loader, val_loader, epochs=100, lr=0.001, class_weights=None):
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"使用类别权重: {class_weights.cpu().numpy()}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("未使用类别权重")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break

        return model

    final_model = train_model_with_weights(final_model, train_loader, test_loader, epochs=150, lr=0.001, class_weights=class_weights)

    # 保存训练产物
    artifacts_dir = os.path.join('main', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    # final_model_state_dict PyTorch 模型参数 存储训练好的 MLP 权重
    torch.save(final_model.state_dict(), os.path.join(artifacts_dir, 'final_model_state_dict.pth'))

    # scaler.pkl 特征标准化器 保证训练与推理特征分布一致
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.pkl'))

    # base_scaler.pkl 基准模型标准化器 用于对比实验
    try:
        joblib.dump(base_scaler, os.path.join(artifacts_dir, 'base_scaler.pkl'))
    except Exception:
        pass

    # feature_columns.json 特征顺序说明 保证特征维度和语义不乱
    try:
        feature_columns = master.columns.tolist()[3:]
        with open(os.path.join(artifacts_dir, 'feature_columns.json'), 'w', encoding='utf-8') as f:
            json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # label_map.json 标签映射表 把模型输出转成可读类别
    label_map = {"0": "hate_speech", "1": "offensive_language", "2": "neither"}
    with open(os.path.join(artifacts_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # 在测试集上评估
    pred_original, y_true, y_probs = evaluate_model(final_model, test_loader)

    original_acc = accuracy_score(y_test, pred_original)
    original_prec = precision_score(y_test, pred_original, average='macro', zero_division=0)
    original_rec = recall_score(y_test, pred_original, average='macro', zero_division=0)
    original_f1 = f1_score(y_test, pred_original, average='weighted', zero_division=0)

    print(f"F1分数 (weighted): {original_f1:.3f}")
    print(f"准确率: {original_acc:.3f}")
    print(f"精确率 (macro): {original_prec:.3f}")
    print(f"召回率 (macro): {original_rec:.3f}")

    print("\n最终模型训练完成！")
    print(f"Artifacts 已保存到: {artifacts_dir}")
    print("训练完成！")

if __name__ == "__main__":
    train_and_save_final_model()
