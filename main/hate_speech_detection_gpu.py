# PyTorch GPU 加速版本
# 基于原始 CPU 版本改写，保持输入输出格式一致
# Author: Adapted from Tommy Pawelski's original code
# Created: 2025

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from numpy import interp
import warnings
warnings.filterwarnings('ignore')

# 检查 GPU 是否可用
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # 明确指定使用第一个 CUDA 设备（NVIDIA GPU）
    print(f"使用设备: {device}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
else:
    device = torch.device('cpu')
    print(f"使用设备: {device} (未检测到 CUDA GPU)")

# 读取特征 CSV 文件（和 CPU 版本完全一致）
print("正在加载特征数据...")
class_labels = pd.read_csv('test_feature_dataset/labels.csv', encoding='utf-8')
weighted_tfidf_score = pd.read_csv('test_feature_dataset/tfidf_scores.csv', encoding='utf-8')
sentiment_scores = pd.read_csv('test_feature_dataset/sentiment_scores.csv', encoding='utf-8')
dependency_features = pd.read_csv('test_feature_dataset/dependency_features.csv', encoding='utf-8')
char_bigrams = pd.read_csv('test_feature_dataset/char_bigram_features.csv', encoding='utf-8')
word_bigrams = pd.read_csv('test_feature_dataset/word_bigram_features.csv', encoding='utf-8')
tfidf_sparse_matrix = pd.read_csv('test_feature_dataset/tfidf_features.csv', encoding='utf-8')

# 合并所有特征数据集
df_list = [class_labels, weighted_tfidf_score, sentiment_scores, dependency_features, 
           char_bigrams, word_bigrams, tfidf_sparse_matrix]
master = df_list[0]
for df in df_list[1:]:
    master = master.merge(df, on='index')

# 提取特征和标签
y = master.iloc[:, 2].values  # class labels
X = master.iloc[:, 3:].values  # all features

print(f"特征维度: {X.shape}, 标签维度: {y.shape}")

# 创建训练集和测试集 (80% train, 20% test)
X_train_df, X_test_df, y_train, y_test = train_test_split(
    master.iloc[:, 3:], y, test_size=0.2, random_state=42
)

# 标准化特征（神经网络需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df.values)
X_test_scaled = scaler.transform(X_test_df.values)

# 创建基准特征集（只用 weighted_TFIDF_scores）
# 直接从 DataFrame 中提取，确保列名匹配
try:
    x_base_train = X_train_df[['weighted_TFIDF_scores']].values
    x_base_test = X_test_df[['weighted_TFIDF_scores']].values
except KeyError:
    # 如果列名不匹配，尝试查找包含 TFIDF 的列
    tfidf_cols = [col for col in X_train_df.columns if 'TFIDF' in col.upper() or 'tfidf' in col.lower()]
    if tfidf_cols:
        x_base_train = X_train_df[[tfidf_cols[0]]].values
        x_base_test = X_test_df[[tfidf_cols[0]]].values
        print(f"使用列名: {tfidf_cols[0]}")
    else:
        # 如果找不到，使用第一列
        x_base_train = X_train_df.iloc[:, 0:1].values
        x_base_test = X_test_df.iloc[:, 0:1].values
        print("警告: 未找到 weighted_TFIDF_scores 列，使用第一列作为基准特征")

# 标准化基准特征
base_scaler = StandardScaler()
x_base_train = base_scaler.fit_transform(x_base_train)
x_base_test = base_scaler.transform(x_base_test)

# 定义 PyTorch Dataset
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        # 确保 features 是 2D 数组
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义神经网络模型（类似 MLP）
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

# 训练函数
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        # 训练阶段
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
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        scheduler.step(val_loss)
        
        # 早停机制
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

# 5折交叉验证
print("\n开始 5 折交叉验证...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# 基准模型（只用 weighted_TFIDF_scores）
print("训练基准模型（只用 weighted_TFIDF_scores）...")
base_model = MLPNet(input_dim=1, hidden_dims=[32, 16], num_classes=3).to(device)
base_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x_base_train)):
    X_fold_train = x_base_train[train_idx]
    X_fold_val = x_base_train[val_idx]
    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]
    
    train_dataset = FeatureDataset(X_fold_train, y_fold_train)
    val_dataset = FeatureDataset(X_fold_val, y_fold_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = MLPNet(input_dim=1, hidden_dims=[32, 16], num_classes=3).to(device)
    model = train_model(model, train_loader, val_loader, epochs=100, lr=0.001)
    
    preds, labels, _ = evaluate_model(model, val_loader)
    score = f1_score(labels, preds, average='micro')
    base_scores.append(score)

print(f"baseline model f1-score = {np.mean(base_scores):.6f}")

# 为了和 CPU 版本输出一致，也测试其他模型
print("\n测试其他模型...")
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Gradient Boosting (只用 x_base) - sklearn 不支持 GPU，在 CPU 上运行
# 减少树的数量以加快速度
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=.025)  # 从 500 减少到 200
gb_scores = cross_val_score(gb, x_base_train, y_train, cv=3, scoring="f1_micro")  # CPU 模型用 3 折
print(f"gb cross validation f1-score = {gb_scores.mean():.6f} (CPU, 3折)")

# Random Forest (完整特征) - sklearn 不支持 GPU，在 CPU 上运行
# 减少树的数量以加快速度
rf = RandomForestClassifier(n_estimators=50, max_features=500)  # 从 100 减少到 50
rf_scores = cross_val_score(rf, X_train_scaled, y_train, cv=3, scoring="f1_micro")  # CPU 模型用 3 折
print(f"rf cross validation f1-score = {rf_scores.mean():.6f} (CPU, 3折)")

# XGBoost (完整特征) - 尝试使用 GPU 加速
try:
    from xgboost import XGBClassifier
    if torch.cuda.is_available():
        # XGBoost 3.1.2 版本使用 device='cuda' 参数来启用 GPU
        # tree_method 使用 'hist' 或 'approx'，GPU 会自动处理
        try:
            xgb = XGBClassifier(learning_rate=.025, max_depth=6, device='cuda', tree_method='hist')
            print("XGBoost 使用 GPU 加速")
            use_gpu = True
            xgb_cv = 5  # GPU 版本用 5 折
        except Exception as e:
            # 如果 GPU 配置失败，回退到 CPU
            print(f"XGBoost GPU 配置失败: {e}, 回退到 CPU")
            xgb = XGBClassifier(learning_rate=.025, max_depth=6)
            use_gpu = False
            xgb_cv = 3  # CPU 版本用 3 折
    else:
        xgb = XGBClassifier(learning_rate=.025, max_depth=6)
        print("XGBoost 使用 CPU（未检测到 GPU）")
        use_gpu = False
        xgb_cv = 3  # CPU 版本用 3 折
    xgb_scores = cross_val_score(xgb, X_train_scaled, y_train, cv=xgb_cv, scoring="f1_micro")
    print(f"xgb cross validation f1-score = {xgb_scores.mean():.6f} ({'GPU, 5折' if use_gpu else 'CPU, 3折'})")
except ImportError:
    print("xgb cross validation f1-score = (XGBoost 未安装)")
except Exception as e:
    print(f"xgb cross validation f1-score = (错误: {e})")

# 完整特征模型（PyTorch MLP，使用 GPU）
print(f"\n训练完整特征模型（PyTorch MLP，使用 {device}）...")
full_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_fold_train = X_train_scaled[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]
    
    train_dataset = FeatureDataset(X_fold_train, y_fold_train)
    val_dataset = FeatureDataset(X_fold_val, y_fold_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = MLPNet(input_dim=X_train_scaled.shape[1], hidden_dims=[80, 40, 40, 10], num_classes=3).to(device)
    model = train_model(model, train_loader, val_loader, epochs=100, lr=0.001)
    
    preds, labels, _ = evaluate_model(model, val_loader)
    score = f1_score(labels, preds, average='micro')
    full_scores.append(score)

print(f"mlp cross validation f1-score = {np.mean(full_scores):.6f}")

# 在完整测试集上训练最终模型
print("\n训练最终模型...")
train_dataset = FeatureDataset(X_train_scaled, y_train)
test_dataset = FeatureDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

final_model = MLPNet(input_dim=X_train_scaled.shape[1], hidden_dims=[80, 40, 40, 10], num_classes=3).to(device)
final_model = train_model(final_model, train_loader, test_loader, epochs=150, lr=0.001)

# 预测
pred, y_true, y_probs = evaluate_model(final_model, test_loader)
print(f'\nfscore:{f1_score(y_test, pred, average="micro"):.3f}')

# 混淆矩阵
confusion_mat = confusion_matrix(y_test, pred)
print("\n混淆矩阵:")
print(confusion_mat)

# ROC 曲线计算
print("\n计算 ROC 曲线...")
y_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_binarized.shape[1]

# 基准模型 ROC（使用 OneVsRest 策略）
# 为了和原版输出格式一致，基准模型用简单的逻辑回归
base_lr = LogisticRegression(solver='lbfgs', max_iter=500)
base_lr.fit(x_base_train, y_train)
base_probs = base_lr.predict_proba(x_base_test)

# 完整模型 ROC（使用 PyTorch 模型的概率输出）
_, _, full_probs = evaluate_model(final_model, test_loader)

# 使用 OneVsRest 策略计算 ROC（和原版一致）
classifier_base = OneVsRestClassifier(base_lr)
classifier_base.fit(x_base_train, y_train)
y_score_base = classifier_base.decision_function(x_base_test)

# PyTorch 模型需要转换为 decision_function 格式（使用 logits）
# 获取模型的原始输出（logits）
final_model.eval()
y_score_full = []
with torch.no_grad():
    for features, _ in test_loader:
        features = features.to(device)
        outputs = final_model(features)
        y_score_full.extend(outputs.cpu().numpy())
y_score_full = np.array(y_score_full)

# 计算每个类别的 ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_score_base[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    fpr2[i], tpr2[i], _ = roc_curve(y_binarized[:, i], y_score_full[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])

# Micro-average ROC
fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_score_base.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
fpr2["micro"], tpr2["micro"], _ = roc_curve(y_binarized.ravel(), y_score_full.ravel())
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

# Macro-average ROC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
all_fpr2 = np.unique(np.concatenate([fpr2[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
mean_tpr2 = np.zeros_like(all_fpr2)

for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr2 += interp(all_fpr2, fpr2[i], tpr2[i])

mean_tpr /= n_classes
mean_tpr2 /= n_classes

fpr["micro"] = all_fpr
tpr["micro"] = mean_tpr
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
fpr2["micro"] = all_fpr2
tpr2["micro"] = mean_tpr2
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

# 绘制 ROC 曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label='Base model micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='#EB7D3C', linestyle=':', linewidth=4)
plt.plot(fpr2["micro"], tpr2["micro"],
         label='MLP micro-average ROC curve (area = {0:0.2f})'.format(roc_auc2["micro"]),
         color='#4674C1', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-Average ROC (PyTorch GPU)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.15), fancybox=True)
plt.show()

# 绘制类别 0（仇恨言论）的 ROC
plt.figure(figsize=(10, 8))
lw = 2
plt.plot(fpr[0], tpr[0], color='#EB7D3C',
         lw=lw, label='Base model ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot(fpr2[0], tpr2[0], color='#4674C1',
         lw=lw, label='MLP ROC curve (area = %0.2f)' % roc_auc2[0])
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for "Hatespeech" Label (PyTorch GPU)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.15), fancybox=True)
plt.show()

print("\n训练完成！")

