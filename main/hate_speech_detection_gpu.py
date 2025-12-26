# 由 @Classmateliu 创建、编写及维护
# PyTorch GPU 加速版本，训练时间约为20-30min
# 最后修改日期：2025.12.25

import pandas as pd     # 读取/合并 CSV 特征表、处理表格数据
import numpy as np      # 数值数组操作、矩阵拼接、计算 ROC/均值等
import torch            # PyTorch 主库，检测 CUDA、管理设备
import torch.nn as nn   # 神经网络模块（定义 模型、线性层、激活、损失等）
import torch.optim as optim  # 优化器 权重更新
from torch.utils.data import Dataset, DataLoader
# 封装数据集与批量加载器，支持多线程预取和 GPU 训练时的批处理
from sklearn.model_selection import train_test_split, KFold
# 划分训练/测试集 交叉验证折划分
from sklearn.preprocessing import StandardScaler
# 对特征做标准化（MLP 等需要缩放），保证不同模型输入一致性
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
# roc_curve/auc: 计算 ROC 曲线与 AUC；f1_score: F1 指标（micro）；confusion_matrix: 混淆矩阵，用于错误分析
from sklearn.preprocessing import label_binarize  
# 多分类时将标签二值化，便于计算每一类别的 ROC/AUC
from sklearn.multiclass import OneVsRestClassifier  
# 将二分类评估器扩展为多分类（用于 baseline 的 decision_function）
from sklearn.linear_model import LogisticRegression  
# 逻辑回归：基线模型与 stacking 的元分类器
from sklearn.neural_network import MLPClassifier  
# sklearn 的 MLP，用于与 PyTorch MLP 做对比或参与集成
from sklearn.ensemble import VotingClassifier  
# 投票集成（soft voting），融合多个基分类器的预测概率
from mlxtend.classifier import StackingCVClassifier 
# stacking（堆叠）实现，内部使用交叉验证生成 meta-features
import matplotlib.pyplot as plt  
# 绘图（ROC 曲线等）
from numpy import interp  
# 插值函数，用于计算 macro-average ROC 时的曲线插值
import warnings
warnings.filterwarnings('ignore')  
# 屏蔽不必要的警告（如收敛警告），让输出更清爽

# 检查 GPU 是否可用 保证使用GPU运行 节约运行时间
if torch.cuda.is_available():
    device = torch.device('cuda')  
    print(f"使用设备: {device}")
    try:
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
    except Exception:
        # 备用方式，若 above 方法失败则尝试使用索引0
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

# 把多种不同来源的文本特征，按样本索引统一拼接成一个“总特征表”
# 行 一条文本样本，列 一种特征
# 只有 index 表中都存在的样本才会被保留 
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
# 设置随机种子以保证实验结果的可复现性 random_state
X_train_df, X_test_df, y_train, y_test = train_test_split(
    master.iloc[:, 3:], y, test_size=0.2, random_state=42
)

# 标准化特征（神经网络需要）
# 对数值特征进行标准化处理，使不同尺度的特征具有可比性，从而提升神经网络训练的稳定性与收敛速度
# 标准化有助于改善梯度下降过程的数值稳定性，提高模型训练效率
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df.values)
X_test_scaled = scaler.transform(X_test_df.values)

# 构建一个 基准特征集（baseline） ，仅使用加权 TF-IDF 特征
# 用于与多特征融合模型进行性能对比
# 直接从 weighted_TFIDF_scores(TFIDF特征) 中提取，确保列名匹配
# 高维稀疏 TF-IDF 是“完整表达”，加权 TF-IDF 是“压缩表达”
# 加权 TF-IDF 文本整体“攻击性词汇密度”
# 高维稀疏 TF-IDF 具体仇恨词、攻击性短语
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
# 以保证与主模型在数据预处理流程上的一致性
base_scaler = StandardScaler()
x_base_train = base_scaler.fit_transform(x_base_train)
x_base_test = base_scaler.transform(x_base_test)

# 将已预处理好的特征与标签封装为 PyTorch Dataset
# 使其能够被 DataLoader 按 batch 方式高效加载并用于 GPU 训练。
class FeatureDataset(Dataset):
    def __init__(self, features, labels):   # 监督学习
        # 确保 features 是 2D 数组
        # 通用化设计，让 Dataset 同时支持 baseline 和 多特征模型
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义一个多层感知机（MLP）神经网络
# 全连接层 + ReLU 激活 + Dropout + 最终分类层
# 4 层隐藏层的 MLP
# 80、40、40、10 表示每一层“隐藏层的神经元个数”，也就是特征被压缩/重组后的维度大小
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
    
    # forward：前向传播
    # 输入 x → 按顺序通过所有层 → 输出分类结果
    def forward(self, x):
        return self.network(x)
    # ReLU 激活函数以增强非线性表达能力，并引入 Dropout 机制以减轻过拟合问题。最终通过全连接输出层完成对文本的多类别分类

# 带学习率调度和早停机制的 PyTorch 训练函数，用于在训练集上更新模型参数
# 并根据验证集损失控制训练过程，防止过拟合
# Adam 优化器和交叉熵损失函数，并引入学习率自适应调整和早停机制
# 验证集 loss 停滞 → 自动降低学习率
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6) # 优化器（Adam） L2 正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    # 学习率调度器 监控验证集 loss；如果 连续 10 个 epoch 验证损失不下降 学习率 × 0.5
    
    # 早停机制
    # 如果验证集 loss 连续 20 次没有变好 → 停止训练
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    # 训练 + 验证
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            # 数据搬到 GPU

            optimizer.zero_grad()               # 梯度清零
            outputs = model(features)           # 前向传播
            loss = criterion(outputs, labels)   # 计算损失
            loss.backward()                     # 反向传播
            optimizer.step()                    # 参数更新
            
            train_loss += loss.item()
        
        # 验证阶段
        # 验证集性能变好 → 继续
        # 连续 20 次没变好 → 停
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
# 在不更新模型参数的前提下，对给定数据集进行前向推理
# 得到预测类别、真实标签以及各类别的预测概率，用于后续性能评估
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
            all_labels .extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# 把数据随机打乱后，做 5 折交叉验证
# print("\n开始 5 折交叉验证...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# 基准模型（加权 TF-IDF）
# 使用 仅包含加权 TF-IDF 单一特征 的 MLP 作为基准模型
# 通过 5 折交叉验证计算其平均 F1-score，用于与后续多特征模型进行对比。
print("训练基准模型（加权 TF-IDF）...")
print("\n使用 5 折交叉验证...")
base_model = MLPNet(input_dim=1, hidden_dims=[32, 16], num_classes=3).to(device)
base_scores = []

# 进入 5 折交叉验证循环 80% → 训练 20% → 验证
for fold, (train_idx, val_idx) in enumerate(kf.split(x_base_train)):
    # 从基准特征矩阵中 取训练特征 取验证特征
    X_fold_train = x_base_train[train_idx]
    X_fold_val = x_base_train[val_idx]
    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]
    
    train_dataset = FeatureDataset(X_fold_train, y_fold_train)
    val_dataset = FeatureDataset(X_fold_val, y_fold_val)
    # 训练阶段对样本进行随机打乱，而验证阶段保持数据顺序不变
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = MLPNet(input_dim=1, hidden_dims=[32, 16], num_classes=3).to(device)
    # 训练模型
    model = train_model(model, train_loader, val_loader, epochs=100, lr=0.001)
    
    preds, labels, _ = evaluate_model(model, val_loader)
    # 计算评价指标：准确率、精确率（macro）、召回率（macro）、F1（weighted）
    # 选择不同的平均方式以便得到更细粒度差异的指标
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    base_scores.append(f1)

    # 将每一折的评估结果进行记录，用于最终统计。
    if 'base_accs' not in locals():
        base_accs = []
        base_precs = []
        base_recs = []
    base_accs.append(acc)
    base_precs.append(prec)
    base_recs.append(rec)

# 交叉验证结束后，计算平均性能
print(f"基准模型（加权 TF-IDF） f1分数 (weighted) = {np.mean(base_scores):.6f}")
print(f"基准模型（加权 TF-IDF） 准确率 = {np.mean(base_accs):.6f}")
print(f"基准模型（加权 TF-IDF） 精确率 (macro) = {np.mean(base_precs):.6f}")
print(f"基准模型（加权 TF-IDF） 召回率 (macro) = {np.mean(base_recs):.6f}")


# 继续使用其他模型
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate

# 传统机器学习方法（Gradient Boosting） 只用加权 TF-IDF 基准特征
# 减少树的数量以加快速度
print("训练GB模型（加权 TF-IDF）...")
print("\n使用 3 折交叉验证...")

gb = GradientBoostingClassifier(n_estimators=200, learning_rate=.025)  # 从 500 减少到 200
gb_cv = cross_validate(gb, x_base_train, y_train, cv=3,
                       scoring={'acc':'accuracy','prec':'precision_macro','rec':'recall_macro','f1':'f1_weighted'})
print(f"GB模型（加权 TF-IDF） f1分数 (weighted) = {np.mean(gb_cv['test_f1']):.6f} (CPU, 3折)")
print(f"GB模型（加权 TF-IDF） 准确率 = {np.mean(gb_cv['test_acc']):.6f}")
print(f"GB模型（加权 TF-IDF） 精确率 (macro) = {np.mean(gb_cv['test_prec']):.6f}")
print(f"GB模型（加权 TF-IDF） 召回率 (macro) = {np.mean(gb_cv['test_rec']):.6f}")

# 随机森林 (完整特征) 
# 对完整特征集的随机森林模型进行了 3 折交叉验证
# 减少树的数量以加快速度
print("训练随机森林（完整特征）...")
print("\n使用 3 折交叉验证...")

rf = RandomForestClassifier(n_estimators=50, max_features=500)  # 从 100 减少到 50
rf_cv = cross_validate(rf, X_train_scaled, y_train, cv=3,
                       scoring={'acc':'accuracy','prec':'precision_macro','rec':'recall_macro','f1':'f1_weighted'})
print(f"随机森林（完整特征） f1分数 (weighted) = {np.mean(rf_cv['test_f1']):.6f} (CPU, 3折)")
print(f"随机森林（完整特征） 准确率 = {np.mean(rf_cv['test_acc']):.6f}")
print(f"随机森林（完整特征） 精确率 (macro) = {np.mean(rf_cv['test_prec']):.6f}")
print(f"随机森林（完整特征） 召回率 (macro) = {np.mean(rf_cv['test_rec']):.6f}")

# Extreme Gradient Boosting（极端梯度提升）
# 集成学习方法，属于 梯度提升树（Gradient Boosting Tree, GBT） 的优化实现
# XGBoost (完整特征)
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
    xgb_cv_res = cross_validate(xgb, X_train_df.values, y_train, cv=xgb_cv,
                                scoring={'acc':'accuracy','prec':'precision_macro','rec':'recall_macro','f1':'f1_weighted'})
    print(f"XGBoost (完整特征) f1分数 (weighted) = {np.mean(xgb_cv_res['test_f1']):.6f} ({'GPU, 5折' if use_gpu else 'CPU, 3折'})")
    print(f"XGBoost (完整特征) 准确率 = {np.mean(xgb_cv_res['test_acc']):.6f}")
    print(f"XGBoost (完整特征) 精确率 (macro) = {np.mean(xgb_cv_res['test_prec']):.6f}")
    print(f"XGBoost (完整特征) 召回率 (macro) = {np.mean(xgb_cv_res['test_rec']):.6f}")
    # 在完整训练集上训练 XGBoost 模型，为后续集成实验提供基础模型
    xgb.fit(X_train_df.values, y_train)
except ImportError:
    print("xgb cross validation f1-score = (XGBoost 未安装)")
    xgb = None
except Exception as e:
    print(f"xgb cross validation f1-score = (错误: {e})")
    xgb = None

# 训练sklearn MLPClassifier用于集成学习
# 为后续集成实验训练一个传统 MLP 模型，和 PyTorch MLP baseline 保持特征一致
print("\n训练sklearn MLPClassifier用于集成学习...")
mlp_sklearn = MLPClassifier(
    solver='lbfgs',
    hidden_layer_sizes=(80, 40, 40, 10),
    activation='relu',
    random_state=1,
    learning_rate='adaptive',
    alpha=1e-6,
    max_iter=500
)
# 注意：MLP在交叉验证时使用scaled数据，但在集成学习时使用原始数据
mlp_sklearn.fit(X_train_df.values, y_train)

# 训练GB和RF用于集成学习
print("训练GB和RF模型用于集成学习...")
gb.fit(x_base_train, y_train)  # GB只用 加权 TF-IDF 基准特征
rf.fit(X_train_df.values, y_train)  # RF使用 完整特征集
# GB（基准特征 + 标准化）RF（完整特征 + 原始数据）

# 初始化集成学习
print("\n初始化集成学习...")
estimators = []
if xgb is not None:
    # XGBoost已经在上面训练过了，使用原始特征
    estimators.append(('mlp', mlp_sklearn))
    estimators.append(('rf', rf))
    estimators.append(('xgb', xgb))
    
    # Voting 集成
    print("训练Voting集成分类器...")
    ensemble = VotingClassifier(estimators, voting='soft', weights=[1, 1, 1])
    ensemble.fit(X_train_df.values, y_train)
    # Voting 预测与评估
    pred_ensemble = ensemble.predict(X_test_df.values)
    # 使用不同平均策略以获得更细粒度差异
    ve_f1 = f1_score(y_test, pred_ensemble, average="weighted")
    ve_acc = accuracy_score(y_test, pred_ensemble)
    ve_prec = precision_score(y_test, pred_ensemble, average="macro", zero_division=0)
    ve_rec = recall_score(y_test, pred_ensemble, average="macro", zero_division=0)
    print(f'Voting 集成 f1分数: {ve_f1:.3f}')
    print(f'Voting 集成 准确率: {ve_acc:.3f}')
    print(f'Voting 集成 精确率: {ve_prec:.3f}')
    print(f'Voting 集成 召回率: {ve_rec:.3f}')
    # 每类指标与 macro 汇总（更细粒度）
    print("\nVoting 集成 每类指标:")
    print(classification_report(y_test, pred_ensemble, digits=4))
    print(f"Voting 集成 macro F1: {f1_score(y_test, pred_ensemble, average='macro'):.4f}")
    
    # Stacking 集成
    # Stacking 利用元模型融合基模型预测，进一步提升多分类性能
    # 逻辑回归 Stacking 的元模型
    print("训练Stacking集成分类器...")
    lr_meta = LogisticRegression(solver='lbfgs', max_iter=500)
    # 为了避免 stacking 在交叉验证内并行训练 XGBoost 导致长时间阻塞或中断，
    # 在这里使用轻量化的 XGBoost 实例供 stacking 使用（较少的 n_estimators）
    try:
        from xgboost import XGBClassifier as XGBClf
        xgb_stack_kwargs = {'learning_rate': .025, 'max_depth': 4, 'n_estimators': 50}
        if use_gpu:
            xgb_stack_kwargs.update({'device': 'cuda', 'tree_method': 'hist'})
        xgb_stack = XGBClf(**xgb_stack_kwargs)
    except Exception:
        # 如果无法构造轻量 xgb，则回退使用原始 xgb（可能更慢）
        xgb_stack = xgb

    # 限制 stacking 的并行度为 1（避免与 joblib / xgboost 并行冲突）
    stack = StackingCVClassifier(classifiers=[mlp_sklearn, xgb_stack, rf],
                                 cv=2, meta_classifier=lr_meta, use_probas=True, n_jobs=1)
    try:
        stack.fit(X_train_df.values, y_train)
    except KeyboardInterrupt:
        print("训练过程中收到中断（KeyboardInterrupt），已跳过 Stacking 训练。")
        stack = None
    pred_stack = stack.predict(X_test_df.values)
    # 使用不同平均策略以获得更细粒度差异
    st_f1 = f1_score(y_test, pred_stack, average="weighted")
    st_acc = accuracy_score(y_test, pred_stack)
    st_prec = precision_score(y_test, pred_stack, average="macro", zero_division=0)
    st_rec = recall_score(y_test, pred_stack, average="macro", zero_division=0)
    print(f'Stacking 集成 f1分数: {st_f1:.3f}')
    print(f'Stacking 集成 准确率: {st_acc:.3f}')
    print(f'Stacking 集成 精确率: {st_prec:.3f}')
    print(f'Stacking 集成 召回率: {st_rec:.3f}')
    # 每类指标与 macro 汇总（更细粒度）
    print("\nStacking 集成 每类指标:")
    print(classification_report(y_test, pred_stack, digits=4))
    print(f"Stacking 集成 macro F1: {f1_score(y_test, pred_stack, average='macro'):.4f}")
    
    # Voting集成混淆矩阵
    # 显示 Voting 集成在测试集上的 混淆矩阵
    # 用于分析各类别预测情况 哪些类别容易混淆 哪些类别预测效果好
    confusion_ensemble = confusion_matrix(y_test, pred_ensemble)
    print("\nVoting 集成 混淆矩阵:")
    print(confusion_ensemble)
else:
    print("XGBoost未安装，跳过集成学习")

# 完整特征模型（PyTorch MLP） 完整特征模型训练 + 5 折交叉验证
print(f"\n训练完整特征模型（PyTorch MLP，使用 {device}）...")
full_scores = []

# 5 折交叉验证循环
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_fold_train = X_train_scaled[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]
    
    train_dataset = FeatureDataset(X_fold_train, y_fold_train)
    val_dataset = FeatureDataset(X_fold_val, y_fold_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 初始化 MLP 模型 训练模型
    model = MLPNet(input_dim=X_train_scaled.shape[1], hidden_dims=[80, 40, 40, 10], num_classes=3).to(device)
    model = train_model(model, train_loader, val_loader, epochs=100, lr=0.001)
    
    preds, labels, _ = evaluate_model(model, val_loader)
    # 计算多项评价指标：accuracy / precision(macro) / recall(macro) / f1(weighted)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    full_scores.append(f1)
    if 'full_accs' not in locals():
        full_accs = []
        full_precs = []
        full_recs = []
    full_accs.append(acc)
    full_precs.append(prec)
    full_recs.append(rec)

# 输出 5 折交叉验证平均指标
print(f"完整特征模型（PyTorch MLP） f1分数 = {np.mean(full_scores):.6f}")
print(f"完整特征模型（PyTorch MLP） 准确率 = {np.mean(full_accs):.6f}")
print(f"完整特征模型（PyTorch MLP） 精确率 = {np.mean(full_precs):.6f}")
print(f"完整特征模型（PyTorch MLP） 召回率 = {np.mean(full_recs):.6f}")

# 在完整测试集上训练最终模型
print("\n训练完整模型（完整训练集（80%） 无交叉验证）...")
train_dataset = FeatureDataset(X_train_scaled, y_train)
test_dataset = FeatureDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化最终模型 训练最终模型
final_model = MLPNet(input_dim=X_train_scaled.shape[1], hidden_dims=[80, 40, 40, 10], num_classes=3).to(device)
final_model = train_model(final_model, train_loader, test_loader, epochs=150, lr=0.001)

# 在测试集上预测
pred, y_true, y_probs = evaluate_model(final_model, test_loader)

# 计算并输出多项指标：Accuracy / Precision(macro) / Recall(macro) / F1(weighted)
final_acc = accuracy_score(y_test, pred)
final_prec = precision_score(y_test, pred, average='macro', zero_division=0)
final_rec = recall_score(y_test, pred, average='macro', zero_division=0)
final_f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
print(f'\n完整模型 f1分数 (weighted): {final_f1:.3f}')
print(f'完整模型 准确率: {final_acc:.3f}')
print(f'完整模型 精确率 (macro): {final_prec:.3f}')
print(f'完整模型 召回率 (macro): {final_rec:.3f}')

# 输出混淆矩阵
confusion_mat = confusion_matrix(y_test, pred)
print("\n混淆矩阵:")
print(confusion_mat)
print("\n完整模型（PyTorch MLP） 每类指标:")
print(classification_report(y_test, pred, digits=4))
print(f"完整模型 macro F1: {f1_score(y_test, pred, average='macro'):.4f}")

# ROC 曲线计算
print("\n计算 ROC 曲线...")
y_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_binarized.shape[1]

# 基准模型（Logistic Regression）预测概率
base_lr = LogisticRegression(solver='lbfgs', max_iter=500)
base_lr.fit(x_base_train, y_train)
base_probs = base_lr.predict_proba(x_base_test)
# 基准模型在测试集上的预测与更细粒度评估
pred_base = base_lr.predict(x_base_test)
print("\n基准模型（逻辑回归） 每类指标:")
print(classification_report(y_test, pred_base, digits=4))
print(f"基准模型 macro F1: {f1_score(y_test, pred_base, average='macro'):.4f}")

# 完整模型 ROC（使用 PyTorch 模型的概率输出）
# 最终 PyTorch MLP 模型 的输出概率
_, _, full_probs = evaluate_model(final_model, test_loader)

# 基准模型使用 OneVsRest 计算 decision_function
classifier_base = OneVsRestClassifier(base_lr)
classifier_base.fit(x_base_train, y_train)
y_score_base = classifier_base.decision_function(x_base_test)

# XGBoost 测试集概率
y_score_xgb = None
if xgb is not None:
    try:
        # XGBoost 训练时使用原始特征 X_train_df.values
        xgb_probs = xgb.predict_proba(X_test_df.values)
        y_score_xgb = xgb_probs
    except Exception:
        # 如果 predict_proba 不可用，则尝试使用 predict
        xgb_preds = xgb.predict(X_test_df.values)
        # 将预测转换为 one-hot 概率
        y_score_xgb = np.zeros((len(xgb_preds), 3))
        for i, p in enumerate(xgb_preds):
            y_score_xgb[i, int(p)] = 1.0
    # XGBoost 在测试集上的更细粒度评估
    try:
        xgb_preds_final = xgb.predict(X_test_df.values)
        print("\nXGBoost 每类指标:")
        print(classification_report(y_test, xgb_preds_final, digits=4))
        print(f"XGBoost macro F1: {f1_score(y_test, xgb_preds_final, average='macro'):.4f}")
    except Exception:
        pass

# PyTorch MLP logits 转为 score
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
    # 如果有 XGBoost 的概率，也计算其 ROC 曲线
    if y_score_xgb is not None:
        if 'fpr3' not in locals():
            fpr3 = dict()
            tpr3 = dict()
            roc_auc3 = dict()
        fpr3[i], tpr3[i], _ = roc_curve(y_binarized[:, i], y_score_xgb[:, i])
        roc_auc3[i] = auc(fpr3[i], tpr3[i])

# Micro-average ROC
fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_score_base.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
fpr2["micro"], tpr2["micro"], _ = roc_curve(y_binarized.ravel(), y_score_full.ravel())
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])
if y_score_xgb is not None:
    # 计算 XGBoost 的 micro-average
    fpr3 = dict() if 'fpr3' not in locals() else fpr3
    tpr3 = dict() if 'tpr3' not in locals() else tpr3
    fpr3["micro"], tpr3["micro"], _ = roc_curve(y_binarized.ravel(), y_score_xgb.ravel())
    roc_auc3 = dict() if 'roc_auc3' not in locals() else roc_auc3
    roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])

# Macro-average ROC
# 整体类别的平均 ROC 曲线
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
# 用 基准模型（逻辑回归）、完整模型（PyTorch MLP） 和可选的 XGBoost 在测试集上得到每个类别的预测分数（概率或 logits）
# 将多分类问题转化为 OneVsRest 形式，分别计算每个类别的 FPR/TPR 并求 AUC，接着计算 micro-average（按样本）和 macro-average（按类别）ROC，以便全面评估各模型的分类性能。

# 绘制 ROC 曲线
# 横轴：模型误把负样本预测成正样本的比例（越低越好）
# 纵轴：模型正确预测正样本的比例（越高越好）
# Micro-average ROC：在多分类中把所有样本当作一个整体来计算 ROC，用于衡量模型整体分类性能
# 解决 matplotlib 中文字体显示为方框的问题：优先使用系统常见中文字体（Windows 常见为 Microsoft YaHei / SimHei）
import matplotlib.font_manager as fm
font_candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
available_fonts = [f.name for f in fm.fontManager.ttflist]
for fname in font_candidates:
    if fname in available_fonts:
        plt.rcParams['font.sans-serif'] = [fname]
        plt.rcParams['font.family'] = 'sans-serif'
        break
# 解决坐标轴负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 8))
# 绘制基准模型（逻辑回归）的 micro-average ROC，颜色：橙色
line_base, = plt.plot(fpr["micro"], tpr["micro"],
         label=f'基准模型（逻辑回归） micro-average ROC 曲线 (AUC = {roc_auc["micro"]:.2f})',
         color='#EB7D3C', linestyle=':', linewidth=4)
# 绘制完整特征 MLP 的 micro-average ROC，颜色：蓝色
line_mlp, = plt.plot(fpr2["micro"], tpr2["micro"],
         label=f'完整模型（PyTorch MLP） micro-average ROC 曲线 (AUC = {roc_auc2["micro"]:.2f})',
         color='#4674C1', linestyle=':', linewidth=4)
line_xgb = None
if 'fpr3' in locals() and "micro" in fpr3:
    # 绘制 XGBoost 的 micro-average ROC，颜色：绿色
    line_xgb, = plt.plot(fpr3["micro"], tpr3["micro"],
             label=f'XGBoost micro-average ROC 曲线 (AUC = {roc_auc3["micro"]:.2f})',
             color='#72AC48', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)')
plt.ylabel('真正例率 (True Positive Rate)')
plt.title('Micro-Average ROC 曲线 — 各模型总体性能比较')
# 第一部分图例：显示每条曲线及其 AUC
ax = plt.gca()
legend1 = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True)
ax.add_artist(legend1)
# 第二部分图例：颜色图示，明确颜色对应哪个模型
from matplotlib.patches import Patch
handles = [Patch(color='#EB7D3C', label='橙色 — 基准模型'),
           Patch(color='#4674C1', label='蓝色 — 完整模型（PyTorch MLP）')]
if line_xgb is not None:
    handles.append(Patch(color='#72AC48', label='绿色 — XGBoost'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1)
plt.show()

# 评估模型在类别仇恨言论上的分类性能
# 适合用于分析某个重要类别的性能，比如关注仇恨言论检测效果
plt.figure(figsize=(10, 8))
lw = 2
line_base, = plt.plot(fpr[0], tpr[0], color='#EB7D3C',
         lw=lw, label=f'基准模型 ROC 曲线 (AUC = {roc_auc[0]:.2f})')
line_mlp, = plt.plot(fpr2[0], tpr2[0], color='#4674C1',
         lw=lw, label=f'完整模型（MLP） ROC 曲线 (AUC = {roc_auc2[0]:.2f})')
line_xgb = None
if 'fpr3' in locals() and 0 in fpr3:
    line_xgb, = plt.plot(fpr3[0], tpr3[0], color='#72AC48',
             lw=lw, label=f'XGBoost ROC 曲线 (AUC = {roc_auc3[0]:.2f})')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)')
plt.ylabel('真正例率 (True Positive Rate)')
plt.title('仇恨言论ROC 曲线 — 模型单类性能对比')
# 第一部分图例：显示每条曲线及其 AUC
ax = plt.gca()
legend1 = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True)
ax.add_artist(legend1)
# 第二部分图例：颜色图示，明确颜色对应哪个模型
from matplotlib.patches import Patch
handles = [Patch(color='#EB7D3C', label='橙色 — 基准模型（逻辑回归）'),
           Patch(color='#4674C1', label='蓝色 — 完整模型（MLP）')]
if line_xgb is not None:
    handles.append(Patch(color='#72AC48', label='绿色 — XGBoost'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1)
plt.show()

print("\n训练完成！");