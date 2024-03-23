import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn import metrics
import matplotlib.pylab as plt
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
from itertools import product
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from skfeature.function.statistical_based import CFS
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import FCBF
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from boruta import BorutaPy
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN # Import SMOTEENN
from imblearn.under_sampling import NearMiss,ClusterCentroids,RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
# Load your existing features and labels

train_dataset = pd.read_csv('D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset.csv')
test_dataset = pd.read_csv('D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset.csv')

y_train = train_dataset['label']
y_train = np.array(y_train)
y_test = test_dataset['label']
y_test = np.array(y_test)
train_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_mogen_80.csv'
train_X_data1 = pd.read_csv(train_X_data_name1, header=0,  delimiter=',')
train_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_DDE_150.csv'
train_X_data2 = pd.read_csv(train_X_data_name2, header=0,  delimiter=',')
train_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TrainingDataset_blousum_70.csv'
train_X_data3 = pd.read_csv(train_X_data_name3, header=0,  delimiter=',')
X_train_data = np.concatenate((train_X_data1, train_X_data2,train_X_data3), axis=1)
# X_train_data=train_X_data1+train_X_data2 +train_X_data3
test_X_data_name1 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_mogen_80.csv'
test_X_data1 = pd.read_csv(test_X_data_name1, header=0,  delimiter=',')
test_X_data_name2 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_DDE_150.csv'
test_X_data2 = pd.read_csv(test_X_data_name2, header=0,  delimiter=',')
test_X_data_name3 = r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\TestingDataset_blousum_70.csv'
test_X_data3 = pd.read_csv(test_X_data_name3, header=0,delimiter=',')
X_test_data = np.concatenate((test_X_data1,test_X_data2,test_X_data3), axis=1)
# X_test_data = test_X_data1 +test_X_data2 +test_X_data3
X_train = np.array(X_train_data)
X_test = np.array(X_test_data)
print(X_train.shape,X_test.shape)

# # Use SMOTEENN to balance the dataset
# smote_enn = SMOTEENN(sampling_strategy='auto', random_state=0)
# X_train, y_train = smote_enn.fit_resample(X_train, y_train)
# near_miss = NearMiss(version=1)  # Choose the version of NearMiss you prefer
# X_train_resampled, y_train_resampled = near_miss.fit_resample(X_train_data, y_train)
# Ensure the resampled dataset is shuffled
# X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=0)

# # Ensure the resampled dataset is shuffled
# X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=0)
# 使用 ClusterCentroids 处理数据不平衡
# cluster_centroids = ClusterCentroids(sampling_strategy='auto', random_state=0)
# X_train_resampled, y_train_resampled = cluster_centroids.fit_resample(X_train_data, y_train)
# 使用 RandomUnderSampler 处理数据不平衡
# random_undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=0)
# X_train, y_train = random_undersampler.fit_resample(X_train_data, y_train)
# smote = SMOTE(sampling_strategy='auto', random_state=0)
# X_train, y_train = smote.fit_resample(X_train, y_train)
# Define the Boruta feature selector
# 转换 X_train 到 Pandas DataFrame
# 定义 Boruta 特征选择器
# 创建Boruta特征选择器并使用决策树分类器
# clf_rf = RandomForestClassifier(n_estimators=100, random_state=0)
# # clf_ert = ExtraTreesClassifier(n_estimators=90, random_state=0)
# # clf_lightgbm = lgb.LGBMClassifier( random_state=0)
# cfs = CFS.cfs(X_train, y_train)
# # 执行特征选择
# selected_feature_indices = cfs.feature_ranking(X_train, y_train)
# selected_feature_indices = FCBF.fcbf(X_train, y_train)
# selected_feature_indices = selected_feature_indices[:300]# 选择的特征数量可以根据需要调整
# print(selected_feature_indices)
# # # 根据选择的特征索引筛选特征
# X_train = X_train[:, selected_feature_indices]
# X_test= X_test[:, selected_feature_indices]
# print(X_train,X_train.shape)
# print(X_test,X_test.shape)
# 定义模型
clf_svm = SVC(kernel='linear', gamma=10, probability=True)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)

# clf_gbdt = GradientBoostingClassifier(n_estimators=338, learning_rate=0.025, max_depth=8,min_samples_leaf=9, min_samples_split=8,random_state=0)
clf_gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
clf_gbdt.fit(X_train, y_train)
y_pred_gbdt = clf_gbdt.predict(X_test)

clf_rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

clf_ert = ExtraTreesClassifier(n_estimators=90, max_depth=None, min_samples_split=2, random_state=0)
clf_ert.fit(X_train, y_train)
y_pred_ert = clf_ert.predict(X_test)

# 请根据你的LightGBM模型设置相应的参数
clf_lightgbm = lgb.LGBMClassifier(n_estimators=100, random_state=0)
clf_lightgbm.fit(X_train, y_train)
y_pred_lightgbm = clf_lightgbm.predict(X_test)

clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=1000, learning_rate=0.8)
clf_ada.fit(X_train, y_train)
y_pred_ada = clf_ada.predict(X_test)
# 训练和预测kNN模型
clf_knn = KNeighborsClassifier(n_neighbors=2)
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)

# 训练和预测逻辑回归模型
# clf_lr = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=1000)
clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
# 构建模型预测矩阵
predictions_nb = np.vstack((y_pred_svm,y_pred_gbdt,y_pred_rf, y_pred_dt, y_pred_ert ,y_pred_lightgbm,y_pred_knn,y_pred_ada,y_pred_lr))
# predictions_nb = np.vstack(( y_pred_svm,y_pred_gbdt,y_pred_lightgbm))
# 定义log_loss计算函数
def log_loss_func(weights):
    final_prediction = 0
    for weight, prediction in zip(weights, predictions_nb):
        final_prediction += weight * prediction
    return metrics.log_loss(y_test, final_prediction)

# 初始权重和约束
starting_values = [1 / len(predictions_nb)] * len(predictions_nb)
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(predictions_nb)

# 优化权重
# res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))
matrix_prod = np.dot(res['x'], predictions_nb)
y_prob_nb = matrix_prod

# 计算Ohp_Kmer_pred
Ohp_Kmer_pred = np.mean(predictions_nb, axis=0)
Ohp_Kmer_pred = [np.float(each > 0.5) for each in Ohp_Kmer_pred]

# 计算ROC曲线
fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test, Ohp_Kmer_pred)
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test, y_prob_nb)
print("fpr: ", fpr2)
roc_auc1 = metrics.auc(fpr1, tpr1)
roc_auc2 = metrics.auc(fpr2, tpr2)

# 计算其他性能指标
y_true = y_test
y_pred_ensemble = [np.float(each > 0.5) for each in y_prob_nb]
accuracy_score_ensemble = metrics.accuracy_score(y_true, y_pred_ensemble)
print("ACC",accuracy_score_ensemble )
# 绘制ROC曲线
labels = ['Ohp_Kmer AUC = %0.3f' % roc_auc1, 'Ensemble AUC = %0.3f' % roc_auc2]
plt.figure()
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.plot(fpr1, tpr1, label=labels[0])
plt.plot(fpr2, tpr2, label=labels[1])
plt.legend(loc='lower right')
plt.savefig(r'D:\Coding\P1\Hexapeptide Amyloidogenesis\data\AUC_compared.png')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# 计算AUPR
aupr = average_precision_score(y_true, y_prob_nb)
# 输出AUPR值
print("AUPR: ", aupr)
# 计算精确度
precision = precision_score(y_true, y_pred_ensemble)

# 计算召回率
recall = recall_score(y_true, y_pred_ensemble)

# 计算F1分数
f1 = f1_score(y_true, y_pred_ensemble)

# 计算混淆矩阵
confusion = confusion_matrix(y_true, y_pred_ensemble)

print("精确度: ", precision)
print("召回率: ", recall)
print("F1分数: ", f1)
print("混淆矩阵: ")
print(confusion)
# 计算集成模型的MCC值
mcc_ensemble = matthews_corrcoef(y_true, y_pred_ensemble)
print(f"Ensemble MCC: {mcc_ensemble}")
