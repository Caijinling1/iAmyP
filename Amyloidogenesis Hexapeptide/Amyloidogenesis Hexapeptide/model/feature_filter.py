import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# 加载数据
train_dataset = pd.read_csv(r'D:\Coding\P1\aggregating peptides\HA.csv')
test_dataset = pd.read_csv(r'D:\Coding\P1\aggregating peptides\7-10peptidde\7.csv')

y_train = train_dataset['label']
y_train = np.array(y_train)
y_test = test_dataset['label']
y_test = np.array(y_test)
scaler=StandardScaler()
train_X_data_name4 = r'D:\Coding\P1\aggregating peptides\HA_DDE.csv'
train_X_data4 = pd.read_csv(train_X_data_name4, header=0,index_col=0, delimiter=',')
test_X_data_name4 = r'D:\Coding\P1\aggregating peptides\7-10peptidde\7_DDE.csv'
test_X_data4 = pd.read_csv(test_X_data_name4, header=0,index_col=0,delimiter=',')
X_train = np.array(train_X_data4)
X_test = np.array(test_X_data4)

# print(X_train.shape,X_test.shape)
# 对X_train进行归一化处理
X_train = scaler.fit_transform(X_train)
# 使用相同的scaler对象对X_test进行归一化处理
X_test = scaler.transform(X_test)
# print(X_train)
# print(X_test)

# # # Create the LGBM classifier for feature selection
clf_lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=0)

# # 使用XGBoost进行特征选择
num_features_to_select =150# 指定要选择的特征数量
rfe = RFE(estimator=clf_lgbm, n_features_to_select=num_features_to_select)

# Fit the RFE model
rfe.fit(X_train, y_train)
#
# Get the selected feature indices
selected_feature_indices = np.where(rfe.support_)[0]
print(selected_feature_indices)
#
# Apply the same feature selection to the training and testing data
X_train = X_train[:, selected_feature_indices]
X_test = X_test[:, selected_feature_indices]
print(X_train, X_train.shape)
print(X_test, X_test.shape)
train_df = pd.DataFrame(X_train)
test_df = pd.DataFrame(X_test)
