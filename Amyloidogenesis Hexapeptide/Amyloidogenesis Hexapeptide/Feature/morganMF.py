import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd

# 定义摩根指纹的半径和位数
MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 1024

# 创建一个函数来提取摩根指纹特征并保存在CSV文件中
def extract_and_save_morgan_fingerprints(input_smiles_file, output_csv_file):
    # 读取包含SMILES的文件
    data = pd.read_csv(input_smiles_file)

    # 创建一个空DataFrame来保存特征
    fingerprint_df = pd.DataFrame()

    for index, row in data.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is not None:
            # 提取摩根指纹特征
            features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NUM_BITS)
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)

            # Add the features as columns in the DataFrame
            fingerprint_df[f'feature_{index}'] = features

    # 转置DataFrame
    fingerprint_df = fingerprint_df.T

    # 保存DataFrame到CSV文件
    fingerprint_df.to_csv(output_csv_file, header=False, index=False)

# 调用函数并提取特征并保存到CSV文件
input_smiles_file = r'D:\Coding\P1\ECAmyloid-main\ECAmyloid-main\dataset\test_dataset_smiles.csv'  # 替换为包含SMILES的输入文件的路径
output_csv_file = r'D:\Coding\P1\ECAmyloid-main\ECAmyloid-main\dataset\test_dataset_morgan.csv'  # 保存特征的CSV文件路径

extract_and_save_morgan_fingerprints(input_smiles_file, output_csv_file)

