import pandas as pd

# 读取CSV文件
data = pd.read_csv(r'D:\Coding\P1\aggregating peptides\7-10peptidde\7-10aggpeptide.csv')


# 定义一个函数，根据疏水性残基的数量分为不同的类别
def classify_sequence(row):
    hydrophobic_residues = "EACYVILMFW"  # 疏水性残基
    count = sum(row['sequence'].count(residue) for residue in hydrophobic_residues)

    # 将疏水性残基的数量映射到0-6的范围，可以根据实际需求调整
    if count == 0:
        return 0
    elif count == 1:
        return 1
    elif count == 2:
        return 2
    elif count == 3:
        return 3
    elif count == 4:
        return 4
    elif count == 5:
        return 5
    else:
        return 6


# 创建一个新列 'category' 来存储分类结果
data['category'] = data.apply(classify_sequence, axis=1)

# 保存包含分类结果的CSV文件
data.to_csv(r'D:\Coding\P1\aggregating peptides\7-10peptidde\7-10aggpeptide_fenlei.csv', index=False)
