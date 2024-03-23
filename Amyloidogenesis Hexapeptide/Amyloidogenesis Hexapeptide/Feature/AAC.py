import numpy as np
import pandas as pd
from collections import Counter
import re


def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    # AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = []
    for i in AA:
        header.append(i)
    # encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key] / len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def read_fasta(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    fasta_data = []
    current_sequence = ""
    current_name = ""

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            # Start of a new sequence
            if current_name and current_sequence:
                fasta_data.append((current_name, current_sequence))
            current_name = line[1:]
            current_sequence = ""
        else:
            # Append sequence data
            current_sequence += line

    # Add the last sequence
    if current_name and current_sequence:
        fasta_data.append((current_name, current_sequence))

    return fasta_data
def main():
    # 读取FASTA格式文件
    fasta_data = read_fasta(r"/aggregating peptides/11-20peptide/11-20aggpeptide.fasta")

    # 使用AAC特征提取函数
    aac_features, aac_header = AAC(fasta_data)
    # 这将提取AAC（氨基酸组成）特征
    aac_df = pd.DataFrame(aac_features, columns=aac_header)

    # 保存合并后的特征为CSV文件
    aac_df.to_csv(r"D:\Coding\P1\aggregating peptides\11-20peptide\11-20aggpeptide_AAC.csv", index=False)


if __name__ == "__main__":
    main()