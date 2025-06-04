# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:04:45 2025

@author: maxim
"""

import pandas as pd
from rdkit import Chem
import sascorer  # 需确保 sascorer.py 在当前目录

# 读取CSV文件
df = pd.read_csv('generated_molecules.csv', encoding='GB18030')

# 计算SAscore
def calculate_sascore(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:  # 确保 SMILES 能正确转换为分子
        return sascorer.calculateScore(mol)
    return None  # 处理无效SMILES的情况

# 应用计算
df['SAscore'] = df['SMILES'].apply(calculate_sascore)

# 保存新的CSV文件
df.to_csv('generated_molecules_with_SAscore.csv', encoding='GB18030', index=False)

# 显示前几行数据
print(df.head())
