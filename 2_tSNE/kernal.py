# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 23:25:26 2025

@author: maxim
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 读取数据
df = pd.read_csv("generated_molecules_with_SAscore.csv", encoding='GB18030')  # 修改为你的文件名
smiles_list = df['SMILES'].tolist()

# 2. 生成分子指纹
mols = []
valid_smiles = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mols.append(mol)
        valid_smiles.append(smiles)

# 生成Morgan指纹（Circular Fingerprint）
morgan_generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
fingerprints = [morgan_generator.GetFingerprint(mol) for mol in mols]

# 转换为numpy数组
np_fps = []
for fp in fingerprints:
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    np_fps.append(arr)
np_fps = np.array(np_fps)


# # PCA降维
# pca = PCA(n_components=2)
# components = pca.fit_transform(np_fps)

# # 将components转换为DataFrame
# components_df = pd.DataFrame(components, columns=['PC1', 'PC2'])

# # 合并到原始DataFrame
# df_combined = pd.concat([df, components_df], axis=1)

# # 保存合并后的DataFrame为新的CSV文件
# df_combined.to_csv("generated_molecules_with_SAscore_with_PCA.csv", index=False)

# # 可视化
# plt.figure(figsize=(10, 8))
# plt.scatter(components[:, 0], components[:, 1], alpha=0.6, edgecolors='w', s=40)
# plt.xlabel('Principal Component 1', fontsize=12)
# plt.ylabel('Principal Component 2', fontsize=12)
# plt.title('Molecular Similarity Visualization (PCA)', fontsize=14)
# plt.grid(alpha=0.3)
# plt.show()


from sklearn.manifold import TSNE

# t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
components_tsne = tsne.fit_transform(np_fps)

# 将components转换为DataFrame
components_df = pd.DataFrame(components_tsne, columns=['PC1', 'PC2'])

# 合并到原始DataFrame
df_combined = pd.concat([df, components_df], axis=1)

# 保存合并后的DataFrame为新的CSV文件
df_combined.to_csv("generated_molecules_with_SAscore_with_tSNE.csv", encoding='GB18030', index=False)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(components_tsne[:, 0], components_tsne[:, 1], alpha=0.6, edgecolors='w', s=40)
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)
plt.title('Molecular Similarity Visualization (t-SNE)', fontsize=14)
plt.grid(alpha=0.3)
plt.show()


# import umap

# # UMAP 降维
# umap_reducer = umap.UMAP(random_state=42)
# components_umap = umap_reducer.fit_transform(np_fps)

# # 将components转换为DataFrame
# components_df = pd.DataFrame(components_umap, columns=['PC1', 'PC2'])

# # 合并到原始DataFrame
# df_combined = pd.concat([df, components_df], axis=1)

# # 保存合并后的DataFrame为新的CSV文件
# df_combined.to_csv("generated_molecules_with_SAscore_with_UMAP.csv", index=False)

# # 可视化
# plt.figure(figsize=(10, 8))
# plt.scatter(components_umap[:, 0], components_umap[:, 1], alpha=0.6, edgecolors='w', s=40)
# plt.xlabel('UMAP 1', fontsize=12)
# plt.ylabel('UMAP 2', fontsize=12)
# plt.title('Molecular Similarity Visualization (UMAP)', fontsize=14)
# plt.grid(alpha=0.3)
# plt.show()
