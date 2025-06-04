# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:09:08 2025

@author: maxim
"""

from pandas import DataFrame
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.ML.Descriptors import MoleculeDescriptors
import random
from collections import defaultdict

data = pd.read_csv(r"Molecules.csv", encoding='GB18030').fillna('')
cores = data['Core_SMILES'].replace('', None)
core_types = data['Core_type'].replace('', None)
subs = data['Substituent_SMILES'].replace('', None)

# mols_cores_list = np.array([Chem.MolFromSmiles(mol) for mol in cores if pd.notna(mol) and mol])
mols_cores_list = []
core_types_list = []
for core_smiles, core_type in zip(cores, core_types):
    if pd.notna(core_smiles) and core_smiles:
        mol = Chem.MolFromSmiles(core_smiles)
        if mol is not None:
            mols_cores_list.append(mol)
            core_types_list.append(core_type)
mols_cores_list = np.array(mols_cores_list)  # 转为数组方便后续操作

mols_sub_list = np.array([Chem.MolFromSmiles(mol) for mol in subs if pd.notna(mol) and mol])


def connect_fragments(frag1, frag2):
    """自动连接两个片段（均需包含连接点*）"""
    # 预处理输入分子
    frag1 = Chem.Mol(frag1)
    frag2 = Chem.Mol(frag2)
    
    # 获取连接点信息
    def get_anchor_info(mol):
        dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
        if not dummy_atoms:
            raise ValueError("片段缺少连接点*")
        # 总是选择第一个连接点
        dummy = dummy_atoms[0]
        neighbors = dummy.GetNeighbors()
        if not neighbors:
            raise ValueError("连接点未键接原子")
        bonded_atom = neighbors[0]
        bond_type = mol.GetBondBetweenAtoms(dummy.GetIdx(), bonded_atom.GetIdx()).GetBondType()
        return bonded_atom, bond_type, dummy.GetIdx()

    # 获取两个片段的连接信息
    frag1_atom, frag1_bond_type, frag1_dummy = get_anchor_info(frag1)
    frag2_atom, frag2_bond_type, frag2_dummy = get_anchor_info(frag2)

    # 构建新分子
    combined = Chem.RWMol(Chem.CombineMols(frag1, frag2))
    
    # 添加新键（使用第一个片段的键类型）
    new_bond_idx = combined.AddBond(
        frag1_atom.GetIdx(), 
        frag2_atom.GetIdx() + frag1.GetNumAtoms(), 
        frag1_bond_type
    )
    
    # 移除连接点原子
    combined.RemoveAtom(frag2_dummy + frag1.GetNumAtoms())  # 先移除后面的原子
    combined.RemoveAtom(frag1_dummy)

    # 后处理
    try:
        combined = combined.GetMol()
        combined = Chem.RemoveHs(combined)
        Chem.SanitizeMol(combined)
        # 验证剩余连接点
        if Chem.MolToSmiles(combined).count('*') == (Chem.MolToSmiles(frag1).count('*') + Chem.MolToSmiles(frag2).count('*') - 2):
            return combined
    except Exception as e:
        raise ValueError(f"连接失败: {str(e)}")


# ============== 生成分子进度 ==============
# 按core_type分组核心索引
core_groups = defaultdict(list)
for idx, ct in enumerate(core_types_list):
    core_groups[ct].append(idx)

# 设置每个core_type的目标生成数量
n_per_core_type = 500  # 每个类型生成1000个
core_type_targets = {ct: n_per_core_type for ct in core_groups.keys()}
total_target = sum(core_type_targets.values())

core_type_smiles = {ct: set() for ct in core_type_targets}  # 去重用
core_type_counts = {ct: 0 for ct in core_type_targets}      # 计数用

max_attempts = total_target * 2  # 最大尝试次数
attempts = 0

print(f"\n开始生成分子，每个Core_type目标数量: {n_per_core_type}")

while True:
    remaining_cts = [ct for ct in core_type_targets if core_type_counts[ct] < core_type_targets[ct]]
    if not remaining_cts:
        break  # 所有类型均完成
    if attempts >= max_attempts:
        print("\n达到最大尝试次数，提前终止。")
        break
    attempts += 1
    try:
        # 选择一个未完成的核心类型
        selected_ct = random.choice(remaining_cts)
        # 从该类型中随机选一个核心
        core_idx = random.choice(core_groups[selected_ct])
        core = mols_cores_list[core_idx]
        
        # 连接取代基
        sub = random.choice(mols_sub_list)
        if '*' in Chem.MolToSmiles(core):
            intermediate = connect_fragments(core, sub)
        else:
            intermediate = core
        
        # 确保无剩余连接点
        while '*' in Chem.MolToSmiles(intermediate):
            sub = random.choice(mols_sub_list)
            intermediate = connect_fragments(intermediate, sub)
        
        smile = Chem.MolToSmiles(intermediate).replace('*', '')
        if '*' not in smile:
            if smile not in core_type_smiles[selected_ct]:
                core_type_smiles[selected_ct].add(smile)
                core_type_counts[selected_ct] += 1
                total = sum(core_type_counts.values())
                progress = total / total_target * 100
                print(f"\r生成进度: {total}/{total_target} ({progress:.1f}%)", end="", flush=True)
                
    except Exception as e:
        continue

print("\n分子生成完成！")

# 整理结果并保存
smiles_list = []
for ct in core_type_smiles:
    for smile in core_type_smiles[ct]:
        smiles_list.append({"SMILES": smile, "Core_type": ct})

df = DataFrame(smiles_list)
df.to_csv(r"generated_molecules.csv", encoding='GB18030',index=False)
