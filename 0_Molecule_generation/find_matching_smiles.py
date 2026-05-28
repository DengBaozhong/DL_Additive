import os
import pandas as pd
from rdkit import Chem

def find_matching_smiles(csv_file, target_smiles, strict_csv_canonical=True):
    """
    在CSV文件的SMILES列中查找与目标SMILES代表同一分子的SMILES（优化版，适用于大文件）
    
    优化: 假设CSV中的SMILES已是RDKit canonical标准形式，直接字符串比较。
    如果CSV非标准，设strict_csv_canonical=False，回退到逐行Mol转换（较慢）。
    
    参数:
    csv_file: CSV文件路径
    target_smiles: 目标SMILES字符串
    strict_csv_canonical: 是否假设CSV已是canonical (默认True，速度快)
    
    返回:
    list: 匹配的索引列表 [int], 或空列表
    """
    matches_indices = []
    try:
        # 读取CSV文件，只取SMILES列以节省内存
        df = pd.read_csv(csv_file, usecols=['SMILES'])
        print(f"CSV文件加载成功，共 {len(df)} 行数据。")
        
        # 检查是否存在SMILES列（usecols已确保）
        if df.empty:
            print("错误: CSV文件为空或无法读取SMILES列")
            return matches_indices
        
        # 将目标SMILES转换为规范的mol对象
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            print(f"错误: 目标SMILES '{target_smiles}' 无效")
            return matches_indices
        
        # 生成目标的canonical SMILES
        target_canonical_smiles = Chem.MolToSmiles(target_mol, canonical=True)
        print(f"目标分子的规范SMILES: {target_canonical_smiles}")
        
        # 移除NaN值
        df = df.dropna(subset=['SMILES'])
        
        if strict_csv_canonical:
            # 策略1: 直接字符串比较（假设CSV已是canonical，超快）
            matching_rows = df[df['SMILES'] == target_canonical_smiles]
            if not matching_rows.empty:
                matches_indices = matching_rows.index.tolist()
                print(f"\n找到 {len(matches_indices)} 个匹配的分子:")
            else:
                print("\n未找到匹配的分子")
                return matches_indices
        else:
            # 策略2: 逐行Mol转换比较（如果CSV非标准，回退此模式，较慢）
            print("使用逐行Mol转换模式（适用于非标准CSV）...")
            for index, row in df.iterrows():
                smiles = row['SMILES']
                try:
                    mol = Chem.MolFromSmiles(str(smiles))
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                        if canonical_smiles == target_canonical_smiles:
                            matches_indices.append(index)
                except Exception as e:
                    print(f"警告: 处理SMILES '{smiles}' 时出错: {e}")
                    continue
            
            if matches_indices:
                print(f"\n找到 {len(matches_indices)} 个匹配的分子:")
            else:
                print("\n未找到匹配的分子")
                return matches_indices
        
        # 输出匹配的行号（index）
        for idx in matches_indices:
            print(f"匹配行号 (index): {idx}")
            
    except FileNotFoundError:
        print(f"错误: 文件 '{csv_file}' 未找到")
    except Exception as e:
        print(f"错误: {e}")
    
    return matches_indices

if __name__ == "__main__":
    # 目标SMILES（任意写法，例如BrC1=CC=CC=C1 或 c1ccccc1 都对应苯）
    target_smiles = "FC1=CC(Cl)=C(Cl)C(Cl)=C1"  # 您的例子，或任意输入
    
    # CSV文件路径（基于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "generated_molecules.csv")
    
    print(f"正在查找与 '{target_smiles}' 相同的分子...")
    matches_indices = find_matching_smiles(csv_file, target_smiles, strict_csv_canonical=True)
