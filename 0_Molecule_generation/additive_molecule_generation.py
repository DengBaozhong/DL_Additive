# -*- coding: utf-8 -*-
"""
基于 BRICS 连接点的分子生成器。
从核心片段库 (Cores.csv) 和取代基片段库 (Substituents.csv) 中组合生成有机分子。

系统枚举取代基全组合，每个组合只尝试一次，零浪费。

@author: maxim
@date: 2025-02-25
@updated: 2026-05-27 — 精简为纯 systematic 模式
"""

import itertools
import multiprocessing as mp
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════
#  ██████  用户配置区 — 请在此修改参数  ██████
# ═══════════════════════════════════════════════════════════════════════

# --- 并行参数 ---
NUM_WORKERS: Optional[int] = None   # None=自动检测 CPU 核心数；手动设置如 4

# --- 枚举参数 ---
EXHAUSTIVE_LIMIT: int = 10000    # 组合数 ≤ 此值时全枚举；超过则随机采样
SAMPLING_BUDGET: int = 2000     # 随机采样时的尝试次数
RANDOM_SEED: int = 42             # 随机种子

# --- 输入输出文件 ---
CORES_CSV: str = "Cores.csv"
SUBSTITUENTS_CSV: str = "Substituents.csv"
OUTPUT_CSV: str = "generated_molecules.csv"

# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════════

def get_anchor_indices(mol: Chem.Mol) -> List[int]:
    """返回分子中所有连接点 ``*`` 的原子索引列表."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']


def _anchor_neighbor_and_bond(
    mol: Chem.Mol, anchor_idx: int
) -> Optional[Tuple[Chem.Atom, Chem.BondType]]:
    """获取指定连接点的 (相邻原子, 键类型)；失败返回 None."""
    atom = mol.GetAtomWithIdx(anchor_idx)
    neighbors = atom.GetNeighbors()
    if not neighbors:
        return None
    bonded = neighbors[0]
    bond = mol.GetBondBetweenAtoms(anchor_idx, bonded.GetIdx())
    return bonded, bond.GetBondType()


def connect_fragments(
    frag1: Chem.Mol,
    frag2: Chem.Mol,
    anchor1_idx: int,
    anchor2_idx: int,
) -> Optional[Chem.Mol]:
    """通过指定连接点拼接两个分子片段。成功返回新分子，失败返回 None."""
    info1 = _anchor_neighbor_and_bond(frag1, anchor1_idx)
    info2 = _anchor_neighbor_and_bond(frag2, anchor2_idx)
    if info1 is None or info2 is None:
        return None

    a1, bt1 = info1
    a2, _ = info2  # 键类型由 frag1 的连接点决定

    n1 = frag1.GetNumAtoms()
    combined = Chem.RWMol(Chem.CombineMols(frag1, frag2))
    combined.AddBond(a1.GetIdx(), a2.GetIdx() + n1, bt1)

    # 先移除索引较大的连接点，避免索引偏移
    idx_high = anchor2_idx + n1
    idx_low = anchor1_idx
    if idx_low > idx_high:
        idx_high, idx_low = idx_low, idx_high
    combined.RemoveAtom(idx_high)
    combined.RemoveAtom(idx_low)

    try:
        mol = combined.GetMol()
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def connect_fragments_raw(
    frag1: Chem.Mol,
    frag2: Chem.Mol,
    anchor1_idx: int,
    anchor2_idx: int,
) -> Optional[Chem.RWMol]:
    """拼接两个片段，返回未 sanitize 的 RWMol（中间步骤用，省去反复 sanitize 开销）."""
    info1 = _anchor_neighbor_and_bond(frag1, anchor1_idx)
    info2 = _anchor_neighbor_and_bond(frag2, anchor2_idx)
    if info1 is None or info2 is None:
        return None

    a1, bt1 = info1
    a2, _ = info2

    n1 = frag1.GetNumAtoms()
    combined = Chem.RWMol(Chem.CombineMols(frag1, frag2))
    combined.AddBond(a1.GetIdx(), a2.GetIdx() + n1, bt1)

    idx_high = anchor2_idx + n1
    idx_low = anchor1_idx
    if idx_low > idx_high:
        idx_high, idx_low = idx_low, idx_high
    combined.RemoveAtom(idx_high)
    combined.RemoveAtom(idx_low)

    return combined


# ═══════════════════════════════════════════════════════════════════════
#  取代基组合生成（全枚举 / 随机采样）
# ═══════════════════════════════════════════════════════════════════════

def _generate_systematic(
    core_types: List[str],
    core_groups: Dict[str, List[int]],
    core_mols: np.ndarray,
    core_anchor_lists: List[List[int]],
    sub_mols: np.ndarray,
    sub_anchor_lists: List[List[int]],
    sub_types: List[str],
    core_type_en_map: Dict[str, str],
    exhaustive_limit: int,
    sampling_budget: int,
    seed: int,
    process_id: int,
) -> Dict[str, Dict[str, str]]:
    """
    对每个核心，生成取代基组合并尝试连接。

    - 组合数 ≤ exhaustive_limit：全枚举（itertools.product，字典序）
    - 组合数 > exhaustive_limit：随机采样 sampling_budget 个不重复组合

    Returns
    -------
    {core_type: {SMILES: remark}}
    """
    local_results: Dict[str, Dict[str, str]] = {ct: {} for ct in core_types}

    _connect_raw = connect_fragments_raw
    _get_anchors = get_anchor_indices
    _to_smiles = Chem.MolToSmiles
    _n_subs = len(sub_mols)

    for ct in core_types:
        ct_total_tried = 0
        for core_idx in core_groups[ct]:
            core = core_mols[core_idx]
            core_anchors = sorted(core_anchor_lists[core_idx])
            K = len(core_anchors)

            if K == 0:
                try:
                    smiles = _to_smiles(core)
                except Exception:
                    continue
                local_results[ct][smiles] = core_type_en_map.get(ct, ct)
                continue

            total_combos = _n_subs ** K

            if total_combos <= exhaustive_limit:
                # ── 全枚举 ──
                combos = list(itertools.product(range(_n_subs), repeat=K))
                desc_mode = "枚举"
            else:
                # ── 随机采样 ──
                rng = np.random.RandomState(
                    seed * 10000 + process_id * 1000 + core_idx
                )
                budget = min(sampling_budget, total_combos)
                seen: set = set()
                while len(seen) < budget:
                    seen.add(tuple(rng.randint(0, _n_subs, size=K)))
                combos = list(seen)
                desc_mode = "采样"

            for sub_tuple in tqdm(
                combos,
                desc=f"p{process_id:02d}/{ct}({desc_mode})",
                position=process_id,
                leave=False,
                mininterval=0.5,
            ):
                used_sub = list(sub_tuple)

                # ── 依次连接取代基（RWMol 裸连，不 sanitize）──
                mol = Chem.RWMol(core)
                ok = True
                for sub_idx in sub_tuple:
                    anchors = _get_anchors(mol)
                    if not anchors:
                        ok = False
                        break
                    mol = _connect_raw(
                        mol, sub_mols[sub_idx],
                        anchors[0],
                        sub_anchor_lists[sub_idx][0],
                    )
                    if mol is None:
                        ok = False
                        break

                if not ok:
                    continue

                # ── 最终只 sanitize 一次，然后生成 SMILES ──
                try:
                    mol_final = mol.GetMol()
                    mol_final = Chem.RemoveHs(mol_final)
                    Chem.SanitizeMol(mol_final)
                    smiles = _to_smiles(mol_final)
                except Exception:
                    continue

                if smiles not in local_results[ct]:
                    core_en = core_type_en_map.get(ct, ct)
                    remark = "+".join([core_en] + [sub_types[i] for i in used_sub])
                    local_results[ct][smiles] = remark

            ct_total_tried += len(combos)

        # ── 每个 core_type 完成后的统计 ──
        gen_n = len(local_results[ct])
        if ct_total_tried > 0:
            print(f"  [{ct}] 尝试 {ct_total_tried:,} → 唯一 {gen_n:,} "
                  f"(效率 {gen_n/ct_total_tried*100:.1f}%)")

    return local_results


# ═══════════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════════

def _load_fragments(
    script_dir: str,
) -> Tuple[
    np.ndarray,                     # core_mols
    List[str],                      # core_types_list
    Dict[str, str],                 # core_type_en_map
    Dict[str, List[int]],           # core_groups
    List[List[int]],                # core_anchor_lists
    np.ndarray,                     # sub_mols
    List[str],                      # sub_types
    List[List[int]],                # sub_anchor_lists
]:
    """加载核心与取代基片段，返回预计算数据结构."""
    cores_path = os.path.join(script_dir, CORES_CSV)
    subs_path = os.path.join(script_dir, SUBSTITUENTS_CSV)

    df_cores = pd.read_csv(cores_path)
    df_subs = pd.read_csv(subs_path)
    print(f"核心片段: {len(df_cores)} 条  ({cores_path})")
    print(f"取代基:   {len(df_subs)} 条  ({subs_path})")

    # 解析核心
    core_mols_list: List[Chem.Mol] = []
    core_types_list: List[str] = []
    for smi, ct in zip(df_cores["Core_SMILES"], df_cores["Core_type"]):
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            core_mols_list.append(mol)
            core_types_list.append(ct)

    # 解析取代基
    sub_data = [
        (Chem.MolFromSmiles(smi), st)
        for smi, st in zip(df_subs["Substituent_SMILES"], df_subs["Substituent_type"])
        if smi and not pd.isna(smi)
    ]

    core_mols_arr = np.array(core_mols_list)
    sub_mols_arr = np.array([m for m, _ in sub_data])
    sub_types_list = [st for _, st in sub_data]
    core_type_en_map = dict(zip(df_cores["Core_type"], df_cores["Core_type_en"]))

    # 预计算连接点索引
    core_anchor_lists = [get_anchor_indices(m) for m in core_mols_arr]
    sub_anchor_lists = [get_anchor_indices(m) for m in sub_mols_arr]

    # 按核心类型分组（预构建索引列表）
    core_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, ct in enumerate(core_types_list):
        core_groups[ct].append(idx)

    return (
        core_mols_arr, core_types_list, core_type_en_map, core_groups,
        core_anchor_lists, sub_mols_arr, sub_types_list, sub_anchor_lists,
    )


# ═══════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── 加载数据 ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    (
        core_mols, core_types_list, core_type_en_map, core_groups,
        core_anchor_lists, sub_mols, sub_types, sub_anchor_lists,
    ) = _load_fragments(script_dir)

    # ── 并行参数 ──
    num_workers = min(
        NUM_WORKERS or mp.cpu_count(),
        len(core_groups),
    )

    print(f"\n{'='*50}")
    print(f"  分子生成 — 系统枚举模式")
    print(f"  进程数:           {num_workers}")

    # 将核心类型均匀分配到各进程
    all_ct = sorted(core_groups.keys())  # 排序确保可重复
    chunk_size = max(1, len(all_ct) // num_workers)
    chunks = [
        all_ct[i : i + chunk_size]
        for i in range(0, len(all_ct), chunk_size)
    ]

    # ── 枚举 / 采样 ──
    print(f"  全枚举上限:       {EXHAUSTIVE_LIMIT:,} 组合")
    print(f"  采样预算:         {SAMPLING_BUDGET:,} 组合")
    print(f"  随机种子:         {RANDOM_SEED}")
    print(f"{'='*50}\n")

    with mp.Pool(processes=num_workers) as pool:
        args = [
            (
                chunk, core_groups, core_mols, core_anchor_lists,
                sub_mols, sub_anchor_lists, sub_types, core_type_en_map,
                EXHAUSTIVE_LIMIT, SAMPLING_BUDGET,
                RANDOM_SEED, i,
            )
            for i, chunk in enumerate(chunks)
        ]
        all_results = pool.starmap(_generate_systematic, args)

    # ── 合并 & 保存 ──
    merged: Dict[str, Dict[str, str]] = {ct: {} for ct in core_groups}
    for result in all_results:
        for ct, d in result.items():
            merged[ct].update(d)

    total = sum(len(v) for v in merged.values())
    print(f"\n✅ 分子生成完成！共 {total:,} 个唯一分子")

    rows = [
        {"SMILES": smi, "remark": remark}
        for ct_dict in merged.values()
        for smi, remark in ct_dict.items()
    ]
    output_path = os.path.join(script_dir, OUTPUT_CSV)
    pd.DataFrame(rows).to_csv(output_path, encoding="utf-8-sig", index=False)
    print(f"📁 结果 → {output_path}")


if __name__ == "__main__":
    main()

