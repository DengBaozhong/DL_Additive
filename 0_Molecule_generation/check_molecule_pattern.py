# -*- coding: utf-8 -*-
"""
================================================================================
  分子模式判断脚本
  check_molecule_pattern.py
================================================================================

0. 目的与背景
──────────────────────────────────────────────────────────────────────
本脚本用于对分子生成任务的输出 CSV 进行自动化模式验证,判断
生成结果是否收敛到预期的"最优分子模式"。验证采用逐级收窄的漏斗策略: 从全量
数据出发,每一步只在上一步通过的行子集上继续筛选,最终落到一个极小的候选
集合(通常是几十到几百行),在候选集合中检查目标分子是否存在。

预期模式(自顶向下): 
  - 全局 PCE 排名前 5% 的分子中,benzene(苯环)核心骨架占绝对主导
  - benzene 子集中,fluorine(氟)取代基平均 PCE 最高,chlorine(氯)次之
  - 同时含 F+Cl 的 benzene 分子中,4-取代(四取代苯环)平均 PCE 最高
  - 最终候选集里含有目标分子 Fc1cc(Cl)c(Cl)c(Cl)c1(1-F,3,4,5-三氯苯)

================================================================================

1. 数据格式
──────────────────────────────────────────────────────────────────────
输入 CSV: generated_molecules_test.csv

  ┌──────────┬─────────────────────────────────────────────────┬───────┐
  │ SMILES   │ remark                                          │ PCE   │
  ├──────────┼─────────────────────────────────────────────────┼───────┤
  │ Fc1cc... │ core_benzene_4_substitute+substituent_fluorine+ │ 21.0  │
  │          │ substituent_chlorine                            │       │
  └──────────┴─────────────────────────────────────────────────┴───────┘

remark 标签命名规则: 
  - core_XXXX_N_substitute: 核心骨架 = XXXX,取代位点数 = N
    例如 core_benzene_4_substitute 表示苯环上 4 个位点被取代
  - substituent_YYYY: 取代基类型(fluorine / chlorine / bromine / ...)

标签之间用 "+" 分隔,单行可含多个标签(多取代基分子)。

================================================================================

2. 五步漏斗逻辑(核心算法)
──────────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────┐
  │ 第1步: PCE 降序截断
  │   输入: 全量 ~58 万行
  │   操作: 按 PCE 降序排列,取前 top_ratio(默认 5%) 
  │   输出: ~29,000 行(top_ratio=0.05 时)
  │   依赖: PCE 列(float)
  └───────────────┬─────────────────────────────────────────────┘
                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 第2步: 核心骨架平均PCE统计
  │   输入: 第1步的 top 子集
  │   操作: 按 core_XXX 分组,计算各组的平均 PCE,判断
  │         core_benzene* 的平均 PCE 是否排名第一
  │   判断: core_benzene* 是否为平均PCE最高的 core? 
  │   不满足 → return False(或跳过,取决于 stop_on_fail)
  └───────────────┬─────────────────────────────────────────────┘
                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 第3步: 取代基平均PCE统计(仅 core_benzene 子集)
  │   输入: 第2步中所有含 core_benzene* 标签的行
  │   操作: 按 substituent_XXX 分组,计算各取代基的平均 PCE
  │   判断: substituent_fluorine 平均PCE 排第1 且
  │         substituent_chlorine 平均PCE 排第2?
  │   不满足 → return False
  └───────────────┬─────────────────────────────────────────────┘
                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 第4步: N 平均PCE统计(F+Cl 共存子集)
  │   输入: 第3步子集中同时含 fluorine + chlorine 标签的行
  │   操作: 提取 core_benzene_N_substitute 中的 N 值,按 N 分组
  │         计算各 N 值的平均 PCE
  │   判断: N=4(即 4_substitute)是否为平均PCE最高的 N 值?
  │   不满足 → return False
  └───────────────┬─────────────────────────────────────────────┘
                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 第5步: 目标 SMILES 存在性检查
  │   输入: 第4步子集(core_benzene + F + Cl 的行)
  │   目标: Fc1cc(Cl)c(Cl)c(Cl)c1(1-氟-3,4,5-三氯苯)
  │   方法: 两阶段匹配(详见第3节)
  │   判断: 目标分子是否存在于该子集中?
  │   存在 → return True  /  不存在 → return False
  └─────────────────────────────────────────────────────────────┘

================================================================================

3. 第5步技术细节: SMILES 等价性判断
──────────────────────────────────────────────────────────────────────
问题: 同一分子可有多种 SMILES 写法(如芳香键的不同表示、原子顺序不同等)。
      纯字符串 == 比较会漏检等价分子。

  Fc1cc(Cl)c(Cl)c(Cl)c1   ←→   FC1=CC(Cl)=C(Cl)C(Cl)=C1
  (芳香式写法,氯用 Cl 表示)     (凯库勒式写法,显式写出双键)
  这两种写法指的是同一个分子,但字符串不同。

方案: 分两阶段匹配,兼顾速度与准确性。

  ┌────────────────────────────────────────────────────────────┐
  │ 阶段 A: 字符级快速预筛(无 RDKit,纯 Python 字符串操作)
  │
  │   统计目标分子字符串中的原子/元素计数: 
  │     - F  出现次数(target_raw.count("F"))
  │     - Cl 出现次数(target_raw.count("Cl"))
  │
  │   注意: 不预筛 C 原子数,因为 SMILES 中 Cl 内的 "C" 与
  │   芳香碳 "c" 无法通过字符级计数区分;且第2步已确保苯核心,
  │   碳骨架无需额外验证。
  │
  │   候选 SMILES 只有 F 和 Cl 计数同时匹配才进入阶段 B。
  │   这一步可在 O(n) 内排除绝大多数不相关分子,避免昂贵的
  │   RDKit 解析(MorFromSmiles 是 CPU 密集型操作)。
  └────────────────────────────────────────────────────────────┘
                  ▼
  ┌────────────────────────────────────────────────────────────┐
  │ 阶段 B: RDKit 规范化等价判断
  │
  │   1. 对目标 SMILES 调用 Chem.MolFromSmiles 构建分子对象
  │   2. 调用 Chem.MolToSmiles(..., canonical=True) 生成
  │      标准 SMILES(Canonical SMILES)
  │   3. 对候选 SMILES 同样构建分子 + 生成 Canonical SMILES
  │   4. 比较两个 Canonical SMILES 字符串是否相等
  │
  │   如果 RDKit 无法解析目标 SMILES,则回退到纯字符串 == 比较,
  │   并打印警告信息。
  └────────────────────────────────────────────────────────────┘

  依赖: from rdkit import Chem
  注意: RDKit 的 canonical SMILES 不保证跨版本绝对一致,但在同一
        RDKit 版本内是确定性的,因此同一次运行中比较是可靠的。

================================================================================

4. 性能考量
──────────────────────────────────────────────────────────────────────
  - 58 万行全量读取: 使用 csv.DictReader 逐行处理,内存友好
  - 所有 core/substituent/N 的筛选仅对 remark 列做纯字符串操作
    (split / startswith / in),不涉及 RDKit
  - 只有第5步的最终候选子集(通常几十到几百行)才调用 RDKit,
    且经字符预筛后实际解析次数更少
  - 总耗时预期: 数秒级别(不含绘图部分)

================================================================================

5. stop_on_fail 参数
──────────────────────────────────────────────────────────────────────
  stop_on_fail=True  (默认): 任一步不满足立即 return False,快速退出
  stop_on_fail=False         : 始终走完所有 5 步,每步不满足时仅标记
                               all_pass=False 但不提前退出。适用于需要
                               完整输出所有阶段诊断信息的场景

================================================================================

6. 边界情况处理
──────────────────────────────────────────────────────────────────────
  - CSV 为空: 第1步直接 return False
  - 无 core_ 标签: 第2步 return False
  - core_benzene 子集为空: 第3步 return False
  - substituent 种类不足 2 种: 第3步 return False
  - F+Cl 共存子集为空: 第4步 return False
  - N 值无法提取: 第4步 return False
  - 目标 SMILES 无法被 RDKit 解析: 第5步回退到字符串 == 比较

================================================================================

7. 使用示例
──────────────────────────────────────────────────────────────────────
  # 方式1: 直接运行脚本(需将 CSV 放在同目录)
  python check_molecule_pattern.py

  # 方式2: 作为模块导入
  from check_molecule_pattern import check_molecule_pattern
  
  result = check_molecule_pattern(
      "path/to/generated_molecules_test.csv",
      top_ratio=0.05,      # 取 PCE 前 5%
      stop_on_fail=False,  # 走完所有阶段以获取完整诊断
  )
  print(f"最终判断: {result}")

  # 方式3: 调整截取比例
  result = check_molecule_pattern("data.csv", top_ratio=0.10)  # 前 10%

================================================================================

@author: maxim
@date:   2026-05-28
@update: 2026-05-28 — 第3/4步从出现频率统计改为平均PCE排名;
         第5步 SMILES 比较升级为 RDKit Canonical SMILES + 字符预筛;
         第2步核心骨架判断从频率统计改为平均PCE排名
================================================================================
"""

import csv
import os

import matplotlib.pyplot as plt
import matplotlib
from rdkit import Chem

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════════════════════════════
#  核心函数
# ═══════════════════════════════════════════════════════════════════════


def check_molecule_pattern(csv_path: str, top_ratio: float = 0.05,
                          stop_on_fail: bool = True) -> bool:
    """
    对 generated_molecules_test.csv 执行逐级漏斗筛选统计。

    漏斗逻辑(每步输入为上一步通过的行子集,逐步收窄): 
      第1步: 全量数据按 PCE 降序,截取前 top_ratio
      第2步: top 子集中按 core_XXX 分组计算平均 PCE,判断 core_benzene* 是否排第一
      第3步: 仅 core_benzene 子集,按 substituent_XXX 分组计算平均 PCE,
             判断 substituent_fluorine 是否第1、substituent_chlorine 是否第2
      第4步: 仅 core_benzene 且同时含 substituent_fluorine+substituent_chlorine 的行,
             按 N 值分组计算平均 PCE,判断 4_substitute 是否平均PCE最高
      第5步: 最终漏斗子集中检查目标 SMILES 是否存在

    参数:
        csv_path     : CSV 文件路径
        top_ratio    : PCE 截取比例,默认 0.05(前5%)
        stop_on_fail : True 时任一步不满足即提前返回 False；
                       False 时始终走完所有阶段再返回最终结果

    返回:
        True   : 全部5个判断条件均满足
        False  : 任意一步不满足
    """
    # ── 读取全部数据 ──
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "SMILES": r["SMILES"].strip(),
                "remark": r["remark"].strip(),
                "Predicted_PCE": float(r["Predicted_PCE"]),
            })

    all_pass = True  # 追踪所有阶段是否通过

    if not rows:
        print("[第1步] CSV 为空,退出。")
        return False

    # ── 第1步: 按 PCE 降序,取前 top_ratio ──
    rows.sort(key=lambda x: x["Predicted_PCE"], reverse=True)
    cutoff = max(1, int(len(rows) * top_ratio))
    top_rows = rows[:cutoff]
    print(f"[第1步] 总行数={len(rows)}, top_ratio={top_ratio}, 截取={len(top_rows)} 行")

    # ── 第2步: 统计 core_XXX 平均 PCE ──
    core_pce_sum = {}
    core_pce_count = {}
    for r in top_rows:
        for tag in r["remark"].split("+"):
            tag = tag.strip()
            if tag.startswith("core_"):
                # 聚合基名: 去掉前缀 core_ 和后缀 _substitute，
                # 若存在尾随的数字（如 benzene_4），去掉该数字部分
                base = tag.replace("core_", "").replace("_substitute", "")
                if "_" in base and base.rsplit("_", 1)[1].isdigit():
                    base = base.rsplit("_", 1)[0]
                core_pce_sum[base] = core_pce_sum.get(base, 0.0) + r["Predicted_PCE"]
                core_pce_count[base] = core_pce_count.get(base, 0) + 1
                break  # 每行只取第一个 core_ 标签

    if not core_pce_count:
        print("[第2步] 未找到任何 core_ 标签, False")
        if stop_on_fail:
            return False
        all_pass = False
    else:
        # 计算各 core 的平均 PCE 并排序
        core_avg_pce = {k: core_pce_sum[k] / core_pce_count[k] for k in core_pce_count}
        core_avg_sorted = sorted(core_avg_pce.items(), key=lambda x: x[1], reverse=True)
        top_core, top_avg = core_avg_sorted[0]
        top3_display = [(k, round(v, 4), core_pce_count[k]) for k, v in core_avg_sorted[:3]]
        print(f"[第2步] core 平均PCE Top3 (avg, n): {top3_display}")
        if top_core != "benzene":
            print(f"[第2步] 平均PCE 第一名是 {top_core}(avg={top_avg:.4f}), 不是 benzene, False")
            if stop_on_fail:
                return False
            all_pass = False
        else:
            print(f"[第2步] ✓ benzene 平均PCE 排第一 (avg={top_avg:.4f}, n={core_pce_count['benzene']})")

    # ── 第3步(漏斗第3层): 仅 core_benzene 子集,计算各取代基平均 PCE ──
    benzene_rows = [r for r in top_rows
                    if any(t.startswith("core_benzene") for t in r["remark"].split("+"))]

    if not benzene_rows:
        print("[第3步] core_benzene 子集为空,False")
        if stop_on_fail:
            return False
        all_pass = False

    if benzene_rows:
        # 计算各取代基的平均 PCE（一行可含多个取代基标签,其 PCE 同时计入各取代基）
        sub_pce_sum = {}
        sub_pce_count = {}
        for r in benzene_rows:
            for tag in r["remark"].split("+"):
                tag = tag.strip()
                if tag.startswith("substituent_"):
                    sub_pce_sum[tag] = sub_pce_sum.get(tag, 0.0) + r["Predicted_PCE"]
                    sub_pce_count[tag] = sub_pce_count.get(tag, 0) + 1

        if len(sub_pce_count) < 2:
            print(f"[第3步] substituent 种类不足2种: {list(sub_pce_count.keys())}, False")
            if stop_on_fail:
                return False
            all_pass = False
        else:
            sub_avg_pce = {k: sub_pce_sum[k] / sub_pce_count[k] for k in sub_pce_count}
            sub_avg_sorted = sorted(sub_avg_pce.items(), key=lambda x: x[1], reverse=True)
            first_sub, first_avg = sub_avg_sorted[0]
            second_sub, second_avg = sub_avg_sorted[1] if len(sub_avg_sorted) > 1 else (None, 0)
            top5_display = [(k, round(v, 4), sub_pce_count[k]) for k, v in sub_avg_sorted[:5]]
            print(f"[第3步] substituent 平均PCE Top5 (avg, n): {top5_display}")
            if not (first_sub == "substituent_fluorine" and second_sub == "substituent_chlorine"):
                print(f"[第3步] 平均PCE 第1={first_sub}(avg={first_avg:.4f}), "
                      f"第2={second_sub}(avg={second_avg:.4f}), 不满足 fluorine/chlorine 序, False")
                if stop_on_fail:
                    return False
                all_pass = False
            else:
                print(f"[第3步] ✓ fluorine 平均PCE 第1 (avg={first_avg:.4f}), "
                      f"chlorine 第2 (avg={second_avg:.4f})")
    else:
        sub_avg_pce = {}

    # ── 第4步(漏斗第4层): 仅 core_benzene 且同时含
    #    substituent_fluorine + substituent_chlorine 的行,统计 N 分布 ──
    fcl_rows = []
    for r in benzene_rows:
        tags = r["remark"].split("+")
        has_f = any(t.strip() == "substituent_fluorine" for t in tags)
        has_cl = any(t.strip() == "substituent_chlorine" for t in tags)
        if has_f and has_cl:
            fcl_rows.append(r)

    if not fcl_rows:
        print("[第4步] 同时含 fluorine+chlorine 的子集为空,False")
        if stop_on_fail:
            return False
        all_pass = False

    if fcl_rows:
        # 计算各 N 值的平均 PCE
        n_pce_sum = {}
        n_pce_count = {}
        for r in fcl_rows:
            for tag in r["remark"].split("+"):
                tag = tag.strip()
                if tag.startswith("core_benzene_") and tag.endswith("_substitute"):
                    # 提取 N,如 core_benzene_4_substitute → 4
                    try:
                        n_val = int(tag.replace("core_benzene_", "").replace("_substitute", ""))
                        n_pce_sum[n_val] = n_pce_sum.get(n_val, 0.0) + r["Predicted_PCE"]
                        n_pce_count[n_val] = n_pce_count.get(n_val, 0) + 1
                    except ValueError:
                        pass

        if not n_pce_count:
            print("[第4步] 未能提取 N 值,False")
            if stop_on_fail:
                return False
            all_pass = False
        else:
            n_avg_pce = {k: n_pce_sum[k] / n_pce_count[k] for k in n_pce_count}
            n_avg_sorted = sorted(n_avg_pce.items(), key=lambda x: x[1], reverse=True)
            top_n, top_n_avg = n_avg_sorted[0]
            display = [(k, round(v, 4), n_pce_count[k]) for k, v in n_avg_sorted]
            print(f"[第4步] N 平均PCE (N, avg, n): {display}")
            if top_n != 4:
                print(f"[第4步] 平均PCE 最高的是 N={top_n}(avg={top_n_avg:.4f}), 不是 N=4, False")
                if stop_on_fail:
                    return False
                all_pass = False
            else:
                print(f"[第4步] ✓ N=4 平均PCE 最高 (avg={top_n_avg:.4f}, n={n_pce_count[4]})")
    else:
        n_avg_pce = {}

    # ── 第5步(漏斗最底层): 最终子集中检查目标 SMILES ──
    # 使用 RDKit 规范化比较,避免不同 SMILES 写法导致漏检。
    # 优化: 先按字符级 F/Cl 原子数快速预筛,再走 RDKit 解析。
    target_raw = "Fc1cc(Cl)c(Cl)c(Cl)c1"
    # 字符级预筛计数: 仅用 F 和 Cl 原子数（不含 C，因为 SMILES 中
    # Cl 内的 C 与芳香碳 c 无法通过字符级计数区分；且第2步已确保苯核心）
    target_f_count = target_raw.count("F")
    target_cl_count = target_raw.count("Cl")

    target_mol = Chem.MolFromSmiles(target_raw)
    if target_mol is None:
        print(f"[第5步] ⚠ 目标 SMILES '{target_raw}' 无法被 RDKit 解析,回退到精确字符串匹配")
        found = any(r["SMILES"] == target_raw for r in fcl_rows)
    else:
        target_canon = Chem.MolToSmiles(target_mol, canonical=True)

        found = False
        for r in fcl_rows:
            smi = r["SMILES"]
            # ── 第一关: 字符级快速过滤（只筛 F 和 Cl 原子数）──
            if smi.count("F") != target_f_count:
                continue
            if smi.count("Cl") != target_cl_count:
                continue
            # ── 第二关: RDKit 规范化比较 ──
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if Chem.MolToSmiles(mol, canonical=True) == target_canon:
                found = True
                break

    print(f"[第5步] 目标 SMILES = {target_raw}")
    print(f"[第5步] 预筛条件: F×{target_f_count}, Cl×{target_cl_count} (C 不参与预筛,苯核心已由第2步保证)")
    print(f"[第5步] {'✓ 找到' if found else '✗ 未找到'} 目标 SMILES (在 {len(fcl_rows)} 行的最终子集中)")
    if not found:
        all_pass = False
    return all_pass


# ═══════════════════════════════════════════════════════════════════════
#  绘图辅助函数
# ═══════════════════════════════════════════════════════════════════════


def _read_all(csv_path: str):
    """读取全部数据,返回 (rows, top_rows)"""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "SMILES": r["SMILES"].strip(),
                "remark": r["remark"].strip(),
                "Predicted_PCE": float(r["Predicted_PCE"]),
            })
    rows.sort(key=lambda x: x["Predicted_PCE"], reverse=True)
    cutoff = max(1, int(len(rows) * 0.05))
    return rows, rows[:cutoff]


def _plot_all_stages(core_avg_pce: dict, core_pce_count: dict,
                     sub_avg_pce: dict, n_avg_pce: dict,
                     found: bool, n_fcl: int, out_dir: str) -> None:
    """将阶段2~5的四个图合并为一个 2×2 子图 PNG"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    (ax2, ax3), (ax4, ax5) = axes

    # ── Stage 2: core 平均 PCE ──
    items2 = sorted(core_avg_pce.items(), key=lambda x: x[1], reverse=True)[:15]
    if items2:
        labels2, values2 = zip(*items2)
    else:
        labels2, values2 = (), ()
    labels2 = list(labels2)
    values2 = list(values2)
    colors2 = ["#d62728" if "benzene" in l else "#1f77b4" for l in labels2]
    bars2 = ax2.bar(range(len(labels2)), values2, color=colors2)
    ax2.set_xticks(range(len(labels2)))
    ax2.set_xticklabels(labels2, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Average PCE")
    ax2.set_title("Stage 2: Core Average PCE in Top 5% PCE")
    if values2:
        for bar, v in zip(bars2, values2):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values2) * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # ── Stage 3: substituent 平均 PCE ──
    items3 = sorted(sub_avg_pce.items(), key=lambda x: x[1], reverse=True)[:20]
    if items3:
        labels3, values3 = zip(*items3)
    else:
        labels3, values3 = (), ()
    labels3 = [l.replace("substituent_", "") for l in labels3]
    colors3 = []
    for l in labels3:
        if l == "fluorine":
            colors3.append("#d62728")
        elif l == "chlorine":
            colors3.append("#ff7f0e")
        else:
            colors3.append("#1f77b4")
    bars3 = ax3.bar(range(len(labels3)), values3, color=colors3)
    ax3.set_xticks(range(len(labels3)))
    ax3.set_xticklabels(labels3, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Average PCE")
    ax3.set_title("Stage 3: Substituent Average PCE (within core_benzene)")
    if values3:
        for bar, v in zip(bars3, values3):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values3) * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # ── Stage 4: N 平均 PCE ──
    items4 = sorted(n_avg_pce.items())
    labels4 = [f"N={k}" for k, v in items4]
    values4 = [v for k, v in items4]
    colors4 = ["#d62728" if k == 4 else "#1f77b4" for k, v in items4]
    bars4 = ax4.bar(range(len(labels4)), values4, color=colors4)
    ax4.set_xticks(range(len(labels4)))
    ax4.set_xticklabels(labels4, fontsize=10)
    ax4.set_ylabel("Average PCE")
    ax4.set_title("Stage 4: N-substitute Average PCE (fluorine + chlorine subset)")
    if values4:
        for bar, v in zip(bars4, values4):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values4) * 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # ── Stage 5: 目标 SMILES 存在性 ──
    ax5.axis("off")
    text5 = (
        f"Stage 5: Target SMILES Search\n\n"
        f"Target: Fc1cc(Cl)c(Cl)c(Cl)c1\n"
        f"Search space: {n_fcl} molecules (F+Cl subset)\n"
        f"Result: {'✓ FOUND' if found else '✗ NOT FOUND'}"
    )
    color5 = "green" if found else "red"
    ax5.text(0.5, 0.5, text5, transform=ax5.transAxes, fontsize=12,
            verticalalignment="center", horizontalalignment="center",
            fontfamily="monospace", color=color5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.tight_layout()
    out_path = os.path.join(out_dir, "pattern_check_combined.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → 已保存 pattern_check_combined.png (含 Stage 2~5 四合一子图)")


# ═══════════════════════════════════════════════════════════════════════
#  main: 执行判断 + 绘图
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    CSV_PATH = os.path.join(os.path.dirname(__file__), "generated_molecules_test.csv")
    OUT_DIR = os.path.join(os.path.dirname(__file__), "pattern_check_output")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  分子模式判断 + 阶段平均PCE绘图")
    print("=" * 60)

    # ── 执行判断(stop_on_fail=False: 始终走完所有阶段)──
    result = check_molecule_pattern(CSV_PATH, top_ratio=0.05, stop_on_fail=False)
    print(f"\n{'=' * 60}")
    print(f"  最终判断: {result}")
    print(f"{'=' * 60}")

    # ── 重新遍历数据,绘制各阶段平均PCE图 ──
    print("\n>>> 生成各阶段平均PCE图 ...")

    all_rows, top_rows = _read_all(CSV_PATH)

    # --- Stage 2: core 平均 PCE ---
    core_pce_sum = {}
    core_pce_count = {}
    for r in top_rows:
        for tag in r["remark"].split("+"):
            tag = tag.strip()
            if tag.startswith("core_"):
                base = tag.replace("core_", "").replace("_substitute", "")
                if "_" in base and base.rsplit("_", 1)[1].isdigit():
                    base = base.rsplit("_", 1)[0]
                core_pce_sum[base] = core_pce_sum.get(base, 0.0) + r["Predicted_PCE"]
                core_pce_count[base] = core_pce_count.get(base, 0) + 1
                break
    core_avg_pce = {k: core_pce_sum[k] / core_pce_count[k] for k in core_pce_count}

    # --- Stage 3: substituent 平均 PCE (core_benzene 子集) ---
    benzene_rows = [r for r in top_rows
                    if any(t.startswith("core_benzene") for t in r["remark"].split("+"))]
    sub_pce_sum = {}
    sub_pce_count = {}
    for r in benzene_rows:
        for tag in r["remark"].split("+"):
            tag = tag.strip()
            if tag.startswith("substituent_"):
                sub_pce_sum[tag] = sub_pce_sum.get(tag, 0.0) + r["Predicted_PCE"]
                sub_pce_count[tag] = sub_pce_count.get(tag, 0) + 1
    sub_avg_pce = {k: sub_pce_sum[k] / sub_pce_count[k] for k in sub_pce_count}

    # --- Stage 4: N 平均 PCE (fluorine + chlorine 子集) ---
    fcl_rows = []
    for r in benzene_rows:
        tags = r["remark"].split("+")
        if (any(t.strip() == "substituent_fluorine" for t in tags)
                and any(t.strip() == "substituent_chlorine" for t in tags)):
            fcl_rows.append(r)
    n_pce_sum = {}
    n_pce_count = {}
    for r in fcl_rows:
        for tag in r["remark"].split("+"):
            tag = tag.strip()
            if tag.startswith("core_benzene_") and tag.endswith("_substitute"):
                try:
                    n_val = int(tag.replace("core_benzene_", "").replace("_substitute", ""))
                    n_pce_sum[n_val] = n_pce_sum.get(n_val, 0.0) + r["Predicted_PCE"]
                    n_pce_count[n_val] = n_pce_count.get(n_val, 0) + 1
                except ValueError:
                    pass
    n_avg_pce = {k: n_pce_sum[k] / n_pce_count[k] for k in n_pce_count}

    # --- Stage 5: 目标 SMILES ---
    target = "Fc1cc(Cl)c(Cl)c(Cl)c1"
    found = any(r["SMILES"] == target for r in fcl_rows)

    # 合并为一张 2×2 子图
    _plot_all_stages(core_avg_pce, core_pce_count, sub_avg_pce, n_avg_pce, found, len(fcl_rows), OUT_DIR)

    print(f"\n图片已保存至: {OUT_DIR}")
    print("完成。")
