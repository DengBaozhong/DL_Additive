# 0_Molecule_generation — 分子生成与片段库

基于 BRICS 连接点的有机分子生成模块。从核心片段库和取代基片段库中组合生成候选添加剂分子。

### 生成原理

对每个核心片段，枚举其 K 个锚点上所有可能的取代基排列（`itertools.product`，可重复选取），
每个组合只尝试一次，零浪费。连接策略为确定性：取代基按 tuple 顺序依次连接到当前分子的第一个可用锚点。

**枚举策略**（基于组合数 $S^K$ 与阈值的比较）：
- **全枚举**（$S^K \le$ `EXHAUSTIVE_LIMIT`）：遍历所有 $S^K$ 种组合，保证完整覆盖。
- **随机采样**（$S^K \gt$ `EXHAUSTIVE_LIMIT`）：固定随机种子采样 `SAMPLING_BUDGET` 个不重复组合，兼顾大空间的探索。

- 枚举空间 = $S^K$（$S$=取代基种类数，$K$=核心锚点数）
- 全枚举时效率 ≈ 100%，随机采样时效率取决于空间大小与预算比例

---

## 快速开始

```bash
cd 0_Molecule_generation
python additive_molecule_generation.py
```

生成结果保存在 `generated_molecules.csv`。

---

## 配置参数

所有可调参数集中在 `additive_molecule_generation.py` 顶部的 **用户配置区**：

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_WORKERS` | `None` | 并行进程数，`None` = 自动（不超过核心类型数） |

### 枚举 / 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EXHAUSTIVE_LIMIT` | `10000` | 全枚举上限：组合数 ≤ 此值时全枚举；超过则随机采样 |
| `SAMPLING_BUDGET` | `2000` | 随机采样时的尝试次数 |
| `RANDOM_SEED` | `42` | 随机种子（保证可复现） |

### 输入输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CORES_CSV` | `"Cores.csv"` | 核心片段库路径 |
| `SUBSTITUENTS_CSV` | `"Substituents.csv"` | 取代基片段库路径 |
| `OUTPUT_CSV` | `"generated_molecules.csv"` | 输出文件路径 |

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `additive_molecule_generation.py` | 主生成脚本（配置集中管理、多进程枚举/采样） |
| `check_molecule_pattern.py` | 分子模式验证脚本（五步漏斗法检查生成结果是否收敛到预期最优模式） |
| `find_matching_smiles.py` | SMILES 等价匹配脚本（两阶段：字符预筛 + RDKit 规范化比较） |
| `Cores.csv` | 核心片段库（UTF-8 BOM，~3,046 条去重） |
| `Substituents.csv` | 取代基片段库（UTF-8 BOM，10 条） |
| `generated_molecules.csv` | 生成结果输出 |
| `pattern_check_output/` | 模式验证的输出图表（PCE 分布图 + 漏斗组合图） |


---

## 命名规范

### Core_type_en（核心片段英文代码）

格式：`core_{骨架名}_{N}_substitute`

其中 `N` 为 BRICS 连接点（`*`）数量，即该核心可连接的取代基个数。

| 中文 Core_type | 英文代码 | 骨架说明 |
|:--|:--|:--|
| `N取代苯` | `core_benzene_N_substitute` | 苯环 |
| `N取代联苯` | `core_biphenyl_N_substitute` | 联苯 |
| `N取代二苯醚` | `core_diphenyl_ether_N_substitute` | 二苯醚 |
| `N取代三苯胺` | `core_triphenylamine_N_substitute` | 三苯胺 |
| `N取代萘` | `core_naphthalene_N_substitute` | 萘 |
| `N取代蒽` | `core_anthracene_N_substitute` | 蒽 |
| `N取代菲` | `core_phenanthrene_N_substitute` | 菲 |
| `N取代吡啶` | `core_pyridine_N_substitute` | 吡啶 |
| `N取代44联吡啶` | `core_bipyridine_44_N_substitute` | 4,4'-联吡啶 |
| `N取代嘧啶` | `core_pyrimidine_N_substitute` | 嘧啶 |
| `N取代吡嗪` | `core_pyrazine_N_substitute` | 吡嗪 |
| `N取代噻吩` | `core_thiophene_N_substitute` | 噻吩（单环） |
| `N取代反式并二噻吩` | `core_trans_thienothiophene_N_substitute` | 反式-并二噻吩 |
| `N取代顺式并二噻吩` | `core_cis_thienothiophene_N_substitute` | 顺式-并二噻吩 |
| `N取代反反并三噻吩` | `core_trans_trans_dithienothiophene_N_substitute` | 反反-并三噻吩 |
| `N取代顺顺并三噻吩` | `core_cis_cis_dithienothiophene_N_substitute` | 顺顺-并三噻吩 |
| `N取代顺反并三噻吩` | `core_cis_trans_dithienothiophene_N_substitute` | 顺反-并三噻吩 |
| `N取代苯并二噻吩` | `core_benzodithiophene_N_substitute` | 苯并二噻吩 |

> 共 18 个骨架族，136 种唯一 Core_type_en（去重后）。

### Substituent_type（取代基英文代码）

格式：`substituent_{官能团名}`

| 中文名称 | SMILES | 英文代码 |
|:--|:--|:--|
| 氟 | `F*` | `substituent_fluorine` |
| 氯 | `Cl*` | `substituent_chlorine` |
| 溴 | `Br*` | `substituent_bromine` |
| 碘 | `I*` | `substituent_iodine` |
| 甲氧基 | `*OC` | `substituent_methoxy` |
| 甲基 | `*C` | `substituent_methyl` |
| 1噻吩 | `*C1=CC=CS1` | `substituent_thiophene` |
| 氰基 | `*C#N` | `substituent_cyano` |
| 甲苯基 | `*c1ccccc1` | `substituent_tolyl` |
| 羟基 | `*O` | `substituent_hydroxyl` |

> 共 10 种取代基。

---

## 数据统计

| 指标 | 值 |
|:--|:--|
| 原始 Molecules.csv 行数 | 3,053 |
| 去重后核心片段数 | 3,046 |
| 唯一 Core_type_en | 136 |
| 取代基种类 | 10 |
| SMILES 重复导致丢弃的行 | 7 行（顺反并三噻吩 4 行 + 苯并二噻吩 3 行，均为 1取代↔2取代 SMILES 重合） |

---

## 编码说明

- `Molecules.csv` 原始文件使用 **GBK** 编码
- `Cores.csv` 和 `Substituents.csv` 输出为 **UTF-8 BOM**（utf-8-sig），兼容 Excel 直接打开
