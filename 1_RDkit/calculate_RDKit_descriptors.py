import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_molecular_descriptors(csv_path, output_path):
    """
    
    计算CSV文件中SMILES列的210个分子描述符并保存结果
    
    """
    # 定义210个分子描述符名称
    descriptor_names = [
        'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt',
        'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
        'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 
        'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
        'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v',
        'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
        'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
        'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
        'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6',
        'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
        'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA',
        'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
        'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
        'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9',
        'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
        'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
        'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR',
        'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH',
        'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
        'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
        'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
        'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic',
        'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
        'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
        'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
        'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
        'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
        'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
        'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
        'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea'
    ]
    
    # 获取所有描述符函数
    all_descriptors = [getattr(Descriptors, name) for name in descriptor_names]
    
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='gbk')
    
    # 检查是否存在SMILES列
    if 'SMILES' not in df.columns:
        raise ValueError("Excel文件中未找到SMILES列")
    
    # 定义计算描述符的函数
    def calculate_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * len(descriptor_names)
        return [desc(mol) for desc in all_descriptors]
    
    # 计算每个分子的描述符
    print(f"开始计算{len(df)}个分子的描述符...")
    descriptors_list = []
    
    for i, smiles in enumerate(df['SMILES']):
        if pd.isna(smiles):
            descriptors_list.append([np.nan] * len(descriptor_names))
        else:
            try:
                descriptors = calculate_descriptors(smiles)
                descriptors_list.append(descriptors)
            except Exception as e:
                print(f"计算描述符失败 (SMILES: {smiles}): {e}")
                descriptors_list.append([np.nan] * len(descriptor_names))
        
        # 打印进度
        if (i+1) % 50 == 0 or i+1 == len(df):
            print(f"已完成: {i+1}/{len(df)} ({(i+1)/len(df)*100:.1f}%)")
    
    # 将结果转换为DataFrame
    descriptors_df = pd.DataFrame(descriptors_list, columns=descriptor_names)
    
    # 合并原始数据和描述符数据
    result_df = pd.concat([df, descriptors_df], axis=1)
    
    # 保存结果到新的CSV文件
    result_df.to_csv(output_path, index=False, encoding='gbk')
    print(f"计算完成！结果已保存至: {output_path}")
    
    return result_df

# 使用示例
if __name__ == "__main__":
    input_file = "替换为实际的输入文件路径"  
    output_file = "替换为实际的输出文件路径"  
    calculate_molecular_descriptors(input_file, output_file)