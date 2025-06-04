# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 23:24:44 2025

@author: maxim
"""

import pandas as pd
import re

# 读取CSV文件
df = pd.read_csv('Molecules.csv', encoding='GB18030')

# 定义一个函数，用于去除字符串末尾的数字
def remove_trailing_numbers(s):
    # 使用正则表达式去除末尾的数字
    return re.sub(r'\d+$', '', s)

# 对'Core_type'列应用该函数
df['Core_type'] = df['Core_type'].apply(remove_trailing_numbers)

# 保存修改后的数据到新的CSV文件（可选）
df.to_csv('modified_file.csv', encoding='GB18030',index=False)

# 打印修改后的数据（可选）
print(df)
