#!/usr/bin/env python3
# plot_logs.py
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. 读取原始日志
# --------------------------------------------------
log_file = Path('logs_e36.txt')
assert log_file.exists(), f'{log_file} 不在当前目录！'
print(log_file.name)

records = []
with log_file.open(encoding='utf8') as f:
    for line in f:
        # 把 {} 里的内容抠出来
        m = re.search(r'\{.*\}', line)
        if not m:
            continue
        try:
            records.append(json.loads(m.group()))
        except Exception as e:
            print('解析失败，跳过该行：', e)

df = pd.DataFrame(records)
df.sort_values('epoch', inplace=True)          # 保证按 epoch 排序
df.reset_index(drop=True, inplace=True)

# --------------------------------------------------
# 2. 定义“主要指标”
# --------------------------------------------------
'''
train_loss                反映整体收敛趋势
train_class_error         越低越好，辅助评估分类效果
train_loss_ce	          分类(Cross Entropy交叉熵)损失 -> 约束模型识别每个框中是什么类别
train_loss_bbox	          框回归 L1 损失 -> 让预测框的位置和大小接近真实框
train_loss_giou	          GIoU(交并比) 损失 -> 让预测框和真实框在空间上高度重叠
train_cardinality_error   说明预测框数量是否稳定 -> 数量设置(默认900)过多不具有表达性
train_cardinality_error_unscaled  未加权
'''

TRAIN_KEYS = ['train_loss', 'train_class_error',]
TEST_KEYS = ['test_class_error',]

# 如果日志里缺少某列，自动忽略
train_cols = [k for k in TRAIN_KEYS if k in df.columns]
test_cols = [k for k in TEST_KEYS if k in df.columns]

# --------------------------------------------------
# 3. 绘图函数
# --------------------------------------------------
def plot_curves(data_df, cols, stage: str, save_path: str):
    """把给定列画成一张图，不同曲线用不同颜色区分。"""
    plt.figure(figsize=(10, 6))
    for c in cols:
        plt.plot(data_df['epoch'], data_df[c], label=c)
    plt.title(f'{stage} Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'已保存：{save_path}')
    plt.close()

# --------------------------------------------------
# 4. 出图
# --------------------------------------------------
if train_cols:
    plot_curves(df, train_cols, 'Train', f'{log_file.name}_train_metrics.png')

if test_cols:
    plot_curves(df, test_cols, 'Test', f'{log_file.name}_test_metrics.png')

print('全部完成！')

# --------------------------------------------------
# 5. 单独把 ce / bbox / giou 小数值再画一张图
# --------------------------------------------------
TRAIN_CBG = ['train_loss_ce', 'train_loss_bbox', 'train_loss_giou']
TEST_CBG  = ['test_loss_ce',  'test_loss_bbox',  'test_loss_giou']

train_cbg_cols = [k for k in TRAIN_CBG if k in df.columns]
test_cbg_cols  = [k for k in TEST_CBG  if k in df.columns]

if train_cbg_cols:
    plot_curves(df, train_cbg_cols, 'Train (ce/bbox/giou)', f'{log_file.name}_train_ce_bbox_giou.png')

if test_cbg_cols:
    plot_curves(df, test_cbg_cols, 'Test (ce/bbox/giou)', f'{log_file.name}_test_ce_bbox_giou.png')
