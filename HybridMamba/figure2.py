import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# 1. 数据准备
# ==========================================
auc_data = {
    'Dataset': ['h_b', 'h_k', 'h_l', 'm_b', 'm_h', 'm_k', 'm_l', 'm_t', 'r_b', 'r_k', 'r_l'],
    'DeepMRMP':    [0.8212, 0.8701, 0.8868, 0.8791, 0.8259, 0.8923, 0.7989, 0.8513, 0.8618, 0.9113, 0.8888],
    'DeepPromise': [0.8130, 0.8548, 0.8834, 0.8770, 0.8333, 0.8926, 0.8088, 0.8556, 0.8587, 0.9091, 0.8955],
    'M6A-BERT':    [0.7751, 0.8603, 0.8737, 0.8515, 0.7845, 0.8736, 0.7578, 0.8051, 0.8326, 0.8963, 0.8757],
    'HMB-Net':     [0.8229, 0.8807, 0.8905, 0.8825, 0.8331, 0.8942, 0.8100, 0.8604, 0.8663, 0.9143, 0.9004]
}

# ==========================================
# 2. Nature 风格绘图函数
# ==========================================
def plot_bar_chart_nature():
    # 设置全局字体 (Nature 标准: Arial/Helvetica)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # 转换为 DataFrame 并设置索引
    df = pd.DataFrame(auc_data)
    df = df.set_index('Dataset')
    
    # --- 关键步骤：重新排序模型，确配色对应正确 ---
    # 我们希望 HMB-Net 用红色，其他人用冷色调
    # 顺序：[DeepMRMP, DeepPromise, M6A-BERT, HMB-Net]
    desired_order = ['DeepMRMP', 'DeepPromise', 'M6A-BERT', 'HMB-Net']
    df = df[desired_order]
    
    # --- NPG (Nature Publishing Group) 经典配色 ---
    # 对应上面的顺序
    nature_colors = [
        '#4DBBD5', # Blue (DeepMRMP)
        '#00A087', # Green (DeepPromise)
        '#3C5488', # Dark Blue (M6A-BERT)
        '#E64B35'  # Red (HMB-Net - Highlight)
    ]
    
    # 创建画布 (宽长一点，适合展示多组数据)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘图
    # width=0.85 让柱子紧凑一些
    # zorder=3 确保柱子在网格线上方
    df.plot(kind='bar', width=0.85, color=nature_colors, 
            edgecolor='black', linewidth=0.5, ax=ax, zorder=3)
    
    # --- Nature 风格修饰 ---
    
    # 1. 移除顶部和右侧边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 加粗左侧和底部线条
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # 2. 坐标轴标签与刻度
    ax.set_ylabel('ROC-AUC Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset (Human, Mouse, Rat)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    
    # 3. Y轴范围聚焦 (根据数据范围调整，凸显差异)
    ax.set_ylim(0.70, 0.95)
    
    # 4. 网格线 (仅 Y 轴，浅灰色，虚线，置于底层)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    
    # 5. 图例设置 (去框，放在底部或图中空白处)
    # 这里放在上方居中，一行排列，非常整洁
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), 
               ncol=4, frameon=False, fontsize=12)
    
    # 6. 移除默认标题 (Nature论文中标题通常在图注Caption里，不在图内)
    # 如果必须加，可以用 ax.set_title
    
    plt.tight_layout()
    plt.savefig('Figure2_BarPlot_Nature.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_bar_chart_nature()