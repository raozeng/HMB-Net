import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. 准备数据 (保持不变)
# ==========================================

datasets_component = [
    'Human Brain (h_b)', 'Human Kidney (h_k)', 
    'Rat Kidney (r_k)', 'Mouse Kidney (m_k)',
    'Atlas Human m6A', 'Atlas Human m5C', 
    'RMBase Mouse m6A', 'Atlas Mouse m1A'
]

# ROC-AUC 数据录入
data_component = {
    'MambaOnly':  [0.8174, 0.8745, 0.9121, 0.8937, 0.9281, 0.9721, 0.9460, 0.9040],
    'BiLSTMOnly': [0.8186, 0.8729, 0.9110, 0.8936, 0.9403, 0.9675, 0.9556, 0.9379],
    'HMB-Net':    [0.8229, 0.8807, 0.9143, 0.8942, 0.9435, 0.9779, 0.9573, 0.9429]
}

# ==========================================
# 2. Nature 风格绘图函数
# ==========================================
def plot_figure_5_nature():
    # 设置全局字体 (Nature 标准: Arial/Helvetica)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # 转换数据
    df = pd.DataFrame(data_component, index=datasets_component)
    
    # --- NPG 经典配色 ---
    # MambaOnly(蓝), BiLSTMOnly(绿), HMB-Net(红)
    nature_colors = ['#4DBBD5', '#00A087', '#E64B35']
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘图 (Pandas 集成绘图)
    # zorder=3 确保柱子盖住网格线
    df.plot(kind='bar', width=0.8, color=nature_colors, 
            edgecolor='black', linewidth=0.6, ax=ax, zorder=3)
    
    # --- 风格修饰 ---
    # 1. 移除多余边框
    sns.despine()
    
    # 2. 网格线 (仅 Y 轴，浅灰色虚线)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    
    # 3. 坐标轴设置
    ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('') # 数据集名称够清楚了，不需要 label
    
    # 4. 刻度设置
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    # 5. Y轴范围 (聚焦差异区域)
    ax.set_ylim(0.80, 1.0)
    
    # 6. 图例设置 (放在顶部，水平排列，无边框)
    # HMB-Net 是红色，非常显眼
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), 
               ncol=3, frameon=False, fontsize=11)
    
    # 紧凑布局
    plt.tight_layout()
    plt.savefig('Figure5_Components_Nature.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_figure_5_nature()