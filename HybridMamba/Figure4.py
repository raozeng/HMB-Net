import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. 准备数据 (保持不变)
# ==========================================
data = {
    'Dataset': [
        'Atlas_h_m1A', 'Atlas_h_m5C', 'Atlas_h_m6A', 
        'Atlas_m_m1A', 'Atlas_m_m6A', 'Atlas_r_m6A',
        'RMbase_h_m1A', 'RMbase_h_m5C', 'RMbase_h_m6A', 
        'RMbase_m_m5C', 'RMbase_m_m6A', 'RMbase_r_m6A'
    ],
    # Table 3 Data
    'HMB-Net': [
        0.9954, 0.9779, 0.9435, 0.9429, 0.9684, 0.9616, 
        0.9870, 0.9893, 0.9689, 0.9892, 0.9573, 0.9844
    ],
    'DeepMRMP': [
        0.9957, 0.9685, 0.9395, 0.9390, 0.9662, 0.9605, 
        0.9931, 0.9716, 0.9680, 0.9409, 0.9556, 0.9830
    ],
    'DeepPromise': [
        0.9731, 0.9690, 0.8798, 0.8608, 0.9273, 0.9533, 
        0.9956, 0.9921, 0.9281, 0.9999, 0.9088, 0.9466
    ],
    'M6A-BERT': [
        0.9969, 0.9955, 0.9382, 0.9327, 0.9636, 0.9523, 
        0.9963, 0.9973, 0.9665, 0.9999, 0.9410, 0.9832
    ]
}

# 转换为 DataFrame
df = pd.DataFrame(data)
df_melt = df.melt(id_vars='Dataset', var_name='Model', value_name='ROC-AUC')

# ==========================================
# 2. Nature 风格绘图设置
# ==========================================
def plot_nature_boxplot():
    # 设置全局字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # 画布大小 (Nature 单栏或 1.5 栏宽度)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 颜色定义 (NPG Palette)
    npg_pal = {
        "HMB-Net": "#E64B35",     # Red
        "DeepMRMP": "#4DBBD5",    # Blue
        "DeepPromise": "#00A087", # Green
        "M6A-BERT": "#3C5488"     # Dark Blue
    }
    
    # 顺序
    order = ["HMB-Net", "DeepMRMP", "DeepPromise", "M6A-BERT"]
    
    # 1. 绘制箱线图
    # fliersize=0 隐藏箱线图自带的异常点，因为我们后面要用 stripplot 画所有点
    sns.boxplot(x='Model', y='ROC-AUC', data=df_melt, order=order,
                palette=npg_pal, width=0.6, linewidth=1.2, fliersize=0, ax=ax)
    
    # 2. 绘制散点 (Swarmplot/Stripplot)
    # 使用深灰色，半透明，让数据分布可见
    sns.stripplot(x='Model', y='ROC-AUC', data=df_melt, order=order,
                  color="#2c3e50", size=5, alpha=0.6, jitter=0.2, ax=ax)

    # 3. Nature 风格修饰
    # 去除上方和右侧边框
    sns.despine()
    
    # 设置坐标轴标签字体
    ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('') # 移除 X 轴标题，直接看图例即可
    
    # 设置刻度字体
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Y轴范围聚焦 (重点展示高分段差异)
    ax.set_ylim(0.85, 1.02)
    
    # 添加轻微的网格 (仅 Y 轴)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # 标题 (可选，正式发表时通常不需要图内标题，标题写在 Caption 里)
    # 如果需要，保持简洁
    # plt.title('Robustness across Pan-Modification Datasets', fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('Figure4_Nature_Style.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_nature_boxplot()