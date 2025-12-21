import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ==========================================
# 1. 准备数据 (保持不变)
# ==========================================
radar_datasets = {
    # --- Row 1: Kidney (Top Row) ---
    'Human Kidney (h_k)': { # Col 1
        'HMB-Net':     [0.7896, 0.7538, 0.8255, 0.5808, 0.8807],
        'DeepMRMP':    [0.7925, 0.7955, 0.7894, 0.5850, 0.8701],
        'DeepPromise': [0.7754, 0.7621, 0.7888, 0.5510, 0.8548],
        'M6A-BERT':    [0.7700, 0.7035, 0.8364, 0.5447, 0.8603]
    },
    'Mouse Kidney (m_k)': { # Col 2
        'HMB-Net':     [0.8136, 0.8229, 0.8044, 0.6274, 0.8942],
        'DeepMRMP':    [0.8085, 0.8112, 0.8057, 0.6169, 0.8923],
        'DeepPromise': [0.8086, 0.8297, 0.7874, 0.6177, 0.8926],
        'M6A-BERT':    [0.7919, 0.8105, 0.7733, 0.5842, 0.8736]
    },
    'Rat Kidney (r_k)': {   # Col 3
        'HMB-Net':     [0.8368, 0.8648, 0.8089, 0.6747, 0.9143],
        'DeepMRMP':    [0.8349, 0.8514, 0.8185, 0.6702, 0.9113],
        'DeepPromise': [0.8246, 0.8293, 0.8199, 0.6492, 0.9091],
        'M6A-BERT':    [0.8183, 0.8424, 0.7943, 0.6374, 0.8963]
    },

    # --- Row 2: Liver/Brain (Bottom Row) ---
    'Human Liver (h_l)': {  # Col 1
        'HMB-Net':     [0.8144, 0.8159, 0.8128, 0.6287, 0.8905],
        'DeepMRMP':    [0.8109, 0.8470, 0.7749, 0.6235, 0.8868],
        'DeepPromise': [0.8001, 0.8079, 0.7923, 0.6003, 0.8834],
        'M6A-BERT':    [0.7948, 0.8254, 0.7642, 0.5907, 0.8737]
    },
    'Mouse Brain (m_b)': {  # Col 2
        'HMB-Net':     [0.7961, 0.8264, 0.7657, 0.5932, 0.8825],
        'DeepMRMP':    [0.7914, 0.8072, 0.7756, 0.5831, 0.8791],
        'DeepPromise': [0.7926, 0.8135, 0.7717, 0.5857, 0.8770],
        'M6A-BERT':    [0.7711, 0.7944, 0.7478, 0.5428, 0.8515]
    },
    'Rat Liver (r_l)': {    # Col 3
        'HMB-Net':     [0.8173, 0.8485, 0.7860, 0.6357, 0.9004],
        'DeepMRMP':    [0.8085, 0.8167, 0.8002, 0.6170, 0.8888],
        'DeepPromise': [0.8104, 0.8485, 0.7724, 0.6227, 0.8955],
        'M6A-BERT':    [0.7968, 0.8167, 0.7770, 0.5941, 0.8757]
    }
}

metrics_labels = ['ACC', 'Sn', 'Sp', 'MCC', 'AUC']
models = ['HMB-Net', 'DeepMRMP', 'DeepPromise', 'M6A-BERT']

# ==========================================
# 2. Nature 风格绘图函数
# ==========================================
def plot_radar_nature():
    # 设置全局字体 (Nature 标准: Arial/Helvetica)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # --- NPG 经典配色 ---
    colors = {
        'HMB-Net': '#E64B35',     # Red
        'DeepMRMP': '#4DBBD5',    # Blue
        'DeepPromise': '#00A087', # Green
        'M6A-BERT': '#3C5488'     # Dark Blue
    }
    
    line_styles = {
        'HMB-Net': '-', 
        'DeepMRMP': '--', 
        'DeepPromise': '-.', 
        'M6A-BERT': ':'
    }
    
    # 线宽调整 (HMB-Net 稍微粗一点，但不夸张)
    line_widths = {
        'HMB-Net': 2.0, 
        'DeepMRMP': 1.2, 
        'DeepPromise': 1.2, 
        'M6A-BERT': 1.2
    }
    
    # Z-Order (HMB-Net 在最上层)
    z_orders = {
        'HMB-Net': 10,
        'DeepMRMP': 1,
        'DeepPromise': 2,
        'M6A-BERT': 3
    }

    # 创建画布
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    dataset_keys = list(radar_datasets.keys())
    
    for i, dataset_name in enumerate(dataset_keys):
        ax = axes[i]
        data_values = radar_datasets[dataset_name]
        
        # 计算角度
        N = len(metrics_labels)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # 背景网格优化 (极简风格)
        ax.xaxis.grid(True, color='#888888', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(True, color='#888888', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.spines['polar'].set_visible(False) # 移除外圆框，显得更现代
        
        # 绘制模型
        for model in models:
            values = data_values[model]
            values += values[:1]
            
            ax.plot(angles, values, 
                    linewidth=line_widths[model], 
                    linestyle=line_styles[model], 
                    label=model, 
                    color=colors[model],
                    zorder=z_orders[model])
            
            if model == 'HMB-Net':
                ax.fill(angles, values, color=colors[model], alpha=0.08) # 淡淡的填充
        
        # 设置轴标签 (字体稍微大一点，加粗)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, fontsize=10, fontweight='bold')
        
        # 设置Y轴刻度 (只显示几个关键值，不显示标签以保持整洁)
        ax.set_ylim(0.5, 0.98)
        ax.set_yticks([0.6, 0.7, 0.8, 0.9])
        ax.set_yticklabels([]) 
        
        # 子图标题 (放在底部，更符合 Nature 排版习惯，或者顶部)
        clean_name = dataset_name.split('(')[0].strip()
        ax.set_title(f"{clean_name}", size=12, weight='bold', position=(0.5, 1.12), pad=10)

    # 添加列标签 (Human, Mouse, Rat) - 顶部
    fig.text(0.23, 0.95, "Homo sapiens", ha='center', fontsize=14, fontweight='bold')
    fig.text(0.51, 0.95, "Mus musculus", ha='center', fontsize=14, fontweight='bold')
    fig.text(0.79, 0.95, "Rattus norvegicus", ha='center', fontsize=14, fontweight='bold')

    # 图例 (放在底部，无边框)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.03), 
               fontsize=11, frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12, wspace=0.3, hspace=0.35)
    plt.savefig("Figure3_Radar_Nature.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_radar_nature()