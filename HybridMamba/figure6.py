import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ==========================================
# 1. 数据准备 (基于 Table 4 的完整汇总)
# ==========================================

# HMB-Net (Global -> Local)
auc_hmb = [
    0.8229, 0.8807, 0.8905, 0.8825, 0.8331, 0.8942, 0.8100, 0.8604, 0.8663, 0.9143, 0.9004, # Zhang
    0.9954, 0.9779, 0.9435, 0.9429, 0.9573, 0.9684, 0.9616, # Atlas
    0.9870, 0.9893, 0.9689, 0.9564, 0.9573, 0.9844 # RMBase
]
pr_hmb = [
    0.8043, 0.8731, 0.8758, 0.8723, 0.8058, 0.8902, 0.7842, 0.8390, 0.8561, 0.9063, 0.8952, 
    0.9969, 0.9935, 0.9484, 0.9485, 0.9671, 0.9710, 0.9541, 
    0.9896, 0.9892, 0.9707, 0.9694, 0.9598, 0.9852
]

# R-HMB-Net (Local -> Global)
auc_rhmb = [
    0.8201, 0.8832, 0.8859, 0.8801, 0.8283, 0.8906, 0.8050, 0.8530, 0.8633, 0.9114, 0.8865,
    0.9969, 0.9630, 0.9399, 0.9431, 0.9110, 0.9674, 0.9607,
    0.9607, 0.9763, 0.9693, 0.9123, 0.9515, 0.9834
]
pr_rhmb = [
    0.7944, 0.8737, 0.8668, 0.8696, 0.7969, 0.8842, 0.7795, 0.8325, 0.8530, 0.9032, 0.8772,
    0.9978, 0.9888, 0.9452, 0.9485, 0.9410, 0.9699, 0.9530,
    0.9530, 0.9711, 0.9710, 0.9527, 0.9540, 0.9843
]

# ==========================================
# 2. 计算统计量
# ==========================================
def get_stats(data):
    return np.mean(data), stats.sem(data)

mean_auc_hmb, sem_auc_hmb = get_stats(auc_hmb)
mean_auc_r, sem_auc_r = get_stats(auc_rhmb)
mean_pr_hmb, sem_pr_hmb = get_stats(pr_hmb)
mean_pr_r, sem_pr_r = get_stats(pr_rhmb)

# ==========================================
# 3. Nature 风格绘图
# ==========================================
def plot_nature_style_average():
    # 设置全局字体为 Arial (Nature 标准)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
    
    fig, ax = plt.subplots(figsize=(6, 5)) # Nature 单栏图宽度通常在 89mm 左右，这里稍微大一点方便调整
    
    labels = ['Average ROC-AUC', 'Average PR-AUC']
    x = np.arange(len(labels))
    width = 0.4
    
    # --- Nature Color Palette (NPG) ---
    color_hmb = '#E64B35'  # Nature Red
    color_rhmb = '#4DBBD5' # Nature Blue
    
    # 绘制柱子 (无边框，颜色纯正)
    # Zorder=3 确保柱子在网格线上方
    rects1 = ax.bar(x - width/2, [mean_auc_hmb, mean_pr_hmb], width, 
                    yerr=[sem_auc_hmb, sem_pr_hmb], label='HMB-Net (Global→Local)', 
                    color=color_hmb, capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': 'black'}, zorder=3)
    
    rects2 = ax.bar(x + width/2, [mean_auc_r, mean_pr_r], width, 
                    yerr=[sem_auc_r, sem_pr_r], label='R-HMB-Net (Local→Global)', 
                    color=color_rhmb, capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': 'black'}, zorder=3)
    
    # --- Nature 风格修饰 ---
    # 1. 移除顶部和右侧的边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. 加粗左侧和底部坐标轴
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # 3. 设置标签字体大小
    ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    
    # 4. Y轴范围聚焦 (显示微小差异)
    ax.set_ylim(0.90, 0.94) 
    # 只显示几个刻度，保持整洁
    ax.set_yticks([0.90, 0.91, 0.92, 0.93, 0.94])
    
    # 5. 添加图例 (去框，放在合适位置)
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    
    # 6. 添加轻微的虚线网格 (只在Y轴)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)

    # 标注数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='black')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('Figure6_Nature_Style.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_nature_style_average()