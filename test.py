import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, precision_score, average_precision_score, recall_score

# 导入相关组件
from dataset import Config, set_seed, RNADataset, load_real_data
from model import get_model_class

def compute_all_metrics(y_true, y_probs):
    """
    计算所有常用的二分类指标
    """
    y_pred = np.round(y_probs) # 默认阈值 0.5
    
    # 1. 基础指标
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 防止除以0
    sn = tp / (tp + fn + 1e-10) # Sensitivity / Recall
    sp = tn / (tn + fp + 1e-10) # Specificity
    pre = precision_score(y_true, y_pred, zero_division=0) # Precision
    
    # 2. 综合指标
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 3. AUC 指标
    try: roc_auc_val = roc_auc_score(y_true, y_probs)
    except: roc_auc_val = 0.5

    try: pr_auc_val = average_precision_score(y_true, y_probs)
    except: pr_auc_val = 0.0

    return {
        "ACC": acc,
        "Sn": sn,         
        "Sp": sp,         
        "Pre": pre,       
        "F1": f1,
        "MCC": mcc,
        "ROC_AUC": roc_auc_val,
        "PR_AUC": pr_auc_val
    }

def save_predictions_csv(targets, probs, filename):
    """
    保存集成后的预测分数
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["True_Label", "Predicted_Score"])
        for t, p in zip(targets, probs): 
            writer.writerow([int(t), f"{p:.6f}"])
    print(f"[Test] Ensemble prediction scores saved to {filename}")

def save_test_metrics(metrics, result_file_path, model_type):
    """
    保存最终集成测试指标
    """
    save_dir = os.path.dirname(result_file_path)
    if not save_dir: save_dir = "."
    os.makedirs(save_dir, exist_ok=True)
    
    filename = os.path.join(save_dir, f"test_metrics_{model_type}_ensemble.csv")
    
    print(f"[Test] Saving ensemble metrics to {filename} ...")
    
    header = ["Model", "ACC", "Sn", "Sp", "Pre", "F1", "MCC", "ROC_AUC", "PR_AUC"]
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        writer.writerow([
            f"{model_type}_Ensemble",
            f"{metrics['ACC']:.4f}",
            f"{metrics['Sn']:.4f}",
            f"{metrics['Sp']:.4f}",
            f"{metrics['Pre']:.4f}",
            f"{metrics['F1']:.4f}",
            f"{metrics['MCC']:.4f}",
            f"{metrics['ROC_AUC']:.4f}",
            f"{metrics['PR_AUC']:.4f}"
        ])

# --- 核心测试流程 (集成模式) ---
def run_test_pipeline(data_file, model_path, result_file, model_type='hybrid'):
    set_seed(Config.SEED)
    print(f"\n[Test] Device: {Config.DEVICE} | Model: {model_type} (5-Fold Ensemble Mode)")
    print(f"[Test] Data: {data_file}")
    
    # 1. 确定模型目录
    model_dir = os.path.dirname(model_path)
    if not model_dir: model_dir = "."
    print(f"[Test] Searching for fold models in: {model_dir}")

    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found.")
        return

    # 2. 加载数据
    seqs, labels = load_real_data(data_file)
    test_loader = DataLoader(RNADataset(seqs, labels), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 3. 初始化累加器
    ensemble_probs = np.zeros(len(labels))
    final_targets = np.array(labels) # 目标标签是不变的
    models_found = 0

    # 4. 循环加载 5 折模型进行预测
    ModelClass = get_model_class(model_type)
    
    # 假设 K_FOLDS = 5，遍历 fold_1 到 fold_5
    for fold_idx in range(1, 6):
        # 拼接文件名: fold_1_model_hybrid.pth
        fold_model_name = f"fold_{fold_idx}_model_{model_type}.pth"
        fold_model_path = os.path.join(model_dir, fold_model_name)
        
        if not os.path.exists(fold_model_path):
            print(f"[Warning] Model file not found: {fold_model_path}. Skipping.")
            continue
            
        print(f"--> Processing Fold {fold_idx}: {fold_model_name}")
        
        # 初始化模型并加载权重
        model = ModelClass().to(Config.DEVICE)
        try:
            model.load_state_dict(torch.load(fold_model_path, map_location=Config.DEVICE, weights_only=True))
        except:
            model.load_state_dict(torch.load(fold_model_path, map_location=Config.DEVICE))
        
        model.eval()
        
        # 单模型推理
        fold_probs = []
        with torch.no_grad():
            for seqs_batch, _ in test_loader:
                seqs_batch = seqs_batch.to(Config.DEVICE)
                outputs = model(seqs_batch).squeeze(1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_probs.extend(probs)
        
        # 累加概率
        ensemble_probs += np.array(fold_probs)
        models_found += 1

    if models_found == 0:
        print("[Error] No fold models found! Please run train.py first.")
        return

    # 5. 取平均 (Soft Voting)
    avg_probs = ensemble_probs / models_found
    print(f"\n[Test] Ensemble completed using {models_found} models.")

    # 6. 计算最终指标
    metrics = compute_all_metrics(final_targets, avg_probs)
    
    # 7. 打印摘要
    print("-" * 50)
    print(f"ENSEMBLE TEST SUMMARY ({model_type})")
    print("-" * 50)
    print(f"ACC     : {metrics['ACC']:.4f}")
    print(f"Sn      : {metrics['Sn']:.4f}")
    print(f"Sp      : {metrics['Sp']:.4f}")
    print(f"Pre     : {metrics['Pre']:.4f}")
    print(f"F1      : {metrics['F1']:.4f}")
    print(f"MCC     : {metrics['MCC']:.4f}")
    print(f"ROC_AUC : {metrics['ROC_AUC']:.4f}")
    print(f"PR_AUC  : {metrics['PR_AUC']:.4f}")
    print("-" * 50)
    
    # 8. 保存结果
    save_predictions_csv(final_targets, avg_probs, result_file)
    save_test_metrics(metrics, result_file, model_type)