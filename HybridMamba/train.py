import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, precision_score
import numpy as np
import os
import csv
from tqdm import tqdm

# 导入 model.py 中的组件
from model import Config, set_seed, RNADataset, load_real_data, get_model_class

def compute_all_metrics(y_true, y_probs):
    y_pred = np.round(y_probs)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn + 1e-10)
    sp = tn / (tn + fp + 1e-10)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try: auc_val = roc_auc_score(y_true, y_probs)
    except: auc_val = 0.5
    return {"ACC": acc, "Sn": sn, "Sp": sp, "Pre": pre, "F1": f1, "MCC": mcc, "AUC": auc_val}

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(Config.DEVICE), labels.to(Config.DEVICE)
        optimizer.zero_grad()
        outputs = model(seqs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(seqs).squeeze(1)
            probs.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return np.array(probs), np.array(targets)

def save_training_logs(fold_results, save_dir, model_type):
    """
    将每折的详细结果和平均结果保存为 CSV 文件
    """
    filename = os.path.join(save_dir, f"training_log_{model_type}.csv")
    metrics = ["ACC", "Sn", "Sp", "Pre", "F1", "MCC", "AUC"]
    
    print(f"\n[Log] Saving training metrics to {filename} ...")
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # 1. 写入表头
        header = ["Fold"] + metrics
        writer.writerow(header)
        
        # 2. 写入每一折的数据
        num_folds = len(fold_results["ACC"])
        for i in range(num_folds):
            row = [f"Fold_{i+1}"]
            for m in metrics:
                row.append(f"{fold_results[m][i]:.4f}")
            writer.writerow(row)
            
        # 3. 写入分隔空行
        writer.writerow([])
        
        # 4. 写入平均值和标准差
        avg_row = ["Average"]
        std_row = ["Std Dev"]
        for m in metrics:
            avg_val = np.mean(fold_results[m])
            std_val = np.std(fold_results[m])
            avg_row.append(f"{avg_val:.4f}")
            std_row.append(f"{std_val:.4f}")
            
        writer.writerow(avg_row)
        writer.writerow(std_row)

# --- 核心训练流程 ---
def run_training_pipeline(data_file, save_path, model_type='hybrid', k_folds=5, epochs=30):
    set_seed(Config.SEED)
    print(f"\n[Train] Device: {Config.DEVICE} | Model: {model_type} | K-Fold: {k_folds} | Epochs: {epochs}")
    print(f"[Train] Data: {data_file}")
    
    # 提取保存目录 (例如 zr/Hybrid/models/)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return

    full_seqs, full_labels = load_real_data(data_file)
    dataset = RNADataset(full_seqs, full_labels)
    labels_np = np.array(full_labels)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=Config.SEED)

    ModelClass = get_model_class(model_type)
    
    # 记录每折的最佳结果
    fold_results = {"ACC": [], "Sn": [], "Sp": [], "Pre": [], "F1": [], "MCC": [], "AUC": []}
    global_best_acc = 0.0

    # === K-Fold 循环 ===
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np)):
        print(f"\n{'='*15} Fold {fold_idx + 1}/{k_folds} {'='*15}")
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=Config.BATCH_SIZE, shuffle=False)
        
        model = ModelClass().to(Config.DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        
        fold_best_acc = 0.0
        fold_best_metrics = {}
        
        pbar = tqdm(range(epochs), desc=f"Fold {fold_idx+1}", unit="epoch")
        
        for epoch in pbar:
            t_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            v_probs, v_targets = validate(model, val_loader, criterion)
            metrics = compute_all_metrics(v_targets, v_probs)
            
            # 更新每折最佳记录
            if metrics["ACC"] > fold_best_acc:
                fold_best_acc = metrics["ACC"]
                fold_best_metrics = metrics
                
                # [新增] 保存这一折的最佳模型 (例如: fold_1_model_hybrid.pth)
                fold_model_name = f"fold_{fold_idx+1}_model_{model_type}.pth"
                torch.save(model.state_dict(), os.path.join(save_dir, fold_model_name))
                
                # 如果是全局最好，更新 best_model.pth (用于后续独立测试)
                if fold_best_acc > global_best_acc:
                    global_best_acc = fold_best_acc
                    torch.save(model.state_dict(), save_path)
            
            pbar.set_postfix({
                'Loss': f"{t_loss:.4f}",
                'Val_Acc': f"{metrics['ACC']:.4f}",
                'Best': f"{fold_best_acc:.4f}"
            })
        
        # 记录本折最终结果
        print(f"Fold {fold_idx+1} Finished. Best ACC: {fold_best_metrics['ACC']:.4f}")
        for k, v in fold_best_metrics.items(): 
            fold_results[k].append(v)

    # === 最终处理 ===
    # 1. 保存日志文件
    save_training_logs(fold_results, save_dir, model_type)

    # 2. 打印摘要
    print(f"\n{'#'*20} Training Summary {'#'*20}")
    print(f"Global Best Accuracy: {global_best_acc:.4f}")
    print(f"Best model saved to : {save_path}")
    print(f"Fold models saved in: {save_dir}")
    print("-" * 50)
    print(f"{'Metric':<10} | {'Average':<10} | {'Std Dev':<10}")
    print("-" * 50)
    
    metric_order = ["ACC", "Sn", "Sp", "Pre", "F1", "MCC", "AUC"]
    for m in metric_order:
        print(f"{m:<10} | {np.mean(fold_results[m]):.4f}     | +/- {np.std(fold_results[m]):.4f}")
    print("-" * 50)