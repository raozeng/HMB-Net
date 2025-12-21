import argparse
import os
from train import run_training_pipeline
from test import run_test_pipeline

def main():
    parser = argparse.ArgumentParser(description="RNA Methylation Prediction Runner")
    
    # === 基础参数 ===
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'all'],
                        help="Execution mode: 'train', 'test', or 'all' (both).")
    
    parser.add_argument('--model_type', type=str, default='hybrid', choices=['hybrid', 'cnn', 'mamba','transformer','resnet','bilstm','textcnn','performer','flashattn','bilstm_base','reverse'],
                        help="Model architecture: 'hybrid' (Ours), 'cnn' (Ablation), 'mamba' (Ablation).")
    
    # === 路径参数 ===
    parser.add_argument('--train_data', type=str, default='train_dataset.fasta',
                        help="Path to training FASTA file.")
    
    parser.add_argument('--test_data', type=str, default='test_dataset.fasta',
                        help="Path to independent test FASTA file.")
    
    parser.add_argument('--model_dir', type=str, default='checkpoints',
                        help="Directory to save/load trained model weights.")
    
    parser.add_argument('--result_dir', type=str, default='results',
                        help="Directory to save prediction CSVs.")
    
    # === 训练超参数 ===
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs per fold.")
    parser.add_argument('--k_folds', type=int, default=5, help="Number of Cross-Validation folds.")
    
    args = parser.parse_args()
    
    # 构建文件名
    # 模型保存路径: checkpoints/best_model_hybrid.pth
    model_save_path = os.path.join(args.model_dir, f"best_model_{args.model_type}.pth")
    
    # 结果保存路径: results/pred_result_hybrid.csv
    result_csv_path = os.path.join(args.result_dir, f"pred_result_{args.model_type}.csv")

    # === 执行逻辑 ===
    if args.mode in ['train', 'all']:
        print(f"=== Starting Training ({args.model_type}) ===")
        run_training_pipeline(
            data_file=args.train_data,
            save_path=model_save_path,
            model_type=args.model_type,
            k_folds=args.k_folds,
            epochs=args.epochs
        )

    if args.mode in ['test', 'all']:
        print(f"\n=== Starting Testing ({args.model_type}) ===")
        run_test_pipeline(
            data_file=args.test_data,
            model_path=model_save_path,
            result_file=result_csv_path,
            model_type=args.model_type
        )

if __name__ == "__main__":
    main()