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
                        help="Model architecture | Main: 'hybrid' (HMB-Net) | Ablations: 'bilstm' (BiLSTM-Only), 'mamba' (Mamba-Only), 'reverse' (R-HMB-Net) | Others: cnn, transformer, resnet, etc.")
    
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
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for AdamW optimizer.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training/testing.")

    args = parser.parse_args()
    
    # Dynamically override Config via CLI arguments
    from dataset import Config
    if hasattr(args, 'lr'): Config.LEARNING_RATE = args.lr
    if hasattr(args, 'batch_size'): Config.BATCH_SIZE = args.batch_size

    
    # 构建文件名
    # 构建层级化文件路径
    # 训练模型保存路径: train_model/<model_type>/<train_dataset>/best_model.pth
    train_dataset_name = os.path.splitext(os.path.basename(args.train_data))[0]
    train_model_dir = os.path.join('train_model', args.model_type, train_dataset_name)
    os.makedirs(train_model_dir, exist_ok=True)
    model_save_path = os.path.join(train_model_dir, 'best_model.pth')
    
    # 测试结果保存路径: test_result/<model_type>/<test_dataset>/pred_result.csv
    test_dataset_name = os.path.splitext(os.path.basename(args.test_data))[0]
    test_result_dir = os.path.join('test_result', args.model_type, test_dataset_name)
    os.makedirs(test_result_dir, exist_ok=True)
    result_csv_path = os.path.join(test_result_dir, 'pred_result.csv')

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