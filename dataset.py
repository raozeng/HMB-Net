import torch
import numpy as np
import os
import random
from torch.utils.data import Dataset

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
class Config:
    SEED = 64
    SEQ_LEN = 101       # 序列长度
    VOCAB_SIZE = 5      # N=0, A=1, U=2, G=3, C=4
    EMBED_DIM = 128     
    BATCH_SIZE = 64
    EPOCHS = 50         
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. 数据处理工具 (Single Nucleotide)
# ==========================================
class RNATokenizer:
    def __init__(self):
        self.mapper = {'N': 0, 'A': 1, 'U': 2, 'G': 3, 'C': 4, 'T': 2}

    def encode(self, seq_str):
        seq_str = seq_str.upper().strip()
        if len(seq_str) > Config.SEQ_LEN:
            seq_str = seq_str[:Config.SEQ_LEN]
        encoded = [self.mapper.get(base, 0) for base in seq_str]
        if len(encoded) < Config.SEQ_LEN:
            encoded += [0] * (Config.SEQ_LEN - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)

class RNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = RNATokenizer()
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        seq_tensor = self.tokenizer.encode(self.sequences[idx])
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq_tensor, label_tensor

def load_real_data(file_path):
    sequences, labels = [], []
    if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f: lines = f.readlines()
    curr_lbl = None
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if "+" in line: curr_lbl = 1
            elif "-" in line: curr_lbl = 0
            else: curr_lbl = None
        elif curr_lbl is not None:
            sequences.append(line)
            labels.append(curr_lbl)
            curr_lbl = None
    return sequences, labels
