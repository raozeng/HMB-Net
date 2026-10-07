import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Config

# ==========================================
# 3. 核心组件 (Core Components)
# ==========================================

# [新增] Deep BiLSTM Branch (替代 CNN)
class DeepBiLSTMBranch(nn.Module):
    def __init__(self, d_model, num_layers=2, dropout=0.3):
        super().__init__()
        # Bidirectional LSTM
        # Hidden Size = d_model // 2, 拼接后输出维度刚好是 d_model
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # 全局残差归一化
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        # global_residual = x
        
        # LSTM output: [Batch, Seq, Dim]
        out, _ = self.lstm(x)
        
        # Global Residual + Norm
        return self.final_norm(out)

# [保留] Deep Mamba Branch
class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.core = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            self.is_mamba = True
        except:
            self.core = nn.GRU(d_model, d_model//2, batch_first=True, bidirectional=True)
            self.is_mamba = False
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.core(x) if self.is_mamba else self.core(x)[0]
        return self.norm(out + x)

class DeepMambaBranch(nn.Module):
    def __init__(self, d_model, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

# [保留] Deep Multi-Scale CNN (用于对比 Baseline)
class MultiScaleCNNBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model, 5, padding=2)
        self.conv7 = nn.Conv1d(d_model, d_model, 7, padding=3)
        self.fusion = nn.Conv1d(d_model*3, d_model, 1)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x_in = x.transpose(1, 2)
        cat = torch.cat([self.relu(self.conv3(x_in)), self.relu(self.conv5(x_in)), self.relu(self.conv7(x_in))], dim=1)
        out = self.fusion(cat).transpose(1, 2)
        return self.norm(out + x)

class DeepMultiScaleCNNBranch(nn.Module):
    def __init__(self, d_model, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([MultiScaleCNNBlock(d_model) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        global_residual = x
        for layer in self.layers: x = layer(x)
        return self.final_norm(x + global_residual)

# [保留] Attention Fusion
class BiDirectionalGatedCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, h1, h2):
        z1, _ = self.attn1(h1, h2, h2)
        z2, _ = self.attn2(h2, h1, h1)
        return self.norm(z1 + z2 + h1 + h2)


# ==========================================
# 4. 主模型与消融模型 (New Hybrid: BiLSTM + Mamba)
# ==========================================

class HybridPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.bilstm = DeepBiLSTMBranch(Config.EMBED_DIM, num_layers=4)
        self.mamba = DeepMambaBranch(Config.EMBED_DIM, num_layers=1)
        self.fusion = BiDirectionalGatedCrossAttention(Config.EMBED_DIM)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(nn.Linear(Config.EMBED_DIM, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))
        
    def forward(self, x):
        x = self.emb(x) 
        # 双流特征提取
        h_mamba = self.mamba(x) # 擅长全局长距离
        h_lstm = self.bilstm(h_mamba) # 擅长序列前后依赖
        # 融合
        v = self.pool(self.fusion(h_lstm, h_mamba).transpose(1, 2)).squeeze(-1)
        return self.clf(v)

# [消融 1] 纯 Mamba (保持不变)
class MambaOnlyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.mamba = DeepMambaBranch(Config.EMBED_DIM, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(nn.Linear(Config.EMBED_DIM, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        v = self.pool(self.mamba(self.emb(x)).transpose(1, 2)).squeeze(-1)
        return self.clf(v)

# [消融 2] 纯 BiLSTM (新增)
class BiLSTMOnlyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.bilstm = DeepBiLSTMBranch(Config.EMBED_DIM, num_layers=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(nn.Linear(Config.EMBED_DIM, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        v = self.pool(self.bilstm(self.emb(x)).transpose(1, 2)).squeeze(-1)
        return self.clf(v)

# [消融 3] 纯 CNN (保留用于对比)
class CNNOnlyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.cnn = DeepMultiScaleCNNBranch(Config.EMBED_DIM, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(nn.Linear(Config.EMBED_DIM, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        v = self.pool(self.cnn(self.emb(x)).transpose(1, 2)).squeeze(-1)
        return self.clf(v)


# [新增] 反向顺序模型 (用于消融实验：BiLSTM -> Mamba)
class ReverseHybridPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.bilstm = DeepBiLSTMBranch(Config.EMBED_DIM, num_layers=4)
        self.mamba = DeepMambaBranch(Config.EMBED_DIM, num_layers=1)
        self.fusion = BiDirectionalGatedCrossAttention(Config.EMBED_DIM)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(nn.Linear(Config.EMBED_DIM, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        
    def forward(self, x):
        x = self.emb(x) 
        # Local-to-Global Strategy
        h_lstm = self.bilstm(x)       # 先提取局部序列特征
        h_mamba = self.mamba(h_lstm)  # 再基于局部特征提取全局依赖
        v = self.pool(self.fusion(h_lstm, h_mamba).transpose(1, 2)).squeeze(-1)
        return self.clf(v)

# ==========================================
# 5. Baseline 模型 
# ==========================================

class TransformerBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, Config.SEQ_LEN, Config.EMBED_DIM))
        enc = nn.TransformerEncoderLayer(d_model=Config.EMBED_DIM, nhead=4, dim_feedforward=Config.EMBED_DIM*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(nn.Linear(Config.EMBED_DIM, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.emb(x) + self.pos_emb[:, :x.size(1), :]
        return self.clf(self.pool(self.transformer(x).transpose(1, 2)).squeeze(-1))

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(c_out)
        if c_in != c_out: self.sc = nn.Conv1d(c_in, c_out, 1)
        else: self.sc = nn.Identity()
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.sc(x))

class ResNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.layers = nn.Sequential(ResBlock(Config.EMBED_DIM, 128), ResBlock(128, 256), ResBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        return self.clf(self.pool(self.layers(self.emb(x).transpose(1, 2))).squeeze(-1))

class BiLSTMAttnBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(Config.EMBED_DIM, 64, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.u = nn.Linear(128, 1, bias=False)
        self.clf = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        att = F.softmax(self.u(torch.tanh(h)), dim=1)
        return self.clf(torch.sum(h * att, dim=1))

class TextCNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(Config.EMBED_DIM, 128, k) for k in [3, 5, 7]])
        self.clf = nn.Sequential(nn.Linear(128*3, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        return self.clf(torch.cat([F.adaptive_max_pool1d(F.relu(c(x)), 1).squeeze(-1) for c in self.convs], 1))

class PerformerBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        try: from performer_pytorch import Performer
        except ImportError: raise ImportError("Run: pip install performer-pytorch")
        self.performer = Performer(dim=Config.EMBED_DIM, depth=2, heads=4, dim_head=32, causal=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(nn.Linear(Config.EMBED_DIM, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.clf(self.pool(self.performer(self.emb(x)).transpose(1, 2)).squeeze(-1))

class FlashAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))
    def forward(self, x):
        B, L, D = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn_out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=0.1)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = shortcut + self.proj(attn_out)
        return x + self.ffn(self.norm2(x))

class FlashAttentionBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        if not hasattr(F, 'scaled_dot_product_attention'): raise ImportError("Need PyTorch >= 2.0")
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, Config.SEQ_LEN, Config.EMBED_DIM))
        self.layers = nn.ModuleList([FlashAttentionBlock(Config.EMBED_DIM, 4) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(nn.Linear(Config.EMBED_DIM, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.emb(x) + self.pos_emb[:, :x.size(1), :]
        for l in self.layers: x = l(x)
        return self.clf(self.pool(x.transpose(1, 2)).squeeze(-1))

# ==========================================
# 7. 模型工厂
# ==========================================
def get_model_class(model_type):
    t = model_type.lower()
    
    # Ours
    if t == 'hybrid': return HybridPredictor
    
    # Ablations
    if t == 'cnn': return CNNOnlyPredictor
    if t == 'mamba': return MambaOnlyPredictor
    if t == 'bilstm': return BiLSTMOnlyPredictor # 新增选项
    if t == 'reverse': return ReverseHybridPredictor 
    
    # Baselines
    if t == 'transformer': return TransformerBaseline
    if t == 'resnet': return ResNetBaseline
    if t == 'bilstm_base': return BiLSTMAttnBaseline # 注意区分名字
    if t == 'textcnn': return TextCNNBaseline
    if t == 'performer': return PerformerBaseline
    if t == 'flashattn': return FlashAttentionBaseline
    
    raise ValueError(f"Unknown model type: {model_type}")