# -*- coding: utf-8 -*-
"""
visualizations_full_dash.py (FULL + AUTO DASH)
一键生成：训练过程/预测结果/序列可视化/模型对比/模型解释/动量分析/赛场层级/关键分 可视化图集
并且：运行结束后自动启动 Dash 仪表盘（无需任何命令行参数）

特点（交付检查友好）：
- 自动递归找文件：从脚本所在目录开始（不依赖运行时cwd，不写死路径）
- 三模型（LSTM/GRU/Transformer）自动加载；Transformer 自动适配 state_dict 命名（layers.* / encoder.layers.*）
- 所有图：保存到脚本同目录 + plt.show()弹窗
- 中文标题 + 英文图例；图例不出现空白框
- 动量相关图：优先使用模型预测概率 prob 重算动量（EMA + robust标准化 + 去趋势），避免直线/点状
- 若缺某些文件/依赖（shap、原始逐分csv、训练日志、dash等）会提示 SKIP，不会中断
- Dash 版本兼容：app.run / app.run_server 自动适配；端口 8050-8059 自动切换

运行：
    python visualizations_full_dash.py
"""

import os
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 0) Matplotlib 设置（中文字体）
# =========================
def setup_matplotlib():
    candidates = [
        "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
        "Source Han Sans SC", "Heiti SC", "WenQuanYi Micro Hei",
        "Arial Unicode MS"
    ]
    available = set([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    chosen = None
    for c in candidates:
        if c in available:
            chosen = c
            break
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
    else:
        print("[WARN] 未检测到常见中文字体（SimHei/微软雅黑等），中文可能显示为方块。")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


# =========================
# 1) 文件自动发现（从脚本目录递归找）
# =========================
def script_dir():
    return Path(__file__).resolve().parent

def candidate_roots(max_up=2):
    sd = script_dir().resolve()
    roots = [sd]
    cur = sd
    for _ in range(max_up):
        parent = cur.parent
        if parent == cur:
            break
        if str(parent) == parent.anchor:
            break
        roots.append(parent)
        cur = parent
    uniq, seen = [], set()
    for r in roots:
        s = str(r)
        if s not in seen and r.exists() and r.is_dir():
            seen.add(s)
            uniq.append(r)
    return uniq

def find_one(patterns, root=None):
    roots = [Path(root).resolve()] if root is not None else candidate_roots()
    for base in roots:
        for pat in patterns:
            hits = glob.glob(str(base / "**" / pat), recursive=True)
            if hits:
                hits_sorted = sorted(hits, key=lambda x: len(Path(x).parts))
                return Path(hits_sorted[0]).resolve()
    return None

def find_all(patterns, root=None):
    roots = [Path(root).resolve()] if root is not None else candidate_roots()
    out = []
    for base in roots:
        for pat in patterns:
            out += [Path(x).resolve() for x in glob.glob(str(base / "**" / pat), recursive=True)]
    uniq, seen = [], set()
    for p in out:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            uniq.append(p)
    return uniq


# =========================
# 2) 画图保存 + legend 安全
# =========================
def save_and_show(fig, filename):
    out = script_dir() / filename
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {out.name}")
    plt.show()

def safe_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l is None:
            continue
        s = str(l).strip()
        if s and s.lower() != "none":
            h2.append(h)
            l2.append(s)
    if h2:
        ax.legend(h2, l2, **kwargs)


# =========================
# 3) 数据加载
# =========================
def load_npy_pair():
    X_path = find_one(["X_test_seq.npy"], root=None)
    y_path = find_one(["y_test_seq.npy"], root=None)
    if X_path is None or y_path is None:
        raise FileNotFoundError("找不到 X_test_seq.npy / y_test_seq.npy（请确认与脚本同目录或子目录内）。")
    X = np.load(X_path)
    y = np.load(y_path)
    y = np.asarray(y).reshape(-1)
    return X_path, y_path, X, y


# =========================
# 4) 模型：LSTM / GRU
# =========================
class BiLSTMOptimized(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.fc(out)
        return self.sigmoid(out).squeeze(-1)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.fc(out)
        return self.sigmoid(out).squeeze(-1)


# =========================
# 5) Transformer：兼容两种 state_dict 命名
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.2, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.last_attn = None

    def forward(self, src, need_attn=False):
        if need_attn:
            attn_out, attn_w = self.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
            self.last_attn = attn_w
        else:
            attn_out = self.self_attn(src, src, src, need_weights=False)[0]
        src = self.norm1(src + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src

class TransformerLayersStyle(nn.Module):
    """layers.0.xxx 命名"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2, max_len=1000):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_attn=False):
        x = self.input_norm(self.input_proj(x))
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, need_attn=return_attn)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        prob = self.sigmoid(self.fc(pooled)).squeeze(-1)
        if return_attn:
            return prob, [layer.last_attn for layer in self.layers]
        return prob

class EncoderWithLayers(nn.Module):
    """encoder.layers.0.xxx 命名"""
    def __init__(self, num_layers, d_model, nhead, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, x, need_attn=False):
        attns = []
        for layer in self.layers:
            x = layer(x, need_attn=need_attn)
            if need_attn:
                attns.append(layer.last_attn)
        return x, attns

class TransformerEncoderStyle(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=1, dropout=0.2, max_len=1000):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.encoder = EncoderWithLayers(num_layers=num_layers, d_model=d_model, nhead=nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_attn=False):
        x = self.input_norm(self.input_proj(x))
        x = self.pos_encoding(x)
        x, attns = self.encoder(x, need_attn=return_attn)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        prob = self.sigmoid(self.fc(pooled)).squeeze(-1)
        if return_attn:
            return prob, attns
        return prob

def build_transformer_from_state_dict(sd, input_dim):
    keys = list(sd.keys())
    is_encoder_style = any(k.startswith("encoder.layers.") for k in keys)
    is_layers_style = any(k.startswith("layers.") for k in keys)

    d_model = int(sd["input_proj.weight"].shape[0]) if "input_proj.weight" in sd else 64

    if is_encoder_style:
        layer_ids = []
        for k in keys:
            if k.startswith("encoder.layers."):
                try:
                    layer_ids.append(int(k.split(".")[2]))
                except Exception:
                    pass
        num_layers = (max(layer_ids) + 1) if layer_ids else 1
    elif is_layers_style:
        layer_ids = []
        for k in keys:
            if k.startswith("layers."):
                try:
                    layer_ids.append(int(k.split(".")[1]))
                except Exception:
                    pass
        num_layers = (max(layer_ids) + 1) if layer_ids else 2
    else:
        is_encoder_style = True
        num_layers = 1

    nhead = 4

    if is_encoder_style:
        model = TransformerEncoderStyle(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                        dropout=0.2, max_len=1000)
        style = "encoder"
    else:
        model = TransformerLayersStyle(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                       dropout=0.2, max_len=1000)
        style = "layers"
    return model, {"style": style, "d_model": d_model, "nhead": nhead, "num_layers": num_layers}

def load_transformer_auto(path, input_dim):
    sd = torch.load(path, map_location="cpu")
    model, info = build_transformer_from_state_dict(sd, input_dim=input_dim)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[OK] Transformer loaded: style={info['style']} d_model={info['d_model']} nhead={info['nhead']} num_layers={info['num_layers']}")
    if missing:
        print(f"[INFO] missing keys (ok for buffers): {missing[:6]}{'...' if len(missing)>6 else ''}")
    if unexpected:
        print(f"[INFO] unexpected keys (check if many): {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")
    return model


# =========================
# 6) 预测
# =========================
def load_model_state_strict(path, model):
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

@torch.no_grad()
def predict_prob(model, X, batch_size=512, device="cpu"):
    model.to(device)
    probs = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        pb = model(xb)
        probs.append(pb.detach().cpu().numpy())
    return np.concatenate(probs, axis=0)

@torch.no_grad()
def predict_prob_and_attn(transformer_model, X, sample_index=0, device="cpu"):
    transformer_model.to(device)
    x = torch.tensor(X[sample_index:sample_index+1], dtype=torch.float32, device=device)
    out = transformer_model(x, return_attn=True)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        prob, attns = out
        return float(prob.detach().cpu().numpy()[0]), attns
    return None, None


# =========================
# 7) 训练过程图：学习率曲线
# =========================
def try_load_training_logs():
    log_path = find_one(["wandb-history.csv", "*history*.csv", "*train*log*.csv", "*metrics*.csv"], root=None)
    if log_path is None:
        return None, None
    try:
        df = pd.read_csv(log_path)
        return log_path, df
    except Exception as e:
        print(f"[WARN] 训练日志读取失败：{log_path} ({e})")
        return None, None

def plot_learning_rate_curve():
    log_path, df = try_load_training_logs()
    if df is None:
        print("[SKIP] 学习率曲线：未找到训练日志（wandb-history.csv / *history*.csv / *train*log*.csv 等）。")
        return

    lr_cols = [c for c in df.columns if "lr" in c.lower() or "learning_rate" in c.lower()]
    step_cols = [c for c in df.columns if c.lower() in ["epoch", "step", "_step", "global_step"]]
    if not lr_cols:
        print(f"[SKIP] 学习率曲线：日志里没找到lr相关列。文件：{log_path.name}")
        return

    lr_col = lr_cols[0]
    x_col = step_cols[0] if step_cols else None

    fig, ax = plt.subplots(figsize=(8, 4))
    if x_col:
        ax.plot(df[x_col].values, df[lr_col].values, label=f"LR ({lr_col})")
        ax.set_xlabel(x_col)
    else:
        ax.plot(df[lr_col].values, label=f"LR ({lr_col})")
        ax.set_xlabel("Step")

    ax.set_ylabel("Learning Rate")
    ax.set_title("训练过程：学习率曲线（Learning Rate Curve）")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, "训练过程_学习率曲线.png")


# =========================
# 8) 预测结果图：校准 / Brier分桶 / 直方图
# =========================
def plot_calibration_curve(y_true, prob, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o", label=f"{model_name} Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"预测结果：校准曲线（可靠性图）- {model_name}")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, f"预测结果_校准曲线_{model_name}.png")

def plot_brier_binned(y_true, prob, model_name, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    centers = (bins[:-1] + bins[1:]) / 2
    brier_each = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = idx == b
        counts[b] = mask.sum()
        brier_each[b] = np.mean((prob[mask] - y_true[mask]) ** 2) if counts[b] > 0 else np.nan

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(centers, brier_each, marker="o", label="Brier per bin")
    ax1.set_xlabel("Predicted probability bin center")
    ax1.set_ylabel("Brier score (MSE)")
    ax1.set_title(f"预测结果：Brier score 分桶图 - {model_name}")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(centers, counts, width=0.08, alpha=0.25, label="Count per bin")
    ax2.set_ylabel("Count")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax1.legend(h1 + h2, l1 + l2, loc="best")

    save_and_show(fig, f"预测结果_Brier分桶_{model_name}.png")

def plot_prediction_hist(prob, model_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(prob, bins=30, alpha=0.85, label=f"{model_name} prob")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"预测结果：预测分布直方图 - {model_name}")
    ax.grid(True, alpha=0.25)
    safe_legend(ax, loc="best")
    save_and_show(fig, f"预测结果_预测分布直方图_{model_name}.png")


# =========================
# 9) 序列数据：时序热力图
# =========================
def plot_sequence_heatmap(X, feature_names=None, sample_index=0, max_features=30):
    seq = X[sample_index]  # (T,F)
    T, Fdim = seq.shape
    use_F = min(Fdim, max_features)
    data = seq[:, :use_F].T

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(Fdim)]
    use_names = feature_names[:use_F]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto")
    ax.set_xlabel("Time step (point index in window)")
    ax.set_ylabel("Feature")
    ax.set_yticks(np.arange(use_F))
    ax.set_yticklabels(use_names)
    ax.set_title("序列数据：时序热力图（单样本特征随时间变化）")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Feature value")
    save_and_show(fig, "序列数据_时序热力图.png")


# =========================
# 10) 模型对比：ROC / PR
# =========================
def plot_multi_model_roc(y_true, prob_dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, prob in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("模型对比：多模型 ROC 同图（ROC Curves）")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, "模型对比_多模型ROC同图.png")

def plot_multi_model_pr(y_true, prob_dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, prob in prob_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("模型对比：多模型 PR 曲线（Precision-Recall Curves）")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, "模型对比_多模型PR曲线.png")


# =========================
# 11) 模型解释：SHAP / Attention
# =========================
def plot_shap_importance(model, X, feature_names=None, model_name="GRU", max_samples=200):
    try:
        import shap
    except Exception:
        print("[SKIP] SHAP importance：未安装 shap（pip install shap）。")
        return

    n = X.shape[0]
    m = min(n, max_samples)
    idx = np.random.RandomState(42).choice(n, size=m, replace=False)
    Xs = X[idx]
    Xflat = Xs.reshape(m, -1)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[-1])]
    T, Fdim = X.shape[1], X.shape[2]
    flat_names = [f"{feature_names[f]}@t{t}" for t in range(T) for f in range(Fdim)]

    def f_predict(x_flat):
        x = x_flat.reshape((-1, T, Fdim))
        with torch.no_grad():
            xb = torch.tensor(x, dtype=torch.float32)
            return model(xb).detach().cpu().numpy()

    bg = Xflat[:min(50, m)]
    print(f"[INFO] SHAP：开始计算（{model_name}），样本={m}，特征维度={Xflat.shape[1]}（可能较慢）...")
    explainer = shap.KernelExplainer(f_predict, bg)
    shap_values = explainer.shap_values(Xflat, nsamples=200)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    imp = np.mean(np.abs(shap_values), axis=0)
    k = 25
    top_idx = np.argsort(imp)[-k:][::-1]
    top_imp = imp[top_idx]
    top_names = [flat_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(k)[::-1], top_imp[::-1], label="mean(|SHAP|)")
    ax.set_yticks(range(k)[::-1])
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"模型解释：SHAP importance（Top {k}）- {model_name}")
    ax.grid(True, axis="x", alpha=0.25)
    safe_legend(ax, loc="best")
    save_and_show(fig, f"模型解释_SHAP_importance_{model_name}.png")

def plot_attention_heatmap(transformer_model, X, sample_index=0, device="cpu"):
    prob, attns = predict_prob_and_attn(transformer_model, X, sample_index=sample_index, device=device)
    if attns is None or len(attns) == 0:
        print("[SKIP] Attention可视化：未捕获attention。")
        return
    last = attns[-1]
    if last is None:
        print("[SKIP] Attention可视化：last_attn为空。")
        return

    w = last.detach().cpu().numpy()
    if w.ndim == 4:
        w_mean = w[0].mean(axis=0)
    elif w.ndim == 3:
        w_mean = w[0]
    else:
        print(f"[SKIP] Attention权重shape异常：{w.shape}")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(w_mean, aspect="auto")
    ax.set_xlabel("Key position (time step)")
    ax.set_ylabel("Query position (time step)")
    ax.set_title("模型解释：Attention 可视化（最后一层，平均所有头）")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight")
    save_and_show(fig, "模型解释_Attention可视化.png")


# =========================
# 12) 动量分析（基于 prob 重算）
# =========================
def ema(x, alpha=0.15):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def robust_scale(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / mad

def compute_momentum_from_prob(prob, alpha=0.15, detrend=True):
    p = np.asarray(prob, dtype=float)
    m_ema = ema(p, alpha=alpha)
    m = robust_scale(m_ema)
    if detrend:
        trend = pd.Series(m).rolling(80, min_periods=1, center=True).mean().values
        m = m - trend
    return m_ema, m

def compute_turning_points(momentum, delta_thresh=0.15):
    m = np.asarray(momentum)
    dm = np.diff(m, prepend=m[0])
    sign = np.sign(dm)
    turn = []
    for i in range(2, len(m)):
        if sign[i-1] == 0 or sign[i] == 0:
            continue
        if sign[i] != sign[i-1] and abs(dm[i]) >= delta_thresh:
            turn.append(i)
    return np.array(turn, dtype=int)

def plot_momentum_phase_plane(momentum, title_suffix="", fname_suffix=""):
    m = np.asarray(momentum)
    dm = np.diff(m, prepend=m[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m, dm, s=14, alpha=0.8, label="(momentum, slope)")
    ax.set_xlabel("Momentum")
    ax.set_ylabel("Slope ΔMomentum")
    ax.set_title(f"动量分析：动量相图（Phase-plane）{title_suffix}")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, f"动量分析_动量相图{fname_suffix}.png")

def plot_momentum_turning_from_prob(prob, alpha=0.15, delta_thresh=0.15, tag="GRU"):
    _, m = compute_momentum_from_prob(prob, alpha=alpha, detrend=True)
    turns = compute_turning_points(m, delta_thresh=delta_thresh)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prob, alpha=0.25, label="Prob (raw)")
    ax.plot(m, label=f"Momentum (EMA+scaled+detrend, alpha={alpha})")
    if len(turns) > 0:
        ax.scatter(turns, m[turns], marker="x", s=70, label="Turning points")

    ax.set_xlabel("Point index")
    ax.set_ylabel("Value")
    ax.set_title(f"动量分析：基于预测概率的动量与转折点 - {tag}")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, f"动量分析_Prob动量转折点_{tag}.png")

    plot_momentum_phase_plane(m, title_suffix=f"（基于预测概率 {tag}）", fname_suffix=f"_prob_{tag}")

def plot_momentum_slope_from_prob(prob, alpha=0.15, tag="GRU"):
    _, m = compute_momentum_from_prob(prob, alpha=alpha, detrend=True)
    slope = np.diff(m, prepend=m[0])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(slope, label="Slope ΔMomentum")
    ax.axhline(0, linestyle="--", label="Zero slope")
    ax.set_xlabel("Point index")
    ax.set_ylabel("ΔMomentum")
    ax.set_title(f"动量分析：基于预测概率的动量坡度（斜率）- {tag}")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, f"动量分析_Prob动量坡度_{tag}.png")

def plot_momentum_transition_from_prob(prob, alpha=0.15, n_states=5, tag="GRU"):
    _, m = compute_momentum_from_prob(prob, alpha=alpha, detrend=False)
    qs = np.quantile(m, np.linspace(0, 1, n_states+1))
    states = np.digitize(m, qs[1:-1], right=True)

    trans = np.zeros((n_states, n_states), dtype=float)
    for i in range(1, len(states)):
        trans[states[i-1], states[i]] += 1

    row_sum = trans.sum(axis=1, keepdims=True)
    trans_p = np.where(row_sum > 0, trans / row_sum, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(trans_p, aspect="auto")
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title(f"动量分析：动量分段状态转移矩阵（基于预测概率）- {tag}")
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels([f"S{i}" for i in range(n_states)])
    ax.set_yticklabels([f"S{i}" for i in range(n_states)])

    for i in range(n_states):
        for j in range(n_states):
            ax.text(j, i, f"{trans_p[i,j]:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Transition probability")
    save_and_show(fig, f"动量分析_Prob动量状态转移矩阵_{tag}.png")

def plot_momentum_compare_with_csv(prob, tag="GRU"):
    m_path = find_one(["momentum_match_*.csv"], root=None)
    if m_path is None:
        print("[SKIP] 动量对比：未找到 momentum_match_*.csv")
        return
    try:
        df = pd.read_csv(m_path)
    except Exception:
        print("[SKIP] 动量对比：csv读取失败")
        return

    col_prob = "p1_win_prob" if "p1_win_prob" in df.columns else None
    col_m = "momentum_score" if "momentum_score" in df.columns else None
    if col_prob is None and col_m is None:
        print("[SKIP] 动量对比：csv里没有 p1_win_prob/momentum_score")
        return

    L = min(len(prob), len(df))
    p = np.asarray(prob)[:L]
    _, m_scaled = compute_momentum_from_prob(p, alpha=0.15, detrend=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(p, alpha=0.25, label="Model prob (raw)")
    ax.plot(m_scaled, label="Recomputed momentum (scaled+detrend)")
    if col_prob is not None:
        ax.plot(df[col_prob].values[:L], alpha=0.35, label="CSV p1_win_prob")
    if col_m is not None:
        ax.plot(df[col_m].values[:L], alpha=0.35, label="CSV momentum_score")

    ax.set_xlabel("Point index (approx aligned)")
    ax.set_ylabel("Value")
    ax.set_title(f"动量分析：CSV动量 vs 重算动量 对比 - {tag}")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, f"动量分析_CSV对比_重算动量_{tag}.png")


# =========================
# 13) 赛场层级：胜率时间演化
# =========================
def plot_match_winprob_evolution(prob, window=50, tag="GRU"):
    p = np.asarray(prob)
    if len(p) < 5:
        print("[SKIP] 比赛胜率时间演化：样本太少。")
        return
    w = min(window, len(p))
    roll = pd.Series(p).rolling(w, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(p, alpha=0.35, label="Point win prob (raw)")
    ax.plot(roll, label=f"Rolling mean (window={w})")
    ax.set_xlabel("Point index (test set order)")
    ax.set_ylabel("P(Player1 wins next point)")
    ax.set_title(f"赛场层级：比赛胜率时间演化（Win Probability Over Time）- {tag}")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, f"赛场层级_比赛胜率时间演化_{tag}.png")


# =========================
# 14) 关键分：breakpoint-only（需要原始逐分csv）
# =========================
def is_break_point(p1_score, p2_score, server):
    def to_label(s):
        if isinstance(s, (int, np.integer)):
            mapping = {0:"0", 1:"15", 2:"30", 3:"40", 4:"AD"}
            return mapping.get(int(s), str(s))
        s = str(s).strip().upper()
        if s in ["0", "00"]:
            return "0"
        if s in ["15"]:
            return "15"
        if s in ["30"]:
            return "30"
        if s in ["40"]:
            return "40"
        if s in ["AD", "A"]:
            return "AD"
        return s

    p1 = to_label(p1_score)
    p2 = to_label(p2_score)

    try:
        server = int(server)
    except Exception:
        return False

    if server == 1:
        receiver_score, server_score = p2, p1
    elif server == 2:
        receiver_score, server_score = p1, p2
    else:
        return False

    if receiver_score == "AD":
        return True
    if receiver_score == "40" and server_score in ["0", "15", "30"]:
        return True
    return False

def plot_breakpoint_only(prob, tag="GRU"):
    raw_path = find_one(["2024_Wimbledon_featured_matches.csv"], root=None)
    if raw_path is None:
        print("[SKIP] breakpoint-only：找不到 2024_Wimbledon_featured_matches.csv")
        return

    df = pd.read_csv(raw_path)
    need_cols = ["p1_score", "p2_score", "server"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        print(f"[SKIP] breakpoint-only：原始csv缺少列 {miss}")
        return

    bp = []
    for _, row in df.iterrows():
        bp.append(is_break_point(row["p1_score"], row["p2_score"], row["server"]))
    bp = np.array(bp, dtype=bool)

    L = min(len(prob), len(bp))
    p = np.asarray(prob)[:L]
    bp = bp[:L]
    idx = np.where(bp)[0]
    if len(idx) == 0:
        print("[SKIP] breakpoint-only：未识别到break point（可能数据列格式不同）。")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(p, alpha=0.25, label="Win prob (raw)")
    ax.scatter(idx, p[idx], s=28, label="Break points")
    ax.set_xlabel("Point index (approx aligned)")
    ax.set_ylabel("P(Player1 wins next point)")
    ax.set_title(f"关键分：breakpoint-only 可视化（Break Points Only）- {tag}")
    ax.grid(True, alpha=0.3)
    safe_legend(ax, loc="best")
    save_and_show(fig, f"关键分_breakpoint_only_{tag}.png")


# =========================
# 15) Dash（自动启动，无需参数）
# =========================
def dash_run_compat(app, host="127.0.0.1", port=8050, debug=False):
    """Dash 新旧版本兼容：Dash>=3 用 app.run；老版本用 app.run_server"""
    if hasattr(app, "run"):
        return app.run(host=host, port=port, debug=debug)
    return app.run_server(host=host, port=port, debug=debug)

def launch_dash_dashboard(X_test, y_test, prob_dict, transformer_model, device="cpu", feature_names=None):
    try:
        from dash import Dash, dcc, html, Input, Output
        import plotly.graph_objects as go
    except Exception:
        print("[SKIP] Dash 仪表盘：未安装 dash/plotly（pip install dash plotly）。")
        return

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_test.shape[-1])]

    N = len(y_test)
    models = list(prob_dict.keys())
    default_model = "GRU" if "GRU" in models else models[0]

    def clip_range(a, b):
        a = int(max(0, min(a, N-1)))
        b = int(max(a+1, min(b, N)))
        return a, b

    def fig_roc_all():
        fig = go.Figure()
        for name, prob in prob_dict.items():
            fpr, tpr, _ = roc_curve(y_test, prob)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc(fpr,tpr):.3f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
        fig.update_layout(title="模型对比：多模型 ROC", template="plotly_white", height=520,
                          xaxis_title="FPR", yaxis_title="TPR")
        return fig

    def fig_pr_all():
        fig = go.Figure()
        for name, prob in prob_dict.items():
            prec, rec, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{name} (AP={ap:.3f})"))
        fig.update_layout(title="模型对比：多模型 PR", template="plotly_white", height=520,
                          xaxis_title="Recall", yaxis_title="Precision")
        return fig

    def fig_calib(y, p, name):
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=name))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash")))
        fig.update_layout(title=f"预测结果：校准曲线 - {name}", template="plotly_white", height=420,
                          xaxis_title="Mean predicted probability", yaxis_title="Fraction positives")
        return fig

    def fig_hist(p, name):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=p, nbinsx=30, name=name))
        fig.update_layout(title=f"预测结果：预测分布直方图 - {name}", template="plotly_white", height=420,
                          xaxis_title="Predicted probability", yaxis_title="Count")
        return fig

    def fig_winprob(p, window, name):
        w = int(max(2, min(int(window), len(p))))
        roll = pd.Series(p).rolling(w, min_periods=1).mean().values
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=p, mode="lines", name="raw", opacity=0.25))
        fig.add_trace(go.Scatter(y=roll, mode="lines", name=f"rolling(w={w})"))
        fig.update_layout(title=f"赛场层级：胜率时间演化 - {name}", template="plotly_white", height=420,
                          xaxis_title="index", yaxis_title="P(win next point)")
        return fig

    def fig_momentum(p, alpha, delta, name):
        _, m = compute_momentum_from_prob(p, alpha=float(alpha), detrend=True)
        turns = compute_turning_points(m, delta_thresh=float(delta))
        slope = np.diff(m, prepend=m[0])
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=p, mode="lines", name="prob(raw)", opacity=0.22))
        fig.add_trace(go.Scatter(y=m, mode="lines", name=f"momentum(alpha={alpha:.2f})"))
        if len(turns) > 0:
            fig.add_trace(go.Scatter(x=turns, y=m[turns], mode="markers", name="turning",
                                     marker=dict(symbol="x", size=10)))
        fig.add_trace(go.Scatter(y=slope, mode="lines", name="slope", opacity=0.35))
        fig.update_layout(title=f"动量分析：动量/转折点/坡度 - {name}", template="plotly_white", height=520,
                          xaxis_title="index", yaxis_title="value")
        return fig

    def fig_seq_heatmap(sample_index, max_features=30):
        import plotly.graph_objects as go
        seq = X_test[sample_index]
        T, Fdim = seq.shape
        use_F = min(Fdim, int(max_features))
        z = seq[:, :use_F].T
        fig = go.Figure(data=go.Heatmap(z=z, x=list(range(T)), y=feature_names[:use_F], colorbar=dict(title="Value")))
        fig.update_layout(title=f"序列数据：时序热力图（样本 {sample_index}）", template="plotly_white", height=560)
        return fig

    def fig_attention(sample_index):
        import plotly.graph_objects as go
        _, attns = predict_prob_and_attn(transformer_model, X_test, sample_index=sample_index, device=device)
        if attns is None or len(attns) == 0 or attns[-1] is None:
            fig = go.Figure()
            fig.update_layout(title="模型解释：Attention（未捕获/不支持）", template="plotly_white", height=420)
            return fig
        w = attns[-1].detach().cpu().numpy()
        w_mean = w[0].mean(axis=0) if w.ndim == 4 else w[0]
        fig = go.Figure(data=go.Heatmap(z=w_mean, colorbar=dict(title="Weight")))
        fig.update_layout(title=f"模型解释：Attention（最后一层平均）样本 {sample_index}", template="plotly_white", height=560)
        return fig

    app = Dash(__name__)
    app.title = "Tennis Momentum Dashboard"

    app.layout = html.Div(style={"maxWidth":"1200px","margin":"0 auto","padding":"16px"}, children=[
        html.H2("网球比赛动量分析与预测结果展示仪表盘（Dash）"),
        html.Div("默认流程：先生成 PNG 图并弹窗显示，然后自动启动此仪表盘。", style={"color":"#444"}),
        html.Hr(),

        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}, children=[
            html.Div(style={"border":"1px solid #eee","borderRadius":"12px","padding":"12px","background":"#fafafa"}, children=[
                html.Div("模型选择（Model）", style={"fontWeight":"600"}),
                dcc.Dropdown(id="model", options=[{"label":m, "value":m} for m in models],
                             value=default_model, clearable=False),

                html.Div("区间截取（只看某段更容易观察动量）", style={"fontWeight":"600","marginTop":"10px"}),
                dcc.RangeSlider(id="rng", min=0, max=N, step=1, value=[0, min(N, 600)],
                                marks={0:"0", int(N/2):f"{int(N/2)}", N:str(N)}),

                html.Div("Rolling window（胜率演化）", style={"fontWeight":"600","marginTop":"10px"}),
                dcc.Slider(id="win_w", min=5, max=200, step=1, value=50,
                           marks={10:"10",50:"50",100:"100",200:"200"}),
            ]),
            html.Div(style={"border":"1px solid #eee","borderRadius":"12px","padding":"12px","background":"#fafafa"}, children=[
                html.Div("动量参数（Momentum）", style={"fontWeight":"600"}),
                html.Div("alpha（EMA）", style={"marginTop":"8px"}),
                dcc.Slider(id="alpha", min=0.03, max=0.40, step=0.01, value=0.15,
                           marks={0.05:"0.05",0.15:"0.15",0.30:"0.30",0.40:"0.40"}),
                html.Div("转折阈值 Δ", style={"marginTop":"8px"}),
                dcc.Slider(id="delta", min=0.05, max=0.60, step=0.01, value=0.15,
                           marks={0.10:"0.10",0.15:"0.15",0.30:"0.30",0.50:"0.50"}),

                html.Hr(),
                html.Div("序列/注意力采样索引", style={"fontWeight":"600"}),
                dcc.Slider(id="sample", min=0, max=min(200, X_test.shape[0]-1), step=1, value=0,
                           marks={0:"0", 50:"50", 100:"100", 150:"150", 200:"200"}),
            ]),
        ]),

        html.Hr(),
        dcc.Tabs(value="tab_overview", id="tabs", children=[
            dcc.Tab(label="总览（ROC/PR）", value="tab_overview"),
            dcc.Tab(label="单模型（校准/分布/胜率/动量）", value="tab_single"),
            dcc.Tab(label="序列与解释（热力图/Attention）", value="tab_explain"),
        ]),
        html.Div(id="content", style={"marginTop":"12px"})
    ])

    @app.callback(
        Output("content", "children"),
        Input("tabs", "value"),
        Input("model", "value"),
        Input("rng", "value"),
        Input("win_w", "value"),
        Input("alpha", "value"),
        Input("delta", "value"),
        Input("sample", "value"),
    )
    def render(tab, model, rng, win_w, alpha, delta, sample):
        from dash import html
        import plotly.graph_objects as go

        a, b = clip_range(rng[0], rng[1])
        p_all = prob_dict[model]
        p = p_all[a:b]
        y = y_test[a:b]

        if tab == "tab_overview":
            return html.Div([
                dcc.Graph(figure=fig_roc_all()),
                dcc.Graph(figure=fig_pr_all()),
            ])

        if tab == "tab_single":
            return html.Div([
                html.Div(f"当前区间：[{a}, {b})  长度={len(p)}", style={"color":"#555"}),
                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}, children=[
                    dcc.Graph(figure=fig_calib(y, p, model)),
                    dcc.Graph(figure=fig_hist(p, model)),
                ]),
                dcc.Graph(figure=fig_winprob(p, win_w, model)),
                dcc.Graph(figure=fig_momentum(p, alpha, delta, model)),
            ])

        if tab == "tab_explain":
            s = int(sample)
            return html.Div([
                dcc.Graph(figure=fig_seq_heatmap(s, max_features=30)),
                dcc.Graph(figure=fig_attention(s)),
            ])

        return html.Div("unknown tab")

    # 端口自动切换
    port = 8050
    while port < 8060:
        try:
            print(f"\n[DASH] 启动仪表盘：http://127.0.0.1:{port} （按 Ctrl+C 停止）")
            dash_run_compat(app, host="127.0.0.1", port=port, debug=False)
            break
        except OSError:
            port += 1
    if port >= 8060:
        print("[ERROR] 8050-8059 端口都被占用，请关闭占用端口的程序后重试。")


# =========================
# 16) 主流程（先出图，再自动启动 Dash）
# =========================
def main():
    setup_matplotlib()

    print("=== [1/4] 载入测试数据（X_test_seq / y_test_seq） ===")
    X_path, y_path, X_test, y_test = load_npy_pair()
    print(f"[OK] X: {X_path.name} shape={X_test.shape}")
    print(f"[OK] y: {y_path.name} shape={y_test.shape}")

    input_dim = X_test.shape[-1]
    feature_names = [f"f{i}" for i in range(input_dim)]

    print("\n=== [2/4] 加载模型（LSTM / GRU / Transformer）并预测概率 ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    lstm_path = find_one(["model_lstm_optimized.pt", "lstm_model.pt"], root=None)
    gru_path  = find_one(["model_gru.pt"], root=None)
    tf_path   = find_one(["model_transformer_v2.pt", "model_transformer.pt"], root=None)

    if lstm_path is None or gru_path is None or tf_path is None:
        print("[ERROR] 未找到全部三种模型文件：")
        print(f"  LSTM: {lstm_path}")
        print(f"  GRU : {gru_path}")
        print(f"  TF  : {tf_path}")
        sys.exit(1)

    lstm = BiLSTMOptimized(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.4)
    gru  = GRUModel(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.5)

    lstm = load_model_state_strict(lstm_path, lstm)
    print(f"[OK] LSTM loaded: {lstm_path.name}")

    gru = load_model_state_strict(gru_path, gru)
    print(f"[OK] GRU loaded: {gru_path.name}")

    transformer = load_transformer_auto(tf_path, input_dim=input_dim)

    prob_lstm = predict_prob(lstm, X_test, device=device)
    prob_gru  = predict_prob(gru, X_test, device=device)
    prob_tf   = predict_prob(transformer, X_test, device=device)

    prob_dict = {"LSTM": prob_lstm, "GRU": prob_gru, "Transformer": prob_tf}
    print("[OK] 预测完成：", {k: v.shape for k, v in prob_dict.items()})

    print("\n=== [3/4] 生成所有图（保存到脚本同目录 + 弹窗显示） ===")

    plot_learning_rate_curve()

    for name, prob in prob_dict.items():
        plot_calibration_curve(y_test, prob, name)
        plot_brier_binned(y_test, prob, name, n_bins=10)
        plot_prediction_hist(prob, name)

    plot_sequence_heatmap(X_test, feature_names=feature_names, sample_index=0, max_features=30)

    plot_multi_model_roc(y_test, prob_dict)
    plot_multi_model_pr(y_test, prob_dict)

    plot_shap_importance(gru, X_test, feature_names=feature_names, model_name="GRU", max_samples=200)

    plot_attention_heatmap(transformer, X_test, sample_index=0, device=device)

    plot_momentum_turning_from_prob(prob_gru, alpha=0.15, delta_thresh=0.15, tag="GRU")
    plot_momentum_slope_from_prob(prob_gru, alpha=0.15, tag="GRU")
    plot_momentum_transition_from_prob(prob_gru, alpha=0.15, n_states=5, tag="GRU")
    plot_momentum_compare_with_csv(prob_gru, tag="GRU")

    plot_match_winprob_evolution(prob_gru, window=50, tag="GRU")

    plot_breakpoint_only(prob_gru, tag="GRU")

    print("\n=== [4/4] PNG 生成结束：自动启动 Dash ===")
    print("[DONE] PNG 已保存到：", script_dir())

    launch_dash_dashboard(
        X_test=X_test,
        y_test=y_test,
        prob_dict=prob_dict,
        transformer_model=transformer,
        device=device,
        feature_names=feature_names
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
