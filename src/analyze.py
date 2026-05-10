"""
analyze.py

学習済みモデルの特徴量解析スクリプト。

機能:
    1. t-SNE による特徴空間の可視化
       - 各サンプルを分類ヘッド直前の特徴ベクトルに変換して 2D に圧縮
       - クラスごとに色分けして散布図を描画
    2. Grad-CAM によるスペクトログラム上の注目領域可視化
       - CNN 向け: 最終 Conv 層の勾配で活性化マップを生成
       - ViT/AST 向け: アテンション重みを用いた Attention Rollout で代替

実行例:
    python analyze.py                      # ViT で解析（デフォルト）
    python analyze.py --model cnn          # 軽量 CNN で解析
    python analyze.py --model ast          # AST で解析
    python analyze.py --model vit --fold 0 # fold 0 のモデルで解析
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset import (
    GraspSoundDataset,
    get_transforms,
    collect_files_and_labels,
    get_kfold_splits,
)
from models import build_model, SimpleCNN, ViTClassifier, ASTClassifier


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_name, num_classes, model_path, device):
    """学習済み重みをロードして eval モードにする。"""
    model = build_model(model_name, num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# t-SNE 可視化
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device):
    """
    全サンプルの特徴ベクトルとラベルを抽出する。

    Args:
        model: get_features() メソッドを持つ nn.Module
        loader: DataLoader（全データ）
        device: torch.device

    Returns:
        tuple: (features np.ndarray shape(N, D), labels np.ndarray shape(N,))
    """
    feats_list, labels_list = [], []
    for specs, labels in loader:
        specs = specs.to(device)
        feats = model.get_features(specs)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())
    return np.concatenate(feats_list), np.concatenate(labels_list)


def plot_tsne(features, labels, class_names, title, save_path, perplexity=15):
    """
    t-SNE で特徴ベクトルを 2D に圧縮して散布図を描画する。

    Args:
        features (np.ndarray): shape (N, D)
        labels (np.ndarray): shape (N,)
        class_names (list[str]): クラス名リスト
        title (str): タイトル
        save_path (str): 保存先パス
        perplexity (int): t-SNE の perplexity（サンプル数が少ない場合は小さめに）
    """
    # サンプル数が perplexity より少ない場合は調整
    perplexity = min(perplexity, len(features) - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    emb  = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[cmap(cls_idx)], label=cls_name,
            s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
        )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  t-SNE 図を保存: {save_path}")


def run_tsne(model, all_file_paths, all_labels, class_names,
             config, device, save_path, model_name, fold_idx):
    """
    全データで t-SNE を実行して保存する。

    Args:
        model: 学習済みモデル
        all_file_paths (list[str]): 全スペクトログラムパス
        all_labels (list[int]): 全ラベル
        class_names (list[str]): クラス名リスト
        config (dict): config
        device: torch.device
        save_path (str): 保存先パス
        model_name (str): モデル名（タイトル用）
        fold_idx (int): fold 番号（タイトル用）
    """
    img_size   = config["melspec"].get("img_size", 224)
    batch_size = config["train"]["batch_size"]

    _, val_transform = get_transforms(img_size)
    dataset = GraspSoundDataset(all_file_paths, all_labels, transform=val_transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    features, labels = extract_features(model, loader, device)
    title = f"{model_name.upper()} - Fold {fold_idx} Feature Space (t-SNE)"
    plot_tsne(features, labels, class_names, title, save_path)


# ---------------------------------------------------------------------------
# Grad-CAM（CNN 向け）
# ---------------------------------------------------------------------------

class GradCAM:
    """
    CNN の最終 Conv 層に対する Grad-CAM を計算するクラス。

    使い方:
        cam = GradCAM(model, target_layer=model.features[-3])
        heatmap = cam(input_tensor)  # shape (H, W)
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        """
        Grad-CAM ヒートマップを計算する。

        Args:
            x (torch.Tensor): shape (1, C, H, W)
            class_idx (int | None): 対象クラス（None の場合は予測クラスを使用）

        Returns:
            np.ndarray: shape (H_feat, W_feat) の正規化済みヒートマップ
        """
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        # Global Average Pooling で各チャンネルの重みを計算
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = cam[0, 0].cpu().numpy()

        # [0, 1] に正規化
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam


def plot_gradcam(spec_np, cam, class_name, pred_name, title, save_path):
    """
    スペクトログラムに Grad-CAM ヒートマップを重ねて描画する。

    Args:
        spec_np (np.ndarray): shape (H, W)  スペクトログラム（L チャンネル）
        cam (np.ndarray): shape (h, w) Grad-CAM ヒートマップ
        class_name (str): 正解クラス名
        pred_name (str): 予測クラス名
        title (str): タイトル
        save_path (str): 保存先パス
    """
    # CAM をスペクトログラムのサイズにリサイズ
    cam_resized = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(
        cam_resized, size=spec_np.shape, mode="bilinear", align_corners=False
    )[0, 0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(spec_np, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Mel-spectrogram (L ch)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cam_resized, aspect="auto", origin="lower", cmap="jet")
    axes[1].set_title("Grad-CAM", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(spec_np, aspect="auto", origin="lower", cmap="viridis")
    axes[2].imshow(cam_resized, aspect="auto", origin="lower",
                   cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    correct = "✓" if class_name == pred_name else "✗"
    fig.suptitle(
        f"{title}\nTrue: {class_name}  Pred: {pred_name}  {correct}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM 図を保存: {save_path}")


def run_gradcam_cnn(model, file_paths, labels, class_names,
                    config, device, results_dir, fold_idx, n_samples=7):
    """
    CNN モデルに対して Grad-CAM を実行する。
    クラスごとに 1 サンプルずつ可視化する。

    Args:
        model (SimpleCNN): 学習済み CNN
        file_paths (list[str]): スペクトログラムパスリスト
        labels (list[int]): ラベルリスト
        class_names (list[str]): クラス名リスト
        config (dict): config
        device: torch.device
        results_dir (str): 保存先ディレクトリ
        fold_idx (int): fold 番号
        n_samples (int): 可視化するサンプル数（デフォルト: クラス数分）
    """
    img_size = config["melspec"].get("img_size", 224)
    _, val_transform = get_transforms(img_size)

    # 最終 Conv 層（features の最後の Conv2d）を取得
    target_layer = None
    for m in reversed(list(model.features.modules())):
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m
            break

    if target_layer is None:
        print("  [警告] Conv2d 層が見つかりません。Grad-CAM をスキップします。")
        return

    grad_cam = GradCAM(model, target_layer)
    os.makedirs(results_dir, exist_ok=True)

    # クラスごとに最初の 1 サンプルを選ぶ
    shown = set()
    for fp, lbl in zip(file_paths, labels):
        if lbl in shown or lbl >= n_samples:
            continue
        shown.add(lbl)

        spec_np = np.load(fp).astype(np.float32)    # (3, H, W)
        spec_t  = torch.from_numpy(spec_np)
        if val_transform:
            spec_t = val_transform(spec_t)
        x = spec_t.unsqueeze(0).to(device)

        cam = grad_cam(x, class_idx=lbl)

        # 逆正規化して [0,1] に戻す（L チャンネルのみ可視化）
        spec_vis = spec_np[0]  # (H, W)

        pred_idx  = grad_cam.model(x).argmax(dim=1).item()
        pred_name = class_names[pred_idx]
        true_name = class_names[lbl]

        save_path = os.path.join(results_dir, f"gradcam_fold{fold_idx}_class{lbl}.png")
        plot_gradcam(
            spec_vis, cam, true_name, pred_name,
            title=f"CNN Grad-CAM - Fold {fold_idx}",
            save_path=save_path,
        )


# ---------------------------------------------------------------------------
# Attention Rollout（ViT / AST 向け）
# ---------------------------------------------------------------------------

def attention_rollout_vit(model, x, device):
    """
    ViT の全アテンション層の重みを掛け合わせて [CLS] トークンの
    注目マップを計算する（Attention Rollout）。

    Args:
        model (ViTClassifier): 学習済み ViT
        x (torch.Tensor): shape (1, 3, H, W)
        device: torch.device

    Returns:
        np.ndarray: shape (14, 14)  各パッチへの注目スコア（224px → 16px patches → 14x14）
    """
    attn_maps = []

    def pre_hook_fn(module, args, kwargs):
        # torchvision の EncoderBlock は need_weights=False で呼ぶため、
        # Attention Rollout 用に head ごとの attention weight を返す設定へ差し替える。
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return args, kwargs

    def hook_fn(module, input, output):
        # MultiheadAttention の出力は (attn_output, attn_weights)。
        attn_weights = output[1] if isinstance(output, tuple) else output
        if attn_weights is None:
            raise RuntimeError("Attention weights were not returned from MultiheadAttention.")
        # attn_weights: (B, num_heads, N, N)
        attn_maps.append(attn_weights.detach().cpu())

    hooks = []
    for layer in model.vit.encoder.layers:
        hooks.append(layer.self_attention.register_forward_pre_hook(pre_hook_fn, with_kwargs=True))
        hooks.append(layer.self_attention.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            model(x.to(device))
    finally:
        for h in hooks:
            h.remove()

    if not attn_maps:
        raise RuntimeError("No attention maps were captured.")

    # Rollout: 全層のアテンション行列を掛け合わせる
    # shape: (B, num_heads, N, N) → head 方向に平均
    result = torch.eye(attn_maps[0].shape[-1])
    for attn in attn_maps:
        attn_avg = attn[0].mean(dim=0)  # (N, N)
        # Residual connection を考慮（0.5 * attn + 0.5 * I）
        attn_avg = 0.5 * attn_avg + 0.5 * torch.eye(attn_avg.shape[-1])
        attn_avg /= attn_avg.sum(dim=-1, keepdim=True)
        result = torch.mm(attn_avg, result)

    # [CLS] トークン（index 0）から各パッチへの注目スコア
    cls_attn = result[0, 1:].numpy()
    num_patches = int(cls_attn.shape[0] ** 0.5)
    return cls_attn.reshape(num_patches, num_patches)


def plot_attention_map(spec_np, attn_map, class_name, pred_name, title, save_path):
    """
    スペクトログラムにアテンションマップを重ねて描画する。

    Args:
        spec_np (np.ndarray): shape (H, W) スペクトログラム（L チャンネル）
        attn_map (np.ndarray): shape (P, P) アテンションマップ
        class_name (str): 正解クラス名
        pred_name (str): 予測クラス名
        title (str): タイトル
        save_path (str): 保存先パス
    """
    # アテンションマップをスペクトログラムサイズにリサイズ
    attn_t = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()
    attn_resized = F.interpolate(
        attn_t, size=spec_np.shape, mode="bilinear", align_corners=False
    )[0, 0].numpy()
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(spec_np, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Mel-spectrogram (L ch)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(attn_norm, aspect="auto", origin="lower", cmap="hot")
    axes[1].set_title("Attention Rollout", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(spec_np, aspect="auto", origin="lower", cmap="viridis")
    axes[2].imshow(attn_norm, aspect="auto", origin="lower", cmap="hot", alpha=0.6)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    correct = "✓" if class_name == pred_name else "✗"
    fig.suptitle(
        f"{title}\nTrue: {class_name}  Pred: {pred_name}  {correct}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Attention Rollout 図を保存: {save_path}")


def run_attention_rollout(model, model_name, file_paths, labels, class_names,
                          config, device, results_dir, fold_idx):
    """
    ViT/AST モデルに対して Attention Rollout を実行する。
    クラスごとに 1 サンプルずつ可視化する。

    Args:
        model: 学習済み ViT/AST
        model_name (str): "vit" or "ast"
        file_paths (list[str]): スペクトログラムパスリスト
        labels (list[int]): ラベルリスト
        class_names (list[str]): クラス名リスト
        config (dict): config
        device: torch.device
        results_dir (str): 保存先ディレクトリ
        fold_idx (int): fold 番号
    """
    if model_name not in ("vit",):
        print(f"  Attention Rollout は現在 ViT のみ対応。{model_name} はスキップします。")
        return

    img_size = config["melspec"].get("img_size", 224)
    _, val_transform = get_transforms(img_size)
    os.makedirs(results_dir, exist_ok=True)

    shown = set()
    for fp, lbl in zip(file_paths, labels):
        if lbl in shown:
            continue
        shown.add(lbl)

        spec_np = np.load(fp).astype(np.float32)
        spec_t  = torch.from_numpy(spec_np)
        if val_transform:
            spec_t = val_transform(spec_t)
        x = spec_t.unsqueeze(0)

        attn_map = attention_rollout_vit(model, x, device)

        with torch.no_grad():
            pred_idx = model(x.to(device)).argmax(dim=1).item()
        pred_name = class_names[pred_idx]
        true_name = class_names[lbl]

        save_path = os.path.join(results_dir, f"attn_rollout_fold{fold_idx}_class{lbl}.png")
        plot_attention_map(
            spec_np[0], attn_map, true_name, pred_name,
            title=f"ViT Attention Rollout - Fold {fold_idx}",
            save_path=save_path,
        )


# ---------------------------------------------------------------------------
# メインエントリポイント
# ---------------------------------------------------------------------------

def main(model_name="vit", fold_idx=0, config_path="config.yaml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    spec_dir    = config["paths"]["spectrogram_dir"]
    model_dir   = os.path.join(config["paths"]["model_dir"], model_name)
    results_dir = os.path.join(config["paths"].get("results_dir", "results"), model_name, "analyze")
    objects     = config["objects"]
    n_splits    = config["train"]["n_splits"]
    seed        = config["train"].get("seed", 42)
    num_classes = len(objects)

    file_paths, labels, class_names = collect_files_and_labels(spec_dir, objects)
    print(f"総サンプル数: {len(file_paths)} ({len(class_names)} クラス)")

    splits   = get_kfold_splits(file_paths, labels, n_splits=n_splits, random_state=seed)
    _, val_idx = splits[fold_idx]

    model_path = os.path.join(model_dir, f"best_fold{fold_idx}.pt")
    if not os.path.exists(model_path):
        print(f"モデルが見つかりません: {model_path}")
        return

    model = load_model(model_name, num_classes, model_path, device)
    os.makedirs(results_dir, exist_ok=True)

    # --- t-SNE（全データで可視化）---
    print("\n[1/2] t-SNE 解析を実行中...")
    tsne_path = os.path.join(results_dir, f"tsne_fold{fold_idx}.png")
    run_tsne(model, file_paths, labels, class_names,
             config, device, tsne_path, model_name, fold_idx)

    # --- 注目領域可視化 ---
    print("\n[2/2] 注目領域解析を実行中...")
    val_file_paths = [file_paths[i] for i in val_idx]
    val_labels     = [labels[i] for i in val_idx]

    if model_name == "cnn":
        run_gradcam_cnn(model, val_file_paths, val_labels, class_names,
                        config, device, results_dir, fold_idx)
    elif model_name == "vit":
        run_attention_rollout(model, model_name, val_file_paths, val_labels, class_names,
                              config, device, results_dir, fold_idx)
    else:
        print(f"  {model_name} の注目領域可視化は未実装です（AST は複雑な入力形式のため省略）。")

    print(f"\n解析完了。結果: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="vit", choices=["cnn", "vit", "ast"])
    parser.add_argument("--fold",   type=int, default=0, help="解析する fold 番号")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(model_name=args.model, fold_idx=args.fold, config_path=args.config)
