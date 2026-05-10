"""
train.py

5-fold クロスバリデーションによる学習ループ。

学習戦略（小規模データ向け）:
    Phase 1（最初の freeze_epochs エポック）: backbone を凍結し、分類ヘッドだけ学習。
    Phase 2（残りのエポック）             : 後半ブロックを解凍して全体を微調整。

実行例:
    python train.py                       # ViT で学習（デフォルト）
    python train.py --model cnn           # 軽量 CNN で学習
    python train.py --model ast           # AST で学習
    python train.py --model vit --fold 0  # 特定の fold だけ実行
"""

import os
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    GraspSoundDataset,
    get_transforms,
    collect_files_and_labels,
    get_kfold_splits,
)
from models import build_model


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """再現性のためにシードを固定する。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    1エポック分の学習を実行（モデルの重みを更新）し、平均 loss と accuracy を返す。

    Args:
        model: nn.Module
        loader: DataLoader（学習用）
        criterion: 損失関数
        optimizer: オプティマイザ
        device: torch.device

    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()  # 重み更新はしない
def evaluate_one_epoch(model, loader, criterion, device):
    """
    1エポック分の検証を実行し、平均 loss と accuracy を返す。

    Args:
        model: nn.Module
        loader: DataLoader（検証用）
        criterion: 損失関数
        device: torch.device

    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


def run_fold(fold_idx, train_idx, val_idx, file_paths, labels,
             model_name, config, device, save_dir):
    """
    1つの fold の学習・検証を実行し、最良モデルを保存する。

    Args:
        fold_idx (int): fold 番号（ログ表示用）
        train_idx (np.ndarray): 学習サンプルのインデックス
        val_idx (np.ndarray): 検証サンプルのインデックス
        file_paths (list[str]): 全スペクトログラムのパスリスト
        labels (list[int]): 全ラベルリスト
        model_name (str): "cnn" / "vit" / "ast"
        config (dict): config.yaml の内容
        device (torch.device): 使用デバイス
        save_dir (str): モデル保存先ディレクトリ

    Returns:
        dict: {"best_val_acc": float, "history": list[dict]}
    """
    train_cfg     = config["train"]
    epochs        = train_cfg["epochs"]
    freeze_epochs = train_cfg["freeze_epochs"]
    batch_size    = train_cfg["batch_size"]
    lr_head       = train_cfg["lr_head"]
    lr_backbone   = train_cfg["lr_backbone"]
    weight_decay  = train_cfg.get("weight_decay", 1e-4)
    img_size      = config["melspec"].get("img_size", 224)
    num_classes   = len(config["objects"])

    train_transform, val_transform = get_transforms(img_size)

    # train/val Datasetを作る
    train_dataset = GraspSoundDataset(
        [file_paths[i] for i in train_idx],
        [labels[i]     for i in train_idx],
        transform=train_transform,
    )
    val_dataset = GraspSoundDataset(
        [file_paths[i] for i in val_idx],
        [labels[i]     for i in val_idx],
        transform=val_transform,
    )

    # DataLoaderを作る（Datasetからデータをまとめて取り出す）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    # モデルを作る（Phase 1 は backbone 凍結）
    model = build_model(model_name, num_classes=num_classes, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Phase 1: 分類ヘッドのみ学習
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=freeze_epochs)

    best_val_acc    = 0.0
    best_model_path = os.path.join(save_dir, f"best_fold{fold_idx}.pt")
    os.makedirs(save_dir, exist_ok=True)
    history = []

    # epoch ごとに学習・検証
    print(f"\n--- Fold {fold_idx} ---")
    print(f"  学習: {len(train_dataset)} サンプル / 検証: {len(val_dataset)} サンプル")

    for epoch in range(epochs):

        # Phase 2 に切り替え: 後半 4 ブロックを解凍し、低い lr で微調整
        if epoch == freeze_epochs and model_name in ("vit", "ast"):
            model.unfreeze_last_n_blocks(n=4)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_backbone, weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs
            )
            print(f"  [Epoch {epoch+1}] Phase 2 開始: backbone 後半 4 ブロックを解凍")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
        })

        print(f"  Epoch {epoch+1:03d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        # best modelを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"  → Fold {fold_idx} 最良 val accuracy: {best_val_acc:.3f}")
    return {"best_val_acc": best_val_acc, "history": history}


def main(model_name="vit", target_fold=None, config_path="config.yaml"):
    # configを読む
    config = load_config(config_path)
    # seed固定
    set_seed(config["train"].get("seed", 42))

    # device決定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    spec_dir = config["paths"]["spectrogram_dir"]
    save_dir = os.path.join(config["paths"]["model_dir"], model_name)
    objects  = config["objects"]
    n_splits = config["train"]["n_splits"]

    #.npyファイルとラベルを集める
    file_paths, labels, class_names = collect_files_and_labels(spec_dir, objects)
    print(f"総サンプル数: {len(file_paths)} ({len(class_names)} クラス)")

    # K-Fold分割する
    splits = get_kfold_splits(
        file_paths, labels,
        n_splits=n_splits,
        random_state=config["train"].get("seed", 42),
    )

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if target_fold is not None and fold_idx != target_fold:
            continue

        # foldごとにrun_foldを呼ぶ（学習を行う）
        result = run_fold(
            fold_idx, train_idx, val_idx,
            file_paths, labels,
            model_name, config, device, save_dir,
        )
        fold_results.append({"fold": fold_idx, **result})

    # 結果を集計
    accs = np.array([r["best_val_acc"] for r in fold_results])
    print(f"\n{'='*40}")
    print(f"モデル: {model_name}")
    print(f"全 fold 平均 accuracy : {accs.mean():.3f} ± {accs.std():.3f}")
    print(f"各 fold accuracy      : {[f'{a:.3f}' for a in accs]}")

    # 結果をJSONに保存
    results_dir = config["paths"].get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{model_name}_fold_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, ensure_ascii=False, indent=2)
    print(f"結果を保存: {result_path}")

# コマンドライン引数を受け取っている
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="vit", choices=["cnn", "vit", "ast"],
                        help="使用するモデル（デフォルト: vit）")
    parser.add_argument("--fold",   type=int, default=None,
                        help="特定の fold だけ実行（デバッグ用）")
    parser.add_argument("--config", default="config.yaml",
                        help="config ファイルのパス")
    args = parser.parse_args()
    main(model_name=args.model, target_fold=args.fold, config_path=args.config)
