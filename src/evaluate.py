"""
evaluate.py

学習済みモデルの評価スクリプト。

機能:
    - 全 fold の best モデルをロードして検証セットで推論
    - 混同行列の可視化・保存
    - クラスごとの Precision / Recall / F1 の集計
    - 全 fold の平均 accuracy 表示

実行例:
    python evaluate.py                  # ViT で評価（デフォルト）
    python evaluate.py --model cnn      # 軽量 CNN で評価
    python evaluate.py --model ast      # AST で評価
    python evaluate.py --model vit --fold 0  # 特定の fold だけ評価
"""

import os
import argparse
import yaml
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # GUI なし環境でも動作するよう Agg バックエンドを使用
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, ConfusionMatrixDisplay,
)
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


@torch.no_grad()  # モデルの重みを更新しない
def predict(model, loader, device):
    """
    DataLoader 全体に対して推論を行い、予測ラベルと正解ラベルを返す。

    Args:
        model: nn.Module（eval モード）
        loader: DataLoader
        device: torch.device

    Returns:
        tuple: (preds np.ndarray, targets np.ndarray)
    """
    model.eval()
    all_preds, all_targets = [], []

    for specs, labels in loader:
        specs = specs.to(device)
        logits = model(specs)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    混同行列をプロットして保存する。

    Args:
        cm (np.ndarray): 混同行列
        class_names (list[str]): クラス名リスト
        title (str): プロットのタイトル
        save_path (str): 保存先パス
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    # 混同行列を保存
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  混同行列を保存: {save_path}")


def evaluate_fold(fold_idx, val_idx, file_paths, labels,
                  model_name, config, device, model_dir, results_dir, class_names):
    """
    1 fold の評価を実行する。

    Args:
        fold_idx (int): fold 番号
        val_idx (np.ndarray): 検証サンプルのインデックス
        file_paths (list[str]): 全スペクトログラムのパスリスト
        labels (list[int]): 全ラベルリスト
        model_name (str): "cnn" / "vit" / "ast"
        config (dict): config.yaml の内容
        device (torch.device): 使用デバイス
        model_dir (str): モデルが保存されているディレクトリ
        results_dir (str): 結果保存先ディレクトリ
        class_names (list[str]): クラス名リスト

    Returns:
        dict: {"fold": int, "accuracy": float, "report": str, "cm": list}
    """
    # 設定値を取り出す
    img_size   = config["melspec"].get("img_size", 224)  # 入力画像サイズ
    batch_size = config["train"]["batch_size"]  # バッチサイズ
    num_classes = len(class_names)  # クラス数

    # 検証用transformだけを使う
    _, val_transform = get_transforms(img_size)

    # 検証データセットを作る
    val_dataset = GraspSoundDataset(
        [file_paths[i] for i in val_idx],
        [labels[i]     for i in val_idx],
        transform=val_transform,
    )
    # DataLoaderにする
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 学習済みモデルをロード（fold_idxの.ptファイルを探す）
    model_path = os.path.join(model_dir, f"best_fold{fold_idx}.pt")
    if not os.path.exists(model_path):
        print(f"  [スキップ] モデルが見つかりません: {model_path}")
        return None

    # 同じ構造のモデルを作る
    model = build_model(model_name, num_classes=num_classes, freeze_backbone=False)
    # 重みを読み込む
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # 推論
    preds, targets = predict(model, val_loader, device)  # モデルの予測ラベルと正解ラベル
    acc = accuracy_score(targets, preds)  # 精度を計算

    report = classification_report(
        targets, preds, target_names=class_names, digits=3, zero_division=0
    )

    # 混同行列をつくる
    cm = confusion_matrix(targets, preds)
    os.makedirs(results_dir, exist_ok=True)
    # 画像として保存
    plot_confusion_matrix(
        cm, class_names,
        title=f"{model_name.upper()} - Fold {fold_idx} Confusion Matrix (acc={acc:.3f})",
        save_path=os.path.join(results_dir, f"{model_name}_fold{fold_idx}_cm.png"),
    )

    print(f"\n--- Fold {fold_idx} ---")
    print(f"  Accuracy: {acc:.3f}")
    print(report)

    return {
        "fold":     fold_idx,
        "accuracy": acc,
        "report":   report,
        "cm":       cm.tolist(),
    }


def summarize_and_plot_aggregate_cm(all_results, class_names, model_name, results_dir):
    """
    全 fold の混同行列を合算して Macro Avg を計算・可視化する。

    Args:
        all_results (list[dict]): 各 fold の評価結果
        class_names (list[str]): クラス名リスト
        model_name (str): モデル名（ファイル名に使用）
        results_dir (str): 保存先ディレクトリ
    """
    cm_sum = np.zeros((len(class_names), len(class_names)), dtype=int)
    for r in all_results:
        cm_sum += np.array(r["cm"])

    plot_confusion_matrix(
        cm_sum, class_names,
        title=f"{model_name.upper()} - All Folds Aggregated Confusion Matrix",
        save_path=os.path.join(results_dir, f"{model_name}_all_folds_cm.png"),
    )

    accs = [r["accuracy"] for r in all_results]
    print(f"\n{'='*40}")
    print(f"モデル: {model_name}")
    print(f"全 fold 平均 accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"各 fold accuracy      : {[f'{a:.3f}' for a in accs]}")


def main(model_name="vit", target_fold=None, config_path="config.yaml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    spec_dir    = config["paths"]["spectrogram_dir"]
    model_dir   = os.path.join(config["paths"]["model_dir"], model_name)
    results_dir = os.path.join(config["paths"].get("results_dir", "results"), model_name)
    objects     = config["objects"]
    n_splits    = config["train"]["n_splits"]
    seed        = config["train"].get("seed", 42)

    file_paths, labels, class_names = collect_files_and_labels(spec_dir, objects)
    print(f"総サンプル数: {len(file_paths)} ({len(class_names)} クラス)")

    splits = get_kfold_splits(file_paths, labels, n_splits=n_splits, random_state=seed)

    all_results = []
    for fold_idx, (_, val_idx) in enumerate(splits):
        if target_fold is not None and fold_idx != target_fold:
            continue

        result = evaluate_fold(
            fold_idx, val_idx, file_paths, labels,
            model_name, config, device, model_dir, results_dir, class_names,
        )
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        summarize_and_plot_aggregate_cm(all_results, class_names, model_name, results_dir)

    # JSON 保存
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{model_name}_eval_results.json")
    serializable = [{k: v for k, v in r.items() if k != "report"} for r in all_results]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n評価結果を保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="vit", choices=["cnn", "vit", "ast"])
    parser.add_argument("--fold",   type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(model_name=args.model, target_fold=args.fold, config_path=args.config)
