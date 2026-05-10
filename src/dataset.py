"""
dataset.py

スペクトログラム (.npy) を読み込む PyTorch Dataset と、
Stratified K-Fold によるデータ分割ユーティリティを提供する。

データ量が少ない（7クラス × 20サンプル = 140件）ため、
単純な train/test 分割ではなく 5-fold クロスバリデーションを用いる。

collect_files_and_labels()でnpyファイルを集めて、ファイルパス一覧とラベル一覧を作る。
get_kfold_splits()では集めたデータを、クラス比率がなるべく保たれるようにtrain/valに分ける。
get_transforms()ではnpyのスペクトログラムをモデル入力用に224*224にリサイズして、学習時のみ軽いデータ拡張。
GraspSoundDatasetは、PyTorchがDataLoader経由で読む本体。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms


class GraspSoundDataset(Dataset):
    """
    保存済みの.npyスペクトログラムを、PyTorchの学習ループで使える（入力, 正解ラベル）の形に変換する橋渡し役。
    PyTorch の DataLoader を用いて、スペクトログラム画像 (.npy) とラベルを1件ずつ返す。

    Args:
        file_paths (list[str]): .npy ファイルのパスリスト
        labels (list[int]): 各ファイルに対応するクラスインデックス
        transform: torchvision の transform（省略可）
    """

    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels     = labels
        self.transform  = transform

    # データセットの件数を返す。len(dataset) としたときに呼ばれる。
    def __len__(self):
        return len(self.file_paths)

    # idx 番目のデータを1件返す
    def __getitem__(self, idx):
        # shape: (3, n_mels, time_frames)、float32、値域 [0, 1]
        spec = np.load(self.file_paths[idx]).astype(np.float32)
        spec = torch.from_numpy(spec)

        if self.transform:
            spec = self.transform(spec)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec, label


def get_transforms(img_size=224):
    """
    学習用・検証用の transform を返す。
    小規模データなので学習時にデータ拡張を適用する。
    スペクトログラムに対する拡張は音声的な意味が崩れないよう控えめにする。
    学習用では、
      - Resize: サイズを 224 x 224 にそろえる
      - RandomHorizontalFlip: 時間方向の反転（前後対称性が低いため p=0.2 と控えめ）
      - ColorJitter: 振幅・コントラストの軽微な揺らぎ（録音環境差の吸収）
      - Normalize: 値の範囲をモデルが扱いやすい形に変える
    検証用では、このうち Resize と Normalize のみ

    Args:
        img_size (int): ViT/AST の入力サイズ（デフォルト 224）

    Returns:
        tuple: (train_transform, val_transform)
    """
    mean = [0.5, 0.5, 0.5]
    std  = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


def collect_files_and_labels(spectrogram_dir, objects):
    """
    spectrogram_dir 以下の .npy ファイルとラベルを収集する。

    Args:
        spectrogram_dir (str): スペクトログラムのルートディレクトリ
        objects (list[str]): オブジェクト名リスト（クラス名として使用）

    Returns:
        tuple:
            file_paths (list[str]): .npy ファイルのパスリスト
            labels (list[int]): クラスインデックスリスト
            class_names (list[str]): クラス名リスト（インデックス順）
    """
    file_paths  = []
    labels      = []
    class_names = objects

    for class_idx, obj in enumerate(objects):
        obj_dir = os.path.join(spectrogram_dir, obj)
        if not os.path.isdir(obj_dir):
            print(f"  [警告] ディレクトリが見つかりません: {obj_dir}")
            continue
        npy_files = sorted([f for f in os.listdir(obj_dir) if f.endswith(".npy")])
        for npy_file in npy_files:
            file_paths.append(os.path.join(obj_dir, npy_file))
            labels.append(class_idx)

    return file_paths, labels, class_names


def get_kfold_splits(file_paths, labels, n_splits=5, random_state=42):
    """
    Stratified K-Fold の分割インデックスを生成して返す。
    集めたデータを、クラス比率がなるべく保たれる（Sratified）ように train/val に分ける。

    Args:
        file_paths (list[str]): ファイルパスリスト
        labels (list[int]): ラベルリスト
        n_splits (int): fold 数（デフォルト 5）
        random_state (int): 再現性のためのシード

    Returns:
        list[tuple]: [(train_indices, val_indices), ...] の長さ n_splits のリスト
    """
    # skfには StratifiedKFold 分割を行うためのオブジェクトが格納される
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # 分割を生成する
    return list(skf.split(file_paths, labels))
