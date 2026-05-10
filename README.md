# Grasp Sound Classification

## Description

Grasp Sound Classification は、ロボットハンドに取り付けたピエゾ素子（接触型マイク）のステレオ音声データから、**掴んだオブジェクトの種類を識別する**深層学習プロジェクトです。

7種類のオブジェクトを掴んだときの衝撃音を録音し、L / R / 差分（L-R）の 3ch メルスペクトログラムに変換したうえで、軽量 CNN・ViT・AST の 3 モデルで分類します。5-fold クロスバリデーションによって、少量データ（7クラス × 20サンプル = 140件）での汎化性能を評価します。

---

## Instructions

### 環境構築

```bash
pip install torch torchvision torchaudio transformers
pip install librosa soundfile scikit-learn matplotlib
```

### 実行順序

```bash
# ① wav からイベントを切り出して保存
python src/segment_extractor.py

# ② 切り出したセグメントを 3ch メルスペクトログラムに変換
python src/wav_to_melspec.py

# ③ 学習（デフォルト: ViT）
python src/train.py --model vit

# ④ 評価（混同行列・クラスごと F1）
python src/evaluate.py --model vit

# ⑤ 特徴量解析（t-SNE・Grad-CAM / Attention Rollout）
python src/analyze.py --model vit --fold 0
```

### モデルの切り替え

```bash
python src/train.py --model cnn   # 軽量 CNN（ベースライン）
python src/train.py --model vit   # ViT-B/16（ImageNet 事前学習済み）
python src/train.py --model ast   # AST（AudioSet 事前学習済み）

# 特定の fold だけ実行（デバッグ用）
python src/train.py --model vit --fold 0
```

---

## Directory Structure

```
.
├── src/
│   ├── segment_extractor.py      # ① イベント検出・セグメント切り出し
│   ├── wav_to_melspec.py         # ② 3ch メルスペクトログラム変換
│   ├── dataset.py                # PyTorch Dataset・K-Fold 分割
│   ├── models.py                 # CNN / ViT / AST モデル定義
│   ├── train.py                  # 学習ループ（5-fold CV）
│   ├── evaluate.py               # 評価・混同行列
│   ├── analyze.py                # t-SNE・Grad-CAM・Attention Rollout
│   └── audio_utils.py            # 音声処理ユーティリティ
├── data/
│   ├── audio/
│   │   ├── raw/                  # 録音した元の wav ファイル
│   │   │   ├── metal_nut/
│   │   │   └── ...
│   │   └── processed/            # イベント切り出し後のセグメント wav
│   │       ├── metal_nut/
│   │       └── ...
│   └── spectrograms/             # 3ch メルスペクトログラム (.npy)
│       ├── metal_nut/
│       └── ...
├── models/                       # 学習済みモデル重み (.pt)
│   ├── vit/
│   └── ...
├── results/                      # 評価結果・可視化画像
│   ├── vit/
│   └── ...
└── config.yaml                   # 全設定ファイル
```

---

## Project Overview

### Pipeline

```
wav 録音データ
    ↓  src/segment_extractor.py   RMS トリガーでイベント検出・切り出し
セグメント wav（pre=0.3s + post=0.7s）
    ↓  src/wav_to_melspec.py      L / R / (L-R) の 3ch メルスペクトログラム変換
スペクトログラム .npy（3, n_mels, time_frames）
    ↓  src/train.py               5-fold CV で学習（Phase1: ヘッドのみ → Phase2: backbone 解凍）
学習済みモデル .pt
    ↓  src/evaluate.py            混同行列・クラスごと F1 スコア
    ↓  src/analyze.py             t-SNE 特徴空間・Grad-CAM / Attention Rollout
```

### Objects（分類クラス）

| インデックス | クラス名 |
|------------|---------|
| 0 | `metal_nut` |
| 1 | `plastic_bolt` |
| 2 | `sponge` |
| 3 | `rubber_ball` |
| 4 | `wood_block` |
| 5 | `silicon_tube` |
| 6 | `cloth_bundle` |

---

## File Reference

### `segment_extractor.py` — イベント検出・切り出し

raw/ 以下の wav を走査し、RMS エネルギーが閾値を超えた時点をトリガーとしてセグメントを切り出します。

| パラメータ（config.yaml） | 説明 |
|--------------------------|------|
| `segment.threshold` | 衝撃検知の RMS 閾値（デフォルト: 0.05） |
| `segment.pre_sec` | トリガー前の切り出し秒数（デフォルト: 0.3s） |
| `segment.post_sec` | トリガー後の切り出し秒数（デフォルト: 0.7s） |
| `dead_time_sec` | 二重検知防止の休止時間（デフォルト: 2.0s） |

### `wav_to_melspec.py` — 3ch スペクトログラム変換

ステレオ wav を L / R / (L-R) の 3ch メルスペクトログラムに変換し、各チャンネルを独立に [0, 1] へ min-max 正規化します。

| パラメータ（config.yaml） | 説明 |
|--------------------------|------|
| `melspec.n_mels` | メルフィルタバンク数（デフォルト: 128） |
| `melspec.fmax` | 最大周波数（デフォルト: 8000 Hz） |
| `melspec.hop_length` | STFT ホップ長（デフォルト: 256） |
| `melspec.n_fft` | FFT 点数（デフォルト: 1024） |

> **L-R 差分チャンネルについて:** 左右のマイクの信号差は接触位置の非対称性を表します。ViT / AST の 3ch 入力に自然に対応させながら、ステレオ固有の情報を明示的に与えます。

### `models.py` — モデル定義

3 種類のアーキテクチャを `build_model(name, num_classes)` で切り替えられます。

#### SimpleCNN（ベースライン）

| レイヤー | 出力サイズ |
|---------|----------|
| Conv2d(3→32) + BN + ReLU + MaxPool | 112×112 |
| Conv2d(32→64) + BN + ReLU + MaxPool | 56×56 |
| Conv2d(64→128) + BN + ReLU + MaxPool | 28×28 |
| AdaptiveAvgPool(4×4) + Linear(256) + Dropout(0.5) + Linear(7) | — |

#### ViTClassifier

`torchvision` の ViT-B/16（ImageNet 事前学習済み）の分類ヘッドを差し替えてファインチューニングします。

#### ASTClassifier

`MIT/ast-finetuned-audioset-10-10-0.4593`（AudioSet 事前学習済み）をベースにした音声特化 ViT です。入力は `(batch, time_frames, n_mels)` 形式のため、3ch スペクトログラムの L チャンネルのみを使用します。

### `train.py` — 学習ループ

小規模データ向けの 2 フェーズ学習を実装しています。

| フェーズ | エポック | 内容 |
|---------|---------|------|
| Phase 1 | 0 〜 `freeze_epochs` | backbone 凍結・分類ヘッドのみ学習（lr: `lr_head`） |
| Phase 2 | `freeze_epochs` 〜 `epochs` | backbone 後半 4 ブロックを解凍して微調整（lr: `lr_backbone`） |

| パラメータ（config.yaml） | デフォルト値 | 説明 |
|--------------------------|------------|------|
| `train.epochs` | 40 | 総エポック数 |
| `train.freeze_epochs` | 10 | Phase 1 のエポック数 |
| `train.batch_size` | 16 | バッチサイズ |
| `train.lr_head` | 1e-3 | Phase 1 学習率 |
| `train.lr_backbone` | 1e-5 | Phase 2 学習率 |
| `train.n_splits` | 5 | K-Fold 数 |

### `evaluate.py` — 評価

| 出力 | 説明 |
|------|------|
| `{model}_fold{N}_cm.png` | fold ごとの混同行列 |
| `{model}_all_folds_cm.png` | 全 fold 合算の混同行列 |
| `{model}_eval_results.json` | accuracy・混同行列の数値 |

各クラスの Precision / Recall / F1 スコアは標準出力に表示されます。

### `analyze.py` — 特徴量解析

| 機能 | 対象モデル | 出力 |
|------|----------|------|
| t-SNE | CNN / ViT / AST | 特徴空間の 2D 散布図（クラス別色分け） |
| Grad-CAM | CNN | スペクトログラム上の注目領域ヒートマップ |
| Attention Rollout | ViT | 全アテンション層を掛け合わせたパッチ注目マップ |

---

## Key Concepts

**なぜ 3ch メルスペクトログラムか？**  
ViT / AST は 3ch 入力を前提とした事前学習重みを持つため、ステレオ 2ch をそのまま渡すことができません。L・R・差分（L-R）の 3ch にすることで、ステレオ情報を保ちながらモデルの入力形式に自然に対応できます。

**なぜ 5-fold クロスバリデーションか？**  
1 クラスあたり 20 サンプルという小規模データでは、単純な train/test 分割だと評価の分散が大きくなります。Stratified K-Fold により各 fold でクラス分布を均等に保ち、全 fold の平均・標準偏差で安定した性能評価を行います。

**2 フェーズ学習の意図は？**  
事前学習済みの ViT / AST に対してすべての重みを最初から更新すると、140 サンプルでは過学習が起きやすくなります。まず分類ヘッドだけを学習してタスクに適応させ、その後 backbone の後半ブロックのみ低い学習率で解凍することで、事前学習の知識を保ちながら微調整します。

**Attention Rollout とは？**  
ViT の各 Transformer 層が持つアテンション行列を、残差接続を考慮しながら全層分掛け合わせることで、最終的に [CLS] トークンがどのパッチを参照しているかを可視化する手法です。

---

## Resources

- [librosa ドキュメント](https://librosa.org/doc/latest/index.html)
- [librosa.feature.melspectrogram](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html)
- [PyTorch ViT (torchvision)](https://pytorch.org/vision/stable/models/vision_transformer.html)
- [AST: Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- [Attention Rollout 論文](https://arxiv.org/abs/2005.00928)
- [Grad-CAM 論文](https://arxiv.org/abs/1610.02391)
