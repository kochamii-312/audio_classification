"""
models.py

3つのモデルアーキテクチャを提供する。

  - SimpleCNN      : 軽量なベースライン。過学習しにくくデバッグが容易。
  - ViTClassifier  : ImageNet 事前学習済み ViT のファインチューニング。
  - ASTClassifier  : AudioSet 事前学習済み AST（音声特化 ViT）のファインチューニング。

使い方:
    model = build_model("vit", num_classes=7)
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import ASTForAudioClassification


# ---------------------------------------------------------------------------
# SimpleCNN（ベースライン）
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    軽量な CNN ベースライン。

    140サンプルの小規模データでも過学習しにくいよう Dropout を強めに設定。
    まずこれを動かしてデータパイプラインを検証し、その後 ViT/AST と比較する。

    Args:
        num_classes (int): 分類クラス数（デフォルト 7）
    """

    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 224 → 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 112 → 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 56 → 28
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def get_features(self, x):
        """分類ヘッド直前の特徴ベクトルを返す（t-SNE 可視化用）。"""
        x = self.features(x)
        pool = nn.AdaptiveAvgPool2d((4, 4))
        x = pool(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# ViT（ImageNet 事前学習済み）
# ---------------------------------------------------------------------------

class ViTClassifier(nn.Module):
    """
    torchvision の ViT-B/16(Vision Transformer Baseサイズ 16×16) をベースにしたファインチューニングモデル。

    学習戦略:
        1. 最初の freeze_epochs エポックは backbone を凍結し、分類ヘッドだけ学習する。
        2. その後 unfreeze_last_n_blocks() で後半ブロックを解凍して全体を微調整する。

    Args:
        num_classes (int): 分類クラス数
        freeze_backbone (bool): True の場合、初期化時に backbone を凍結する
    """

    def __init__(self, num_classes=7, freeze_backbone=True):
        super().__init__()
        # torchvision が用意している ViT-B/16 を読み込む
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """
        分類ヘッド以外のパラメータを凍結する。
        最初は事前学習済み ViT の本体部分は変えず、最後の分類層だけ 7クラス用に変更して学習する。
        データ数が少ないときに、いきなり全部の重みを学習すると過学習しやすいので、まず分類ヘッドだけ学習する。
        """
        for name, param in self.vit.named_parameters():
            if "heads" not in name:
                param.requires_grad = False

    def unfreeze_last_n_blocks(self, n=4):
        """
        必要に応じて、
        ViT の後ろ側の Transformer ブロック n 個のパラメータを解凍する。

        Args:
            n (int): 解凍するブロック数（ViT-B/16 は全 12 ブロック）
        """
        encoder_layers = self.vit.encoder.layers
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        """入力 x をそのまま ViT に渡して、分類スコアを返す。"""
        return self.vit(x)

    def get_features(self, x):
        """
        t-SNE 可視化用。ViT の [CLS] トークンの特徴ベクトルを返す。
        モデルが各サンプルをどういう特徴としてみているかを可視化する。
        """
        x = self.vit._process_input(x)
        cls_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 0]  # CLS トークン


# ---------------------------------------------------------------------------
# AST（AudioSet 事前学習済み）
# ---------------------------------------------------------------------------

class ASTClassifier(nn.Module):
    """
    MIT/ast-finetuned-audioset をベースにした音声特化 ViT。

    ImageNet ではなく AudioSet（200万件の音声）で事前学習されており、
    音響特徴の抽出に特化した注意機構を持つ。
    スペクトログラム入力に最も適したアーキテクチャ。

    Args:
        num_classes (int): 分類クラス数
        freeze_backbone (bool): True の場合、初期化時に backbone を凍結する
    """

    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, num_classes=7, freeze_backbone=True):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """分類ヘッド以外のパラメータを凍結する。"""
        for name, param in self.ast.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_last_n_blocks(self, n=4):
        """
        後半 n ブロックのパラメータを解凍する。

        Args:
            n (int): 解凍するブロック数（AST は全 12 ブロック）
        """
        encoder_layers = self.ast.audio_spectrogram_transformer.encoder.layer
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        # AST は (batch, time_frames, n_mels) の入力を期待するため転置が必要。
        # 入力 x: (batch, 3, n_mels, time_frames) → Lチャンネルのみ使用
        x_mono = x[:, 0, :, :]           # (batch, n_mels, time_frames)
        x_mono = x_mono.permute(0, 2, 1) # → (batch, time_frames, n_mels)
        outputs = self.ast(input_values=x_mono)
        return outputs.logits

    def get_features(self, x):
        """[CLS] トークンの特徴ベクトルを返す（t-SNE 可視化用）。"""
        x_mono = x[:, 0, :, :].permute(0, 2, 1)
        outputs = self.ast.audio_spectrogram_transformer(input_values=x_mono)
        return outputs.last_hidden_state[:, 0]


# ---------------------------------------------------------------------------
# ファクトリ関数
# ---------------------------------------------------------------------------

def build_model(model_name, num_classes=7, freeze_backbone=True):
    """
    モデル名から対応するインスタンスを返すファクトリ関数。

    Args:
        model_name (str): "cnn" / "vit" / "ast"
        num_classes (int): 分類クラス数
        freeze_backbone (bool): backbone を凍結するか（vit/ast のみ有効）

    Returns:
        nn.Module: 初期化済みモデル

    Raises:
        ValueError: 未知のモデル名が渡された場合
    """
    name = model_name.lower()
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    elif name == "vit":
        return ViTClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    elif name == "ast":
        return ASTClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(
            f"未知のモデル名: {model_name}。'cnn' / 'vit' / 'ast' を指定してください。"
        )
