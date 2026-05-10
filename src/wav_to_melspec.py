"""
wav_to_melspec.py

切り出し済みの wav セグメント（stereo）を
L / R / (L-R) の 3ch メルスペクトログラム画像に変換して保存する。

入力:
    data/audio/processed/
    ├── metal_nut/
    │   ├── segment_00.wav   # shape: (samples, 2)
    │   └── ...
    └── ...

出力:
    data/spectrograms/
    ├── metal_nut/
    │   ├── segment_00.npy   # shape: (3, n_mels, time_frames) float32
    │   └── ...
    └── ...
"""

import os
import yaml
import numpy as np
import soundfile as sf
import librosa


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stereo_to_3ch_melspec(y_stereo, sr, n_mels, fmax, hop_length, n_fft):
    """
    ステレオ波形（2ch）を L / R / (L-R) の 3ch メルスペクトログラムに変換する。

    ViT / AST は 3ch 入力を期待するため、
    L/R/差分 の 3ch でステレオ情報と左右非対称の接触情報を表現する。

    Args:
        y_stereo (np.ndarray): shape (2, samples) のステレオ波形
        sr (int): サンプリングレート
        n_mels (int): メルフィルタバンクの数
        fmax (float): 最大周波数 [Hz]
        hop_length (int): STFTのホップ長
        n_fft (int): FFT点数

    Returns:
        np.ndarray: shape (3, n_mels, time_frames) の float32 配列
                    ch0=L, ch1=R, ch2=L-R（差分）
    """
    def _melspec_db(y_mono):
        S = librosa.feature.melspectrogram(
            y=y_mono, sr=sr,
            n_mels=n_mels, fmax=fmax,
            hop_length=hop_length, n_fft=n_fft,
        )
        return librosa.power_to_db(S, ref=np.max).astype(np.float32)

    ch_L = _melspec_db(y_stereo[0])   # shape: (n_mels, time_frames)
    ch_R = _melspec_db(y_stereo[1])
    ch_D = ch_L - ch_R                # 左右の差分：接触位置の非対称性を表現

    return np.stack([ch_L, ch_R, ch_D], axis=0)  # (3, n_mels, time_frames)


def normalize_3ch(spec_3ch):
    """
    各チャンネルを独立に [0, 1] へ正規化する（min-max正規化）。

    Args:
        spec_3ch (np.ndarray): shape (3, n_mels, time_frames)

    Returns:
        np.ndarray: shape (3, n_mels, time_frames) の float32、各chが[0,1]
    """
    out = np.empty_like(spec_3ch)
    for i in range(spec_3ch.shape[0]):
        ch = spec_3ch[i]
        ch_min, ch_max = ch.min(), ch.max()
        # 正規化（0〜1）
        out[i] = (ch - ch_min) / (ch_max - ch_min + 1e-8)
    return out


def process_all(config):
    """
    processed_data_dir 以下の全 wav を変換して spectrogram_dir に保存する。

    Args:
        config (dict): config.yaml をロードした辞書
    """
    sr          = config["sampling_rate"]
    n_mels      = config["melspec"]["n_mels"]
    fmax        = config["melspec"]["fmax"]
    hop_length  = config["melspec"]["hop_length"]
    n_fft       = config["melspec"]["n_fft"]
    proc_dir    = config["paths"]["processed_data_dir"]
    spec_dir    = config["paths"]["spectrogram_dir"]
    objects     = config["objects"]

    for obj in objects:
        in_dir  = os.path.join(proc_dir, obj)
        out_dir = os.path.join(spec_dir, obj)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(in_dir):
            print(f"  [スキップ] ディレクトリが見つかりません: {in_dir}")
            continue

        wav_files = sorted([f for f in os.listdir(in_dir) if f.endswith(".wav")])
        print(f"{obj}: {len(wav_files)} ファイルを変換中...")

        for wav_file in wav_files:
            wav_path = os.path.join(in_dir, wav_file)

            # soundfile は (samples, ch) で返すので転置して (ch, samples) にする
            y, _ = sf.read(wav_path, always_2d=True)  # モノラルも2D配列 (samples, 1) に
            y = y.T.astype(np.float32)  # 転置

            # モノラルの場合はチャンネルを複製してステレオとして扱う
            if y.shape[0] == 1:
                y = np.concatenate([y, y], axis=0)

            spec = stereo_to_3ch_melspec(y, sr, n_mels, fmax, hop_length, n_fft)
            spec = normalize_3ch(spec)

            npy_name = wav_file.replace(".wav", ".npy")
            np.save(os.path.join(out_dir, npy_name), spec)

        print(f"  → {out_dir} に保存完了")


if __name__ == "__main__":
    config = load_config("config.yaml")
    process_all(config)
    print("スペクトログラム変換 完了")
