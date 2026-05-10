"""
main.py
 
各オブジェクトのディレクトリ内にある全 wav ファイルを読み込み、
掴むイベントを検出してセグメントを切り出し、processed_data_dir に保存する。
 
想定するディレクトリ構造（入力）:
    data/audio/raw/
    ├── metal_nut/
    │   ├── trial_01.wav
    │   ├── trial_02.wav
    │   └── ...
    ├── plastic_bolt/
    └── ...
 
出力:
    data/audio/processed/
    ├── metal_nut/
    │   ├── segment_00.wav   # 全ファイルをまたいだ通し番号
    │   ├── segment_01.wav
    │   └── ...
    ├── plastic_bolt/
    └── ...
"""

import yaml
import os
import soundfile as sf
from audio_utils import split_grasp_events

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_object(wav_path, out_dir, sr, threshold, dead_time_sec, pre, post, segment_offset):
    """
    1つの wav ファイルを処理し、セグメントを保存する。
 
    Args:
        wav_path (str): 入力 wav ファイルのパス
        out_dir (str): 保存先ディレクトリ（オブジェクト名のサブディレクトリ）
        sr (int): サンプリングレート
        threshold (float): 衝撃検知の RMS 閾値
        dead_time_sec (float): 二重検知防止の休止時間（秒）
        pre (float): トリガー前に切り出す秒数
        post (float): トリガー後に切り出す秒数
        segment_offset (int): セグメント番号の開始値（複数ファイルをまたぐ通し番号）
 
    Returns:
        int: このファイルで検出されたセグメント数
    """
    # --- 読み込み ---
    y, file_sr = sf.read(wav_path, always_2d=True)  # shape: (samples, ch)
    y = y.T  # → (ch, samples) に転置
 
    if file_sr != sr:
        print(f"    [警告] サンプリングレートが config と異なります: {file_sr} Hz（config: {sr} Hz）")
 
    # --- イベント検出・切り出し ---
    segments = split_grasp_events(y, file_sr, threshold, dead_time_sec, pre, post)
    print(f"    {os.path.basename(wav_path)}: {len(segments)} イベント検出")
 
    # --- 保存（通し番号で連番付け）---
    os.makedirs(out_dir, exist_ok=True)
    for i, seg in enumerate(segments):
        seg_idx = segment_offset + i
        save_path = os.path.join(out_dir, f"segment_{seg_idx:02d}.wav")
        sf.write(save_path, seg.T, file_sr)  # soundfile は (samples, ch) を期待するので転置
 
    return len(segments)
 
 
def main():
    config = load_config("config.yaml")
 
    sr            = config["sampling_rate"]
    dead_time_sec = config["dead_time_sec"]
    pre           = config["segment"]["pre_sec"]
    post          = config["segment"]["post_sec"]
    threshold     = config["segment"]["threshold"]
    raw_dir       = config["paths"]["raw_data_dir"]
    processed_dir = config["paths"]["processed_data_dir"]
    objects       = config["objects"]
 
    print(f"入力ディレクトリ : {raw_dir}")
    print(f"出力ディレクトリ : {processed_dir}")
    print(f"対象オブジェクト : {objects}\n")
 
    for obj in objects:
        obj_dir = os.path.join(raw_dir, obj)
 
        if not os.path.isdir(obj_dir):
            print(f"  [スキップ] ディレクトリが見つかりません: {obj_dir}")
            continue
 
        wav_files = sorted([f for f in os.listdir(obj_dir) if f.endswith(".wav")])
 
        if not wav_files:
            print(f"  [スキップ] wav ファイルが見つかりません: {obj_dir}")
            continue
 
        print(f"処理中: {obj}（{len(wav_files)} ファイル）")
 
        out_dir = os.path.join(processed_dir, obj)
        segment_offset = 0  # オブジェクト単位で通し番号をリセット
 
        for wav_file in wav_files:
            wav_path = os.path.join(obj_dir, wav_file)
            n_detected = process_object(
                wav_path, out_dir, sr, threshold, dead_time_sec, pre, post, segment_offset
            )
            segment_offset += n_detected  # 次のファイルの番号を前のファイルの続きから始める
 
        print(f"  → 合計 {segment_offset} セグメント保存\n")
 
    print("完了")
