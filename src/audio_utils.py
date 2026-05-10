import numpy as np
import librosa
import soundfile as sf

def detect_grasp_trigger(audio_chunk, sr, threshold):
    """
    RMSエネルギーが閾値を超えたらTrueを返す。

    Args:
        audio_chunk (np.array): shape (ch, samples) のステレオチャンク

    Returns:
        bool: RMSが閾値を超えていればTrue
    """
    # ステレオの場合は全チャンネルの平均RMSで判定する
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms > threshold

def extract_grasp_segment(audio, sr, trigger_sample, pre, post):
    """
    トリガー位置を中心に前後を切り出し、長さを (pre + post) 秒に揃えて返す。

    Args:
        audio (np.ndarray): shape (ch, total_samples) のステレオ全体波形
        trigger_sample (int): トリガー検出位置（サンプル番号）
        pre (float): トリガー前に切り出す秒数
        post (float): トリガー後に切り出す秒数
    
    Returns:
        np.array: shape (ch, target_samples) の切り出しセグメント
    """
    start = max(0, trigger_sample - int(pre * sr))
    end   = min(audio.shape[1], trigger_sample + int(post * sr))
    segment = audio[:, start:end]
    # 音声先頭・末尾付近でチャンクが短くなる場合はゼロパディングで補完
    target_len = int((pre + post) * sr)
    if segment.shape[1] < target_len:
        pad_width = target_len - segment.shape[1]
        segment = np.pad(segment, ((0, 0), (0, pad_width)))  # 時間軸方向にのみパディング

    return segment

def split_grasp_events(y, sr, threshold, dead_time_sec, pre, post):
    """
    音声全体をスキャンして「掴むイベント」のセグメントリストを返す。
    50ms ごとにRMSを計算し、閾値釣果をトリガーとして切り出す。
    検出後は dead_time_sec の間スキャンを休止することで二重検知を防ぐ。

    Args:
        y (np.ndarray): shape (ch, total_samples) のステレオ全体波形
        sr (int): サンプリングレート
        threshold (float): 衝撃検知のRMS閾値
        dead_time_sec (float): 一度検知したあとに次の検知を受け付けない時間（秒）
        pre (float): トリガー前に切り出す秒数
        post (float): トリガー後に切り出す秒数
    
    Returns:
        list[np.ndarray]: 各要素が shape (ch (L or R), target_samples) のセグメントリスト ←理想的には20回分
    """
    segments = []
    current_idx = 0
    chunk_samples = int(0.05 * sr)  # 50ms単位でスキャン
    dead_time_samples = int(dead_time_sec * sr)  # 二重検知防止のスキップ幅

    while current_idx < y.shape[1] - chunk_samples:
        chunk = y[:, current_idx : current_idx + chunk_samples]
        
        if detect_grasp_trigger(chunk, sr, threshold):
            segment = extract_grasp_segment(y, sr, current_idx, pre, post)
            segments.append(segment)
            current_idx += dead_time_samples # 二重検知防止:dead_time_sec 分スキップ
        else:
            # 未検知時は 25ms(chunk の半分)ずつ進める（オーバーラップスキャン）
            current_idx += int(chunk_samples / 2)
            
    return segments
