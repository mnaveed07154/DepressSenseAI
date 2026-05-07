from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


def load_audio(path, target_sr=16000, mono=True):
    # soundfile is used instead of librosa.load to avoid backend stalls on some systems.
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1 and mono:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def loudness_normalize(y, target_db=-23.0, eps=1e-8):
    rms = np.sqrt(np.mean(y ** 2) + eps)
    current_db = 20 * np.log10(rms + eps)
    gain = 10 ** ((target_db - current_db) / 20)
    y = y * gain
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def energy_vad(y, sr=16000, frame_length=1024, hop_length=256, top_db=35):
    # Lightweight RMS VAD that avoids heavy audio backends.
    if len(y) < frame_length:
        return y.astype(np.float32)
    frames = []
    keep = []
    for start in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[start:start + frame_length]
        frames.append((start, start + frame_length, float(np.sqrt(np.mean(frame ** 2) + 1e-8))))
    rms = np.array([f[2] for f in frames])
    threshold = max(np.percentile(rms, 35) * 0.5, 1e-4)
    mask = np.zeros(len(y), dtype=bool)
    for start, end, r in frames:
        if r >= threshold:
            mask[start:end] = True
    if not mask.any():
        return y.astype(np.float32)
    return y[mask].astype(np.float32)

def segment_audio(y, sr=16000, segment_seconds=5.0, overlap_seconds=2.5, drop_short=True):
    seg_len = int(segment_seconds * sr)
    hop = max(1, int((segment_seconds - overlap_seconds) * sr))
    if len(y) < seg_len:
        if drop_short:
            return []
        padded = np.zeros(seg_len, dtype=np.float32)
        padded[: len(y)] = y
        return [padded]
    segments = []
    for start in range(0, len(y) - seg_len + 1, hop):
        segments.append(y[start:start + seg_len].astype(np.float32))
    return segments


def preprocess_file(audio_path, sr=16000, segment_seconds=5.0, overlap_seconds=2.5):
    y, _ = load_audio(audio_path, target_sr=sr, mono=True)
    y = loudness_normalize(y)
    y = energy_vad(y, sr=sr)
    return segment_audio(y, sr=sr, segment_seconds=segment_seconds, overlap_seconds=overlap_seconds, drop_short=False)


def save_segments(segments, out_dir, session_id, sr=16000):
    out_dir = Path(out_dir) / str(session_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, seg in enumerate(segments):
        p = out_dir / f"segment_{i:04d}.wav"
        sf.write(p, seg, sr)
        paths.append(str(p))
    return paths
