import warnings
import numpy as np
import torch
from scipy import signal


def _frame_signal(y, frame_length=512, hop_length=256):
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    n = 1 + (len(y) - frame_length) // hop_length
    frames = np.stack([y[i*hop_length:i*hop_length+frame_length] for i in range(max(n,1))])
    return frames * np.hanning(frame_length)


def extract_acoustic_features(y, sr=16000, n_mfcc=20):
    """Fast 64-D acoustic feature vector using numpy/scipy only."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return np.zeros(64, dtype=np.float32)
    try:
        frames = _frame_signal(y)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-8)
        zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2.0
        spec = np.abs(np.fft.rfft(frames, axis=1)) + 1e-8
        freqs = np.fft.rfftfreq(frames.shape[1], 1/sr)
        spec_sum = spec.sum(axis=1) + 1e-8
        centroid = (spec * freqs).sum(axis=1) / spec_sum
        bandwidth = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * spec).sum(axis=1) / spec_sum)
        rolloff_idx = [np.searchsorted(np.cumsum(s), 0.85 * np.sum(s)) for s in spec]
        rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs)-1)]
        # Approximate pitch via autocorrelation on selected high-energy frames.
        high = frames[rms >= np.percentile(rms, 60)]
        pitches = []
        min_lag, max_lag = int(sr/500), int(sr/50)
        for fr in high[:30]:
            ac = signal.correlate(fr, fr, mode='full')[len(fr)-1:]
            if len(ac) > max_lag:
                lag = np.argmax(ac[min_lag:max_lag]) + min_lag
                if lag > 0: pitches.append(sr / lag)
        f0 = np.asarray(pitches if pitches else [0.0])
        # Spectral bands as MFCC-free compact spectral shape.
        bands = np.array_split(spec, 20, axis=1)
        band_energy = np.array([np.log(b.mean(axis=1) + 1e-8) for b in bands]).T
        feats = []
        for arr in [rms, zcr, centroid, bandwidth, rolloff, f0]:
            feats.extend([float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))])
        feats.extend(np.mean(band_energy, axis=0).tolist())
        feats.extend(np.std(band_energy, axis=0).tolist())
        # Speaking-rate proxy: number of RMS peaks per second.
        peaks, _ = signal.find_peaks(rms, distance=3)
        feats.append(len(peaks) / max(len(y)/sr, 1e-6))
    except Exception as exc:
        warnings.warn(f"Acoustic extraction failed: {exc}")
        feats = [0.0] * 64
    feats = np.asarray(feats, dtype=np.float32)
    if len(feats) < 64:
        feats = np.pad(feats, (0, 64 - len(feats)))
    return feats[:64].astype(np.float32)


class SSLEmbeddingExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base", device="cpu", enabled=True):
        self.enabled = enabled
        self.device = torch.device(device)
        self.processor = None
        self.model = None
        if enabled:
            try:
                from transformers import AutoFeatureExtractor, AutoModel
                self.processor = AutoFeatureExtractor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
            except Exception as exc:
                warnings.warn(f"Could not load SSL model '{model_name}'. Falling back to deterministic audio statistics. Error: {exc}")
                self.enabled = False

    @torch.no_grad()
    def extract(self, y, sr=16000):
        if self.enabled and self.model is not None:
            inputs = self.processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs).last_hidden_state.squeeze(0)
            return out.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        base = extract_acoustic_features(y, sr=sr)
        reps = int(np.ceil(768 / len(base)))
        return np.tile(base, reps)[:768].astype(np.float32)
