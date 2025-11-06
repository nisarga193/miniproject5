"""
Generate a tiny synthetic audio dataset for demo/training.
Creates simple tones and chirps for three classes: dog, cat, bird.
Outputs WAV files into data/<class>/*.wav

This script uses only the standard library and numpy (numpy is common). It writes 16-bit PCM WAV files.
"""
import os
import argparse
import math
import wave
import struct
import random

try:
    import numpy as np
except Exception as e:
    print('numpy required to run this script')
    raise


def write_wav(path, samples, sr=22050):
    # samples: float32 array in -1..1
    samples_i16 = (samples * 32767).astype('<i2')
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_i16.tobytes())


def gen_tone(freq, dur, sr=22050, amplitude=0.6):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    return amplitude * np.sin(2*np.pi*freq*t)


def gen_chirp(f0, f1, dur, sr=22050, amplitude=0.6):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    freqs = np.linspace(f0, f1, t.size)
    phase = 2 * np.pi * np.cumsum(freqs) / sr
    return amplitude * np.sin(phase)


def gen_noise(dur, sr=22050, amplitude=0.2):
    return amplitude * np.random.randn(int(sr*dur)).astype(np.float32)


def make_samples_for_class(cls, dur=2.0, sr=22050):
    if cls == 'dog':
        # low bark-like pulses: repeated bursts of low tone
        base = gen_tone(200, dur, sr)
        for _ in range(3):
            start = random.uniform(0, dur-0.2)
            idx0 = int(start*sr); idx1 = min(idx0+int(0.15*sr), base.size)
            base[idx0:idx1] += gen_tone(400, (idx1-idx0)/sr, sr, amplitude=1.0)
        base += 0.06 * gen_noise(dur, sr)
        return base
    if cls == 'cat':
        # higher meow-like frequency sweeps
        s = gen_chirp(400, 900, dur, sr, amplitude=0.6)
        s *= np.hanning(s.size)
        s += 0.03 * gen_noise(dur, sr)
        return s
    if cls == 'bird':
        # short high chirps at intervals
        out = np.zeros(int(sr*dur), dtype=np.float32)
        for i in range(5):
            start = int(random.uniform(0, dur-0.2)*sr)
            chirp = gen_chirp(1000, 3000, 0.12, sr, amplitude=0.5)
            end = start + chirp.size
            if end > out.size: continue
            out[start:end] += chirp * np.hanning(chirp.size)
        out += 0.02 * gen_noise(dur, sr)
        return out
    # fallback
    return gen_noise(dur, sr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='data')
    p.add_argument('--per_class', type=int, default=30)
    p.add_argument('--sr', type=int, default=22050)
    p.add_argument('--dur', type=float, default=2.0)
    args = p.parse_args()

    classes = ['dog', 'cat', 'bird']
    os.makedirs(args.out_dir, exist_ok=True)
    for cls in classes:
        cls_dir = os.path.join(args.out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        print('Creating', args.per_class, 'samples for', cls)
        for i in range(args.per_class):
            s = make_samples_for_class(cls, dur=args.dur, sr=args.sr)
            # normalize
            s = s / (max(1e-5, np.max(np.abs(s))))
            fname = os.path.join(cls_dir, f'{cls}_{i:03d}.wav')
            write_wav(fname, s, sr=args.sr)
    print('Done. Data in', os.path.abspath(args.out_dir))

if __name__ == '__main__':
    main()
