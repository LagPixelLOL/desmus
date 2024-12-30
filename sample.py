import os
import sys
import pydub
import argparse
import numpy as np
import scipy as sp

def mel_scale_spectrum(segment, sample_rate, n_freq):
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    min_mel = hz_to_mel(20)
    max_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(min_mel, max_mel, n_freq, dtype=np.float32)
    hz_points = mel_to_hz(mel_points)
    n = len(segment)
    dst_result = sp.fft.dst(segment, type=2, norm="ortho")
    fft_freq = np.fft.rfftfreq(n, d=1 / sample_rate)
    idx = np.searchsorted(fft_freq, hz_points)
    idx = np.clip(idx, 0, len(fft_freq) - 1)
    mel_magnitudes = dst_result[idx]
    mel_magnitudes /= np.max(np.abs(mel_magnitudes))
    result = np.column_stack((hz_points, mel_magnitudes))
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Segment and sample an audio file.")
    parser.add_argument("-i", "--input", required=True, help="The input to segment")
    parser.add_argument("-s", "--start", type=int, default=0, help="The start time in ms for the segment")
    parser.add_argument("-e", "--end", type=int, default=sys.maxsize, help="The end time in ms for the segment")
    parser.add_argument("-n", "--n-freq", type=int, default=1024, help="The total amount of frequency to sample")
    args = parser.parse_args()
    if args.start < 0:
        print("Start ms cannot be negative!")
        sys.exit(2)
    if args.start >= args.end:
        print("Start ms must be less than end ms!")
        sys.exit(2)
    if args.n_freq < 1:
        print("Amount of frequency must be at least 1!")
        sys.exit(2)
    return args

def main():
    args = parse_args()
    audio = pydub.AudioSegment.from_file(args.input)
    args.end = min(args.end, len(audio))
    segment = audio[args.start:args.end]
    sample_rate = segment.frame_rate
    segment = np.array(segment.get_array_of_samples()).reshape(segment.channels, -1, order="F")
    segment = np.mean(segment, axis=0)
    segment = (segment / np.max(np.abs(segment))).astype(np.float32)
    result = mel_scale_spectrum(segment, sample_rate, args.n_freq)
    result = result[result[:, 1] >= 5e-4]
    unique_indices = np.unique(result[:, 0], return_index=True)[1]
    result = result[unique_indices]
    print("a=[", end="")
    for i, freq in enumerate(result[:, 0]):
        print(f"{"" if i == 0 else ","}{freq:.5f}", end="")
    print("]\nb=[", end="")
    for i, gain in enumerate(result[:, 1]):
        print(f"{"" if i == 0 else ","}{gain:.5f}", end="")
    print("]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
