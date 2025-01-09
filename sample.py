import os
import sys
import pydub
import argparse
import numpy as np

def spectrum(segment, sample_rate, n_freq):
    n = len(segment)
    fft_freq = np.fft.rfftfreq(n, d=1 / sample_rate).astype(np.float32)
    fft_result = np.abs(np.fft.rfft(segment)[:len(fft_freq)])
    fft_result /= fft_result.max()
    fft_result = np.column_stack((fft_freq, fft_result))
    prev_freq = 0
    prev_mag = 0
    prev_diff = 0
    result = []
    for freq, mag in fft_result:
        if freq == 0:
            continue
        diff = mag - prev_mag
        if prev_diff > 0 and diff < 0:
            result.append((prev_freq, prev_mag, prev_diff))
        prev_freq = freq
        prev_mag = mag
        prev_diff = diff
    result.sort(key=lambda x: x[1], reverse=True)
    return np.array(result[:n_freq])

def parse_args():
    parser = argparse.ArgumentParser(description="Segment and sample an audio file.")
    parser.add_argument("-i", "--input", required=True, help="The input to segment")
    parser.add_argument("-s", "--start", type=int, default=0, help="The start time in ms for the segment")
    parser.add_argument("-e", "--end", type=int, default=sys.maxsize, help="The end time in ms for the segment")
    parser.add_argument("-n", "--n-freq", type=int, default=64, help="The total amount of frequency to sample")
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
    segment = audio[args.start:args.end]
    sample_rate = segment.frame_rate
    segment = np.array(segment.get_array_of_samples()).reshape(segment.channels, -1, order="F")
    segment = np.mean(segment, axis=0)
    segment = (segment / np.max(np.abs(segment))).astype(np.float32)
    result = spectrum(segment, sample_rate, args.n_freq)
    print("a=[", end="")
    for i, freq in enumerate(result[:, 0]):
        print(f"{"" if i == 0 else ","}{freq:.2f}".strip("0."), end="")
    print("]\nb=[", end="")
    for i, gain in enumerate(result[:, 1]):
        print(f"{"" if i == 0 else ","}{gain:.3f}".strip("0."), end="")
    print("]")
    for i, diff in enumerate(result[:, 2]):
        print(f"{"" if i == 0 else ","}{diff:.3f}".strip("0."), end="")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
