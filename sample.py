import os
import sys
import pydub
import argparse
import numpy as np
import scipy as sp

def spectrum(segment, sample_rate, n_freq):
    n = len(segment)
    fft_freq = np.fft.rfftfreq(n, d=1 / sample_rate)
    dst_result = sp.fft.dst(segment, type=2, norm="ortho")[:len(fft_freq)]
    dst_result /= np.max(dst_result)
    result = np.column_stack((fft_freq, dst_result))
    result = result[result[:, 1] > 0]
    result = result[result[:, 1].argsort()[::-1]]
    result = result[:n_freq]
    # result = result[result[:, 0].argsort()]
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Segment and sample an audio file.")
    parser.add_argument("-i", "--input", required=True, help="The input to segment")
    parser.add_argument("-s", "--start", type=int, default=0, help="The start time in ms for the segment")
    parser.add_argument("-e", "--end", type=int, default=sys.maxsize, help="The end time in ms for the segment")
    parser.add_argument("-n", "--n-freq", type=int, default=256, help="The total amount of frequency to sample")
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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
