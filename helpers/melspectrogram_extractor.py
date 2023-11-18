import argparse
import os
from pathlib import Path

import numpy as np
from essentia.pytools.extractors.melspectrogram import melspectrogram


SR = 16000
HOP_SIZE = 256
N_MELS = 96
FRAME_SIZE = 512


def melspectrorgam_extractor(audio_file):
    return melspectrogram(
        audio_file,
        sample_rate=SR,
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        window_type='hann',
        low_frequency_bound=0,
        high_frequency_bound=SR / 2,
        number_bands=N_MELS,
        warping_formula='slaneyMel',
        weighting='linear',
        normalize='unit_tri',
        bands_type='power',
        compression_type='shift_scale_log'
    ).astype('float16')


def main(audio_file, melbands_file, force=False, max_duration=300):
    print(f'Processing {audio_file}')
    print(f'Writing to {melbands_file}')
    print("exists:", os.path.exists(melbands_file))
    if not os.path.exists(melbands_file) or force:
        try:
            melspectrogram = melspectrorgam_extractor(audio_file)
            max_timestamps = int(max_duration * SR / HOP_SIZE)

            # trim to max duration
            if len(melspectrogram) > max_timestamps:
                mid = len(melspectrogram) // 2
                melspectrogram = melspectrogram[mid - max_timestamps // 2: mid + max_timestamps // 2, :]

            # write as raw bytes
            Path(melbands_file).parent.mkdir(parents=True, exist_ok=True)
            print(f'Writing to {melbands_file}')
            print(melspectrogram.shape)
            # fp = np.memmap(melbands_file, dtype='float16', mode='w+', shape=melspectrogram.shape)
            # fp[:] = melspectrogram[:]
            with open(melbands_file, 'wb') as f:
                # melspectrogram.tofile(f)
                np.save(f, melspectrogram)
                # melspectrogram.sa(f)

            print(f'Wrote {melbands_file}')
        except RuntimeError as e:
            print(f'Error while processing {audio_file}')
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computes the mel spectrogram of a given audio file.')

    parser.add_argument('--audio_file', default="/Users/per.morten.halvorsen@schibsted.com/data/gtzan_music_dataset/genres_original/blues/blues.00000.wav", type=str,
                        help='the name of the file from which to read')
    parser.add_argument('--melbands_file', type=str, default="/Users/per.morten.halvorsen@schibsted.com/personal/public-data/melbands.npy",
                        help='the name of the output file')
    parser.add_argument('--force', '-f', action='store_true',
                        help='force')
    parser.add_argument('--max-duration', type=float, default=300,
                        help='max duration in seconds')

    main(**vars(parser.parse_args()))