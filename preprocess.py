# audio specific imports
from scipy.fftpack import fft
from scipy import signal
import librosa as lr
import python_speech_features as psf
import scipy.io.wavfile as wav


from plotly.subplots import make_subplots
import IPython.display as ipd
import numpy as np
import os
import pandas as pd 
import plotly.express as ex 
import torch

DATA_DIR = "../../../../data/gtzan_music_dataset/"


def file_to_features(file_path, verbose=1):
    try:
        (rate, sig) = wav.read(file_path)
        mfcc_feat = psf.mfcc(sig, rate, winlen=0.020, appendEnergy=False)  # what does appendEnergy do?
        covariance = np.cov(np.matrix.transpose(mfcc_feat)) # covariance = second moments of a distribution
        mean_matrix = mfcc_feat.mean(0)  # mean along the axis 0 (rows?)
        feature = (mean_matrix, covariance)
        return {
            "file_path": file_path,
            "rate": rate,
            "signal": sig,
            "mfcc_feat": mfcc_feat,
            "covariance": covariance,
            "mean_matrix": mean_matrix,
            "feature": feature
        }
    except:
        print("Error processing " + file_path)
        return None
    

def genre_to_features(data_dir, genre, verbose=1):
    # get all the files from the dataset
    file_paths = []
    rates = []
    signals = []
    mfcc_feats = []
    covariances = []
    mean_matrices = []

    features = []
    labels = []

    for file in os.listdir(os.path.join(data_dir, genre)):  
        file_path = os.path.join(data_dir, genre, file)
        print(file) if verbose > 1 else None
        file_features = file_to_features(file_path, verbose=verbose)

        if file_features is not None:
            file_paths.append(file_features["file_path"])
            rates.append(file_features["rate"])
            signals.append(file_features["signal"])
            mfcc_feats.append(file_features["mfcc_feat"])
            covariances.append(file_features["covariance"])
            mean_matrices.append(file_features["mean_matrix"])

            features.append(file_features["feature"])
            labels.append(genre)

    return {
        "file_paths": file_paths, 
        "rates":rates, 
        "signals": signals, 
        "mfcc_feats": mfcc_feats, 
        "covariances": covariances, 
        "mean_matrices": mean_matrices, 

        # actually useful for training
        "features": features,
        "labels": labels
    }


def load_genre_features(data_dir, verbose=1):
    # get all the files from the dataset
    genre_features = {}

    for i, genre in enumerate(os.listdir(data_dir)):
        print(i, "Loading and featurizing", genre) if verbose else None
        genre_features[genre] = genre_to_features(data_dir, genre, verbose=verbose)

        # if i > :
        #     break # for testing

    return genre_features


def resample(signal, n_bits=16):
    return signal / 2 ** (n_bits - 1)


def get_feature_tensor(genre_features, feature="mfcc_feats", verbose=0, max_dim=2986): # 2986 is lowest mfcc len in our dataset
    feature_tensor = None
    file_paths = []
    labels = []

    for genre in genre_features:
        print("Building", feature, "for", genre) if verbose > 0 else None
        try:
            # needed to catch lengths smaller than max_dim
            genre_feature_tensor = torch.Tensor(np.array([
                row[:max_dim]  # for safe stacking
                for row in genre_features[genre][feature]
            ]))
        except:
            print("Error building", feature, "for", genre)
            print("Feature lengths", set([
                len(row[:max_dim])  
                for row in genre_features[genre][feature]
            ]))
            continue

        file_paths += genre_features[genre]["file_paths"]
        labels += [genre for _ in range(len(genre_features[genre]["file_paths"]))]

        if feature_tensor is None:
            feature_tensor = genre_feature_tensor
        else:
            feature_tensor = torch.cat([feature_tensor, genre_feature_tensor], axis=0)
        print("feature_tensor shape:", feature_tensor.shape) if verbose > 1 else None

    file_paths = np.array(file_paths)
    labels = np.array(labels)

    print("feature_tensor shape:", feature_tensor.shape) if verbose > 0 else None
    print("file_paths shape:", file_paths.shape) if verbose > 0 else None
    print("labels shape:", labels.shape) if verbose > 0 else None

    return feature_tensor, file_paths, labels


def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    """This is a lot of data per signal, so we'll skip"""
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(
        audio,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False
    )
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


if __name__ == "__main__":
    # load 
    genre_dir = DATA_DIR + "genres_original"
    genre_features = load_genre_features(genre_dir, verbose=1)

    # build mfcc tensor
    mfcc_tensor, file_paths, labels = get_feature_tensor(genre_features, feature="mfcc_feats", verbose=1)

    # build covariance tensor
    # covariance is like the QK step of an attention head
    covariance_tensor, _, _ = get_feature_tensor(genre_features, feature="covariances", verbose=1)

    # store
    torch.save(mfcc_tensor, "mfcc.pt")
    torch.save(covariance_tensor, "covariance.pt")
    np.save("file_paths.npy", file_paths)
    np.save("labels.npy", labels)


