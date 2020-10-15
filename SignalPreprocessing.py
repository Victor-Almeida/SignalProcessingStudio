import librosa
import numpy as np
import math

class SignalPreprocessing():
  def __init__(self):
    pass

  def to_filter_bank(file_path, 
                    n_fft = 256, 
                    n_mels = 40, 
                    sr = None, 
                    hop = 0.01, 
                    window = 0.025, 
                    emphasis = 0.97, 
                    file_type = np.float32, 
                    window = "hamming", 
                    center = False,
                    return_type = np.float32):

    data, sr = librosa.load(file_path, sr = sr, dtype = file_type)
    emphasized_signal = librosa.effects.preemphasis(data, coef=emphasis)
    hop_length = math.ceil(hop * sr)
    window_length = math.ceil(window * sr)

    filter_banks = librosa.feature.melspectrogram(y=emphasized_signal, 
                                                  sr=sr, 
                                                  n_fft=n_fft, 
                                                  hop_length=hop_length, 
                                                  win_length=window_length, 
                                                  window=window, 
                                                  center=False,
                                                  n_mels=n_mels,
                                                  dtype = np.float128)
    fb = librosa.power_to_db(filter_banks, ref=np.max)
    mean_fb = (fb - np.mean(fb, axis = 0))/(fb.std(axis = 0) + 1e-20)
    mean_fb = mean_fb.T

    return mean_fb.astype(return_type, casting = 'same_kind')

  def to_mfcc(file_path
              n_fft = 256, 
              n_mels = 40, 
              n_mfcc = 13,
              lifter = 22,
              sr = None, 
              hop = 0.01, 
              window = 0.025, 
              emphasis = 0.97, 
              file_type = np.float32, 
              window = "hamming", 
              center = False,
              return_type = np.float32):

    data, sr = librosa.load(file_path, sr = sr, dtype = file_type)
    emphasized_signal = librosa.effects.preemphasis(data, coef=emphasis)
    hop_length = math.ceil(hop * sr)
    window_length = math.ceil(window * sr)

    mfcc = librosa.feature.mfcc(y=emphasized_signal, 
                                sr=sr, 
                                n_fft=n_fft, 
                                n_mels = n_mels,
                                n_mfcc = n_mfcc,
                                hop_length=hop_length, 
                                win_length=window_length, 
                                window=window, 
                                lifter = lifter,
                                center = False,
                                dtype = np.float128)

    delta_mfcc = librosa.feature.delta(mfcc)
    delta_delta_mfcc = librosa.feature.delta(mfcc, order = 2)
    MFCC = np.vstack([mfcc, delta_mfcc, delta_delta_mfcc])
    mean_mfcc = (MFCC - np.mean(MFCC, axis = 0))/(MFCC.std(axis = 0) + 1e-20)
    mean_mfcc = mean_mfcc.T

    return mean_mfcc.astype(return_type, casting = 'same_kind')