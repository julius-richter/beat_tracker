import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from librosa import clicks
import IPython.display as ipd
import pickle
import torch


def get_audio(file, normalize=False):
    _, signal = wavfile.read('../data/audio/' + file + '.wav', mmap=False)
    if normalize:
        signal = signal / np.max(np.abs(signal))
    return signal


def get_signal(file, normalize=False):
    _, signal = wavfile.read('../data/audio/' + file + '.wav', mmap=False)
    if normalize:
        signal = signal / np.max(np.abs(signal))
    return signal


def get_sample_rate(file, normalize=False):
    sr, _ = wavfile.read('../data/audio/' + file + '.wav', mmap=False)
    return sr


def play_audio(file, normalize=False):
    sr, signal = wavfile.read('../data/audio/' + file + '.wav', mmap=False)
    
    if normalize:
        signal = signal / np.max(np.abs(signal))    
        
    return ipd.Audio(signal, rate=sr)


def get_input(file):
    return pickle.load(open('../data/inputs/'+ file + '.npy', 'rb'))


def get_annotations(file):
    return np.loadtxt('../data/annotations/' + file + '.beats', ndmin=2)[:, 0]


def get_predictions(file):
    return np.loadtxt('../data/predictions/' + file + '.beats')


def play_annotations(file, normalize=True):
    sr, signal = wavfile.read('../data/audio/' + file + '.wav', mmap=False)
    
    if normalize:
        signal = signal / np.max(np.abs(signal))
        
    metronome = clicks(get_annotations(file), sr=sr, length=len(signal))

    signal = signal + metronome    
    signal = signal / np.max(np.abs(signal))
        
    return ipd.Audio(signal, rate=sr)


def play_predictions(file, normalize=True):
    sr, signal = wavfile.read('../data/audio/' + file + '.wav', mmap=False)
    
    if normalize:
        signal = signal / np.max(np.abs(signal))
        
    metronome = clicks(get_predictions(file), sr=sr, length=len(signal))

    signal = signal + metronome
        
    signal = signal / np.max(np.abs(signal))
        
    return ipd.Audio(signal, rate=sr)


def get_activations(file, model):
    input = get_input(file)

    with torch.no_grad():
        out = model(input.view(1, len(input),-1))

    return np.exp(np.array(out[0,1,:]))


def show_activations(file, model):
    
    input = get_input(file)
    annotations = get_annotations(file)
    predictions = get_predictions(file)

    with torch.no_grad():
        out = model(input.view(1, len(input),-1))

    activations = np.exp(np.array(out[0,1,:]))

    plt.figure(figsize=(9,2))

    plt.plot(activations)
    plt.xlabel('Frames')
    plt.ylabel('Beat activation')
#     plt.title('Example: {}   F-measure: {:.3f}'.format(file, data.at[index,'f_measure']))

    for ann in annotations:
        plt.axvline(x=ann*100, color='k', linestyle=':', linewidth=2)

    for ann in predictions:
        plt.axvline(x=ann*100, color='r', linestyle=':', linewidth=2)