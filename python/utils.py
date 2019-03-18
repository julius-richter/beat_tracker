import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import IPython.display as ipd
import pickle
import torch


def clicks(times=None, sr=44100, length=None):
    positions = (np.asanyarray(times) * sr).astype(int)

    click_freq=1000.0
    click_duration=0.1

    angular_freq = 2 * np.pi * click_freq / float(sr)

    click = np.logspace(0, -10,
                        num=int(np.round(sr * click_duration)),
                        base=2.0)

    click *= np.sin(angular_freq * np.arange(len(click)))

    if length is None:
        length = positions.max() + click.shape[0]
    else:
        positions = positions[positions < length]

    click_signal = np.zeros(length, dtype=np.float32)

    for start in positions:
        end = start + click.shape[0]

        if end >= length:
            click_signal[start:] += click[:length - start]
        else:
            click_signal[start:end] += click

    return click_signal


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def get_input(file, subfolder = None):
    if subfolder:
        return pickle.load(open('../data/inputs/'+ subfolder + '/' + file + '.npy', 'rb'))
    else:
        return pickle.load(open('../data/inputs/'+ file + '.npy', 'rb'))


def get_labels(file):
    return pickle.load(open('../data/labels/'+ file + '.npy', 'rb'))


def get_annotations(file):
    return np.loadtxt('../data/annotations/' + file + '.beats', ndmin=2)[:, 0]


def get_predictions(file, subfolder = None):
    if subfolder:
        return np.loadtxt('../data/predictions/' + subfolder + '/' + file + '.beats')
    else:
        return np.loadtxt('../data/predictions/' + file + '.beats')


def play_annotations(file, section=None, normalize=True):
    sr, signal = wavfile.read('../data/audio/' + file + '.wav', mmap=False)
    

    if normalize:
        signal = signal / np.max(np.abs(signal))
        
    metronome = clicks(get_annotations(file), sr=sr, length=len(signal))

    signal = signal + metronome    
    signal = signal / np.max(np.abs(signal))

    if section:
        signal = signal[int(sr/100*section[0]):int(sr/100*section[1])]
        
    return ipd.Audio(signal, rate=sr)


def play_predictions(file, subfolder = None, section=None, normalize=True):
    sr, signal = wavfile.read('../data/audio/' + file + '.wav', mmap=False)
    
    if normalize:
        signal = signal / np.max(np.abs(signal))
        
    metronome = clicks(get_predictions(file, subfolder), sr=sr, length=len(signal))

    signal = signal + metronome
        
    signal = signal / np.max(np.abs(signal))

    if section:
        signal = signal[int(sr/100*section[0]):int(sr/100*section[1])]
        
    return ipd.Audio(signal, rate=sr)


def get_activations(file, model):
    input = get_input(file)

    with torch.no_grad():
        out = model(input.view(1, len(input),-1))

    return np.exp(np.array(out[0,1,:]))


def show_activations(file, model, subfolder = None):
    
    input = get_input(file)
    annotations = get_annotations(file)

    predictions = get_predictions(file, subfolder)


    with torch.no_grad():
        out = model(input.view(1, len(input),-1))

    activations = np.exp(np.array(out[0,1,:]))

    plt.figure(figsize=(9,2))

    plt.plot(activations)
    plt.xlabel('Frames')
    plt.ylabel('Beat activation')

    for ann in annotations:
        plt.axvline(x=ann*100, color='k', linestyle=':', linewidth=2)

    for ann in predictions:
        plt.axvline(x=ann*100, color='r', linestyle=':', linewidth=2)