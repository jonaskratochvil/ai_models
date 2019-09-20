#!/usr/bin/env python3

from scipy.io import wavfile
#import utility
from scipy.signal import fftconvolve


def similarity(template, test):
    corr = fftconvolve(template, test, mode='same')

    return corr


# max(abs(corr))
fs, sig = wavfile.read('file1_new.wav')
fs2, sig2 = wavfile.read('file2_new.wav')

#normalized = utility.pcm2float(sig, 'float32')
#normalized2 = utility.pcm2float(sig2, 'float32')

# print(len(sig))
print(sig[-1000:])


print(sig2[:1000])
