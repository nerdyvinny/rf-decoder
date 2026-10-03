# fm.py
# FM demodulation: turns the radio samples into the station's audio.
#
# An FM station keeps its signal the same strength and moves its frequency
# up and down with the sound (up to +-75 kHz). To get the sound back, measure
# how fast the frequency is changing at every sample.
#
# What comes out (the "baseband") has a few things stacked in it:
#   0-15 kHz    mono audio
#   19 kHz      stereo pilot tone
#   23-53 kHz   stereo part (left minus right)
#   57 kHz      RDS data (see rds.py)

import wave

import numpy as np

MAX_DEVIATION = 75000  # Hz, the biggest frequency swing a US FM station uses
AUDIO_RATE = 50000  # sample rate of the audio we save


def demodulate(iq, sample_rate):
    # The angle between each sample and the one before it is how far the phase
    # moved, and how fast the phase moves is the frequency.
    #
    # Note: no DC offset removal here. When you tune right onto a station, the
    # station itself is at 0 Hz, so removing "DC" cuts into it. (It put a fake
    # 3 kHz tone in the audio.)
    phase_change = np.angle(iq[1:] * np.conj(iq[:-1]))
    return phase_change * sample_rate / (2 * np.pi * MAX_DEVIATION)


def lowpass_filter(cutoff_hz, sample_rate, num_taps=129):
    # windowed sinc low pass filter
    n = np.arange(num_taps) - (num_taps - 1) / 2
    taps = np.sinc(2 * cutoff_hz / sample_rate * n) * np.hamming(num_taps)
    return taps / np.sum(taps)


def get_audio(baseband, sample_rate):
    # keep only the mono audio (under 15 kHz) and lower the sample rate
    filtered = np.convolve(baseband, lowpass_filter(15000, sample_rate), mode="same")
    audio = filtered[::sample_rate // AUDIO_RATE]

    # US stations boost the treble before sending (75 microsecond "pre-emphasis"),
    # so turn it back down with a simple one pole low pass filter
    alpha = 1 - np.exp(-1 / (AUDIO_RATE * 75e-6))
    out = np.zeros(len(audio))
    y = 0.0
    for i, x in enumerate(audio.tolist()):
        y = y + alpha * (x - y)
        out[i] = y
    return out


def save_wav(filename, audio):
    # 16 bit mono wav, scaled so the loudest part is at 90%
    peak = np.max(np.abs(audio))
    if peak == 0:
        peak = 1
    samples = (audio / peak * 0.9 * 32767).astype(np.int16)
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(AUDIO_RATE)
        f.writeframes(samples.tobytes())
