# tests for fm.py, using a fake FM signal so we know exactly what should come out

import wave

import numpy as np

import fm

RATE = 250000


def make_fm(tone_hz, seconds=0.5, swing=0.5, noise=0.02):
    # an FM signal swinging +-(swing x 75 kHz) at tone_hz, plus some noise
    t = np.arange(int(seconds * RATE)) / RATE
    freq = swing * fm.MAX_DEVIATION * np.sin(2 * np.pi * tone_hz * t)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    rng = np.random.default_rng(0)
    return np.exp(1j * phase) + noise * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))


def loudest_freq(signal, rate):
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1 / rate)
    return freqs[np.argmax(spectrum[1:]) + 1]


def test_demodulate_gets_the_tone_back():
    baseband = fm.demodulate(make_fm(1000), RATE)
    assert abs(loudest_freq(baseband, RATE) - 1000) < 5
    # it swung half of the max, so it should come out at about 0.5
    assert abs(np.percentile(np.abs(baseband), 99.9) - 0.5) < 0.05


def test_audio_keeps_the_sound_and_drops_the_pilot():
    audio = fm.get_audio(fm.demodulate(make_fm(1000), RATE), RATE)
    assert abs(loudest_freq(audio, fm.AUDIO_RATE) - 1000) < 5
    # a 19 kHz tone (like the stereo pilot) should not make it into the audio
    pilot_only = fm.get_audio(fm.demodulate(make_fm(19000, swing=0.1), RATE), RATE)
    assert np.std(pilot_only[1000:]) < 0.01


def test_wav_file(tmp_path):
    path = str(tmp_path / "tone.wav")
    fm.save_wav(path, fm.get_audio(fm.demodulate(make_fm(440), RATE), RATE))
    with wave.open(path) as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        assert f.getframerate() == fm.AUDIO_RATE
        assert f.getnframes() > 20000
