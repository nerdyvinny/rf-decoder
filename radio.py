# radio.py
# Talks to the RTL-SDR dongle through librtlsdr (rtlsdr.dll) with ctypes.
# The dongle only gives back raw samples, everything else is done in fm.py and rds.py.

import ctypes
import os

import numpy as np

# folder where the rtl-sdr windows release was unzipped (can be changed with RTLSDR_DIR)
DLL_FOLDER = os.environ.get("RTLSDR_DIR", os.path.join(os.path.expanduser("~"), "tools", "rtl-sdr", "x64"))

# how many bytes to read at a time (same block size rtl_sdr.exe uses)
BLOCK_SIZE = 16 * 16384


def load_library():
    # the dll needs its neighbours (pthreadVC2.dll, msvcr100.dll) so add the folder first
    os.add_dll_directory(DLL_FOLDER)
    return ctypes.CDLL(os.path.join(DLL_FOLDER, "rtlsdr.dll"))


def open_radio(freq_hz, sample_rate, gain_db):
    lib = load_library()
    if lib.rtlsdr_get_device_count() == 0:
        raise RuntimeError("no RTL-SDR dongle found, is it plugged in?")

    dev = ctypes.c_void_p()
    if lib.rtlsdr_open(ctypes.byref(dev), 0) != 0:
        raise RuntimeError("couldn't open the dongle (close SDR# or anything else using it)")

    if lib.rtlsdr_set_sample_rate(dev, sample_rate) != 0:
        raise RuntimeError("couldn't set the sample rate")
    if lib.rtlsdr_set_center_freq(dev, freq_hz) != 0:
        raise RuntimeError("couldn't tune to " + str(freq_hz) + " Hz")

    # This driver turns on "direct sampling" (tuner bypassed) whenever nothing is tuned,
    # and switches it off on the first tune. Make sure it really is off.
    if lib.rtlsdr_get_direct_sampling(dev) != 0:
        raise RuntimeError("tuner is still bypassed (direct sampling) after tuning")

    # Switching direct sampling off resets the tuner, which wipes anything set before
    # the tune. So the gain has to be set AFTER tuning. (Set before, the gain did
    # nothing: the signal level was the same from 0 to 49.6 dB.)
    if hasattr(lib, "rtlsdr_set_bias_tee"):
        lib.rtlsdr_set_bias_tee(dev, 0)  # the eeprom on this dongle says bias tee always on
    lib.rtlsdr_set_tuner_gain_mode(dev, 1)  # 1 = manual gain
    lib.rtlsdr_set_tuner_gain(dev, round(gain_db * 10))  # gain is in tenths of a dB
    return lib, dev


def read_raw(lib, dev, num_samples):
    # Returns num_samples samples as raw bytes (I, Q, I, Q, ...).
    # Old data can sit in the dongle's buffer, so clear it and throw away the first block.
    lib.rtlsdr_reset_buffer(dev)
    buf = (ctypes.c_uint8 * BLOCK_SIZE)()
    n_read = ctypes.c_int()
    chunks = []
    total = 0
    first = True
    while total < num_samples * 2:
        if lib.rtlsdr_read_sync(dev, buf, BLOCK_SIZE, ctypes.byref(n_read)) != 0:
            raise RuntimeError("reading from the dongle failed (unplugged?)")
        if first:
            first = False
            continue
        chunks.append(bytes(buf)[:n_read.value])
        total += n_read.value
    return b"".join(chunks)[:num_samples * 2]


def close_radio(lib, dev):
    lib.rtlsdr_close(dev)


def bytes_to_iq(raw):
    # The dongle sends unsigned bytes centred on 127.5. Turn them into complex numbers
    # between about -1 and 1 (I is the real part, Q is the imaginary part).
    data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    data = (data - 127.5) / 127.5
    return data[0::2] + 1j * data[1::2]
