# main.py
# FM radio decoder. Reads the hidden RDS data that FM stations send (call letters,
# station name, the song that's playing) using an RTL-SDR dongle.
#
#   python main.py 92.3            listen for 30 seconds and show what the station is sending
#   python main.py 92.3 --live     keep listening and print each new song (Ctrl+C to stop)
#   python main.py 92.3 --wav      save 30 seconds of the station's audio to a wav file
#   python main.py --file recordings/fm-92.3.cu8     decode a saved recording instead
#
# Close SDR# first, only one program can use the dongle at a time.

import argparse
import os
from collections import Counter
from datetime import datetime

import numpy as np

import fm
import radio
import rds

SAMPLE_RATE = 250000
RECORDINGS_FOLDER = "recordings"


def show_station(station):
    if station["pi"] is None:
        print("no RDS data found (the signal might be too weak)")
        return

    letters = rds.call_letters(station["pi"]) or "?"
    print(f"call letters:  {letters}  (station code 0x{station['pi']:04X})")
    print(f"program type:  {rds.program_type(station['pty'])}")

    # A lot of stations scroll their 8 letter name like a banner, so show every
    # piece that came through at least twice (once only = half old, half new).
    counts = Counter(station["names"])
    names = []
    for name in station["names"]:
        if counts[name] >= 2 and name.strip() and name.strip() not in names:
            names.append(name.strip())
    if not names:
        names = [name.strip() for name in counts if name.strip()]
    print(f"station name:  {' | '.join(names) if names else '-'}")

    texts = Counter(station["texts"]).most_common()
    if not texts:
        print("radio text:    -")
    for text, count in texts:
        print(f"radio text:    {text}  ({count}x)")

    if station["clock"] is not None:
        time, offset = station["clock"]
        print(f"station clock: {time:%Y-%m-%d %H:%M} (UTC{offset:+g})")

    good = station["good_blocks"]
    total = station["total_blocks"]
    print(f"data quality:  {good:,} of {total:,} blocks readable ({100 * good / max(total, 1):.0f}%)")


def decode_and_show(raw):
    baseband = fm.demodulate(radio.bytes_to_iq(raw), SAMPLE_RATE)
    try:
        station = rds.decode_station(baseband, SAMPLE_RATE)
    except ValueError as error:
        print(error)
        return
    show_station(station)


def record(freq_mhz, seconds, gain):
    lib, dev = radio.open_radio(round(freq_mhz * 1e6), SAMPLE_RATE, gain)
    try:
        return radio.read_raw(lib, dev, int(seconds * SAMPLE_RATE))
    finally:
        radio.close_radio(lib, dev)


def listen_once(freq_mhz, seconds, gain, save):
    print(f"listening to {freq_mhz} MHz for {seconds:g} seconds...")
    raw = record(freq_mhz, seconds, gain)
    if save:
        os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
        path = os.path.join(RECORDINGS_FOLDER, f"fm-{freq_mhz}-{datetime.now():%Y%m%d-%H%M%S}.cu8")
        with open(path, "wb") as f:
            f.write(raw)
        print("saved the raw recording to", path)
    decode_and_show(raw)


def save_audio(freq_mhz, seconds, gain):
    print(f"recording {seconds:g} seconds of {freq_mhz} MHz...")
    raw = record(freq_mhz, seconds, gain)
    audio = fm.get_audio(fm.demodulate(radio.bytes_to_iq(raw), SAMPLE_RATE), SAMPLE_RATE)
    os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
    path = os.path.join(RECORDINGS_FOLDER, f"fm-{freq_mhz}.wav")
    fm.save_wav(path, audio)
    print("saved the audio to", path)

    samples = np.frombuffer(raw, dtype=np.uint8)
    clipped = np.mean((samples == 0) | (samples == 255))
    if clipped > 0.01:
        print(f"note: {100 * clipped:.0f}% of the samples are clipping, try a lower --gain")


def listen_live(freq_mhz, gain):
    # Listens 10 seconds at a time, decodes it, and prints anything new.
    # While it decodes, a few seconds of signal get skipped, which doesn't matter
    # because stations repeat their text over and over.
    lib, dev = radio.open_radio(round(freq_mhz * 1e6), SAMPLE_RATE, gain)
    print(f"listening to {freq_mhz} MHz, press Ctrl+C to stop")
    print("(new songs show up within about 15 seconds)")
    shown_station = False
    last_song = None
    good = 0
    total = 0
    rounds = 0
    try:
        while True:
            raw = radio.read_raw(lib, dev, 10 * SAMPLE_RATE)
            baseband = fm.demodulate(radio.bytes_to_iq(raw), SAMPLE_RATE)
            try:
                station = rds.decode_station(baseband, SAMPLE_RATE)
            except ValueError as error:
                print(error)
                break
            now = datetime.now().strftime("%H:%M:%S")

            if not shown_station and station["pi"] is not None:
                letters = rds.call_letters(station["pi"]) or f"0x{station['pi']:04X}"
                print(f"{now}  station: {letters} ({rds.program_type(station['pty'])})")
                shown_station = True

            song = rds.newest_song(station["texts"])
            if song is not None and song != last_song:
                print(f"{now}  {song}")
                last_song = song

            good += station["good_blocks"]
            total += station["total_blocks"]
            rounds += 1
            if rounds % 5 == 0:
                print(f"{now}  (still listening, {100 * good / max(total, 1):.0f}% of the data readable)")
    except KeyboardInterrupt:
        print("stopped")
    finally:
        radio.close_radio(lib, dev)


def main():
    parser = argparse.ArgumentParser(description="Decode the RDS data (station name, song) from an FM station with an RTL-SDR dongle.")
    parser.add_argument("freq", nargs="?", type=float, help="station frequency in MHz, like 92.3")
    parser.add_argument("--live", action="store_true", help="keep listening and print each new song")
    parser.add_argument("--wav", action="store_true", help="save the station's audio to a wav file")
    parser.add_argument("--file", help="decode a saved recording (.cu8) instead of using the dongle")
    parser.add_argument("--seconds", type=float, default=30, help="how long to listen (default 30)")
    parser.add_argument("--gain", type=float, default=19.7, help="tuner gain in dB (default 19.7)")
    parser.add_argument("--save", action="store_true", help="also save the raw recording to the recordings folder")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "rb") as f:
            decode_and_show(f.read())
    elif args.freq is None:
        parser.error("give a station frequency like 92.3, or a recording with --file")
    elif args.live:
        listen_live(args.freq, args.gain)
    elif args.wav:
        save_audio(args.freq, args.seconds, args.gain)
    else:
        listen_once(args.freq, args.seconds, args.gain, args.save)


if __name__ == "__main__":
    main()
