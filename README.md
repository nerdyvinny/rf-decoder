# FM Radio Data Decoder

Decodes RDS, the hidden data that FM radio stations send along with the music:
the station's call letters, its name, the song that's playing, and the time.

It reads raw samples from a cheap RTL-SDR USB dongle and does everything else
in Python with numpy (no radio libraries): FM demodulation, pulling out the
data signal, bit timing, error checking, and decoding the text.

## Example

30 seconds of 92.3 FM in Orlando:

```
> python main.py --file recordings/fm-92.3.cu8
call letters:  WWKA  (station code 0x8FC4)
program type:  Country
station name:  COUNTRY | MORGTRY | MORGAN | WALLEN | BEEN BY | NOW | ORLANDO' | S  1 FOR | NEW
radio text:    Morgan Wallen - Been By Now  (11x)
radio text:    Anidjar & Levine - 800-747-FREE Accident Attorneys  (1x)
station clock: 2026-10-03 04:26 (UTC-4)
data quality:  1,368 of 1,368 blocks readable (100%)
```

The station name scrolls like a banner, 8 letters at a time. "MORGTRY" is what
you get when the name changes from "MORGAN" to "COUNTRY" halfway through.

## What you need

- An RTL-SDR dongle and antenna (tested with a Nooelec NESDR SMArt v5)
- Windows, with the dongle's driver switched to WinUSB using Zadig
- `rtlsdr.dll` from the rtl-sdr Windows release, unzipped to `~/tools/rtl-sdr/x64`
  (or set `RTLSDR_DIR` to wherever it is)
- Python 3 and numpy: `pip install -r requirements.txt`

## How to run

```
python main.py 92.3            listen for 30 seconds and show what the station is sending
python main.py 92.3 --live     keep listening and print each new song (Ctrl+C to stop)
python main.py 92.3 --wav      save 30 seconds of the station's audio to a wav file
python main.py --file recordings/fm-92.3.cu8     decode a saved recording
```

Close SDR# (or anything else using the dongle) first. `--save` keeps the raw
recording, `--seconds` changes how long it listens, and `--gain` changes the
gain (default 19.7 dB, strong stations can clip above that).

## How it works

1. `radio.py` reads raw samples from the dongle, 250,000 per second.
2. `fm.py` demodulates the FM. The phase change between two samples is the
   frequency, and in FM the frequency is the signal. What comes out is the
   station's "baseband": the audio, a 19 kHz stereo pilot tone, and the RDS
   data at 57 kHz.
3. `rds.py` pulls out the RDS data. Its carrier is exactly 3 times the pilot
   and its bit rate is exactly the pilot / 16, so the loud pilot is used as the
   clock for the quiet data.
4. Each bit is a pulse and then the same pulse flipped, so the decoder compares
   the two halves. Then it undoes the differential coding (a 1 means "changed
   from the last bit").
5. The bits come in 26 bit blocks: 16 bits of data and a 10 bit check (like a
   CRC). Blocks with 1 or 2 bad bits next to each other get fixed, worse ones
   get thrown away. 4 blocks make a group, and the group type says what's
   inside: 2 letters of the station name, 4 letters of radio text, or the
   clock. The station code turns into US call letters.

## Things that went wrong along the way

- **Setting the gain did nothing.** This driver resets the tuner the first time
  it tunes, so the gain has to be set after tuning, not before.
- **The audio had a fake 3 kHz tone in it.** Removing the DC offset (normally a
  good idea) cuts into the station when it's tuned right at the center,
  because then the station itself is at 0 Hz.
- **The pilot frequency drifted** by a few Hz over a few seconds because the FFT
  was running in float32. Switching to float64 fixed it.
- **Decoding 30 seconds took 24 seconds.** An FFT of 7,499,999 samples is really
  slow (3.8 s each). Padding it to a power of 2 made it 0.2 s.
- **Half old, half new text.** When a station changes its text you can catch a
  mix of both, so live mode only prints a text once it's come through twice.

## Tests

```
python -m pytest
```

The tests build a fake FM station with data we choose (an audio tone, the
pilot, and RDS groups with a station name, text, and clock), run it through the
same code, and check that exactly that data comes back. They also cover the
signal arriving upside down, damaged bits, a mono station with no pilot, and a
song change.

(READ ME GENERATED WITH AI)
