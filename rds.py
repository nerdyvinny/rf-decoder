# rds.py
# Decodes RDS, the digital data FM stations send along with the audio
# (call letters, station name, the song that's playing, the time).
#
# How RDS works, the short version:
#  - It sits at 57 kHz in the FM baseband, exactly 3 times the 19 kHz stereo pilot tone.
#  - It sends 1187.5 bits per second, exactly the pilot / 16.
#  - Each bit is sent as a pulse and then the same pulse flipped ("biphase").
#  - The bits are differentially coded: a 1 means "different from the last bit".
#  - Bits come in 26 bit blocks: 16 bits of data and a 10 bit check.
#  - 4 blocks (A, B, C, D) make a group. A is the station code, B says what
#    kind of group it is, C and D carry the actual data.
#
# The pilot is loud and the RDS signal is quiet, so the pilot is used as the
# clock for everything: the RDS carrier is the pilot x3 and the bit clock is
# the pilot / 16. That also cancels out any error in the dongle's own clock.

from datetime import datetime, timedelta

import numpy as np

PILOT = 19000
RDS_CARRIER = 57000
RDS_WIDTH = 2400  # the RDS signal takes up 57 kHz +- 2.4 kHz

# The 10 bit check is the remainder after dividing by this polynomial
# (x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1), same idea as a CRC.
CHECK_POLY = 0b10110111001

# Each block position adds its own "offset" to the check, so a good block also
# tells you which block it is. C' is used instead of C in some groups.
OFFSETS = {"A": 0x0FC, "B": 0x198, "C": 0x168, "C'": 0x350, "D": 0x1B4}

# program types for North America (RBDS). Europe uses a different list.
PROGRAM_TYPES = [
    "None", "News", "Information", "Sports", "Talk", "Rock", "Classic Rock", "Adult Hits",
    "Soft Rock", "Top 40", "Country", "Oldies", "Soft", "Nostalgia", "Jazz", "Classical",
    "Rhythm and Blues", "Soft R&B", "Language", "Religious Music", "Religious Talk",
    "Personality", "Public", "College", "Spanish Talk", "Spanish Music", "Hip Hop",
    "Unassigned", "Unassigned", "Weather", "Emergency Test", "Emergency",
]


# ---------- step 1: FM baseband -> bits ----------

def bandpass(signal, sample_rate, low_hz, high_hz):
    # Keeps only the frequencies between low_hz and high_hz using an FFT.
    # Returns a complex signal (only the positive frequencies are kept).
    # Has to be float64: with float32 the pilot's phase drifted (a 3 second
    # test signal read 6 Hz off), which slowly messes up the bit timing.
    #
    # The FFT is padded with zeros up to a power of 2. A 30 second recording is
    # 7,499,999 samples, and an FFT that length took 3.8 s instead of 0.2 s.
    size = 1
    while size < len(signal):
        size = size * 2
    spectrum = np.fft.fft(np.asarray(signal, dtype=np.float64), size)
    freqs = np.fft.fftfreq(size, 1 / sample_rate)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    return 2 * np.fft.ifft(spectrum)[:len(signal)]


def get_bits(baseband, sample_rate):
    # find the pilot tone
    pilot = bandpass(baseband, sample_rate, PILOT - 15, PILOT + 15)
    nearby = bandpass(baseband, sample_rate, PILOT + 200, PILOT + 230)
    if np.median(np.abs(pilot)) < 4 * np.median(np.abs(nearby)):
        raise ValueError("no 19 kHz pilot tone, this only works on stereo stations")
    pilot = pilot / np.abs(pilot)  # only the phase matters

    # Move the RDS signal down to 0 Hz using the pilot x3.
    # (Cubing the pilot triples its phase, so it turns into a 57 kHz tone locked to the pilot.)
    rds = bandpass(baseband, sample_rate, RDS_CARRIER - RDS_WIDTH, RDS_CARRIER + RDS_WIDTH)
    rds = rds * np.conj(pilot ** 3)

    # The RDS signal sits at some fixed angle compared to the pilot. Squaring it
    # gets rid of the +1/-1 data and leaves 2x that angle, so measure it and
    # rotate it out. Then the data is just the real part.
    angle = np.angle(np.sum(rds ** 2)) / 2
    rds = np.real(rds * np.exp(-1j * angle))

    # Bit timing: the bit clock is the pilot / 16, so count pilot cycles.
    # (Measured over the middle 80% since the FFT filter is messy at the very start and end.)
    cycles = np.unwrap(np.angle(pilot)) / (2 * np.pi)
    a = len(cycles) // 10
    b = len(cycles) * 9 // 10
    cycles_per_sample = (cycles[b] - cycles[a]) / (b - a)
    samples_per_bit = 16 / cycles_per_sample  # about 210.5 at 250k samples per second

    # Each bit is a pulse then its mirror image, so take first half minus second half.
    # We don't know where the first bit starts, so try 32 starting points and keep
    # the one where the bits come out strongest. (running_sum makes the half sums fast)
    running_sum = np.concatenate(([0.0], np.cumsum(rds)))
    best_score = -1
    best_values = None
    for offset in np.linspace(0, samples_per_bit, 32, endpoint=False):
        starts = np.arange(offset, len(rds) - samples_per_bit, samples_per_bit)
        start = starts.astype(int)
        middle = (starts + samples_per_bit / 2).astype(int)
        end = (starts + samples_per_bit).astype(int)
        first_half = running_sum[middle] - running_sum[start]
        second_half = running_sum[end] - running_sum[middle]
        values = first_half - second_half
        score = np.mean(np.abs(values))
        if score > best_score:
            best_score = score
            best_values = values

    # undo the differential coding: a 1 means the sign changed from the last bit
    sent = best_values > 0
    bits = []
    for i in range(1, len(sent)):
        if sent[i] != sent[i - 1]:
            bits.append(1)
        else:
            bits.append(0)
    return bits


# ---------- step 2: bits -> blocks and groups ----------

def calc_check(data):
    # the 10 bit check for 16 bits of data
    reg = data << 10
    for i in range(25, 9, -1):
        if (reg >> i) & 1:
            reg = reg ^ (CHECK_POLY << (i - 10))
    return reg


def block_ok(word, offset):
    data = word >> 10
    check = word & 0x3FF
    return (calc_check(data) ^ offset) == check


def read_block(word, names):
    # Returns the 16 data bits of a block, or None if it's broken.
    # names = which blocks are allowed in this spot, like ["C", "C'"]
    for name in names:
        if block_ok(word, OFFSETS[name]):
            return word >> 10

    # Not good as is, so try to fix it: flip 1 bit, or 2 bits next to each other.
    # 2 next to each other is the most common error: because of the differential
    # coding, one wrong decision on the air flips 2 neighbouring data bits.
    # (Longer bursts could be fixed too, but the more you allow, the more likely
    # a badly broken block gets "fixed" into the wrong data.)
    # Note: this only works because blocks are read at the spot where they belong.
    # A good A block read as a B block can look like a B block with 2 bad bits.
    for name in names:
        for i in range(26):
            if block_ok(word ^ (1 << i), OFFSETS[name]):
                return (word ^ (1 << i)) >> 10
        for i in range(25):
            if block_ok(word ^ (3 << i), OFFSETS[name]):
                return (word ^ (3 << i)) >> 10
    return None


def get_word(bits, i):
    # the 26 bits starting at position i, as one number
    word = 0
    for bit in bits[i:i + 26]:
        word = (word << 1) | bit
    return word


def get_groups(bits):
    # Lines the bits up into groups of 4 blocks.
    # Returns (groups, good_blocks, total_blocks). Each group is [a, b, c, d],
    # each one 16 bits of data, or None if that block was broken.

    # Find where the groups start: move along one bit at a time until there's
    # a good A block with a good B block right after it.
    start = None
    for i in range(len(bits) - 52):
        if block_ok(get_word(bits, i), OFFSETS["A"]) and block_ok(get_word(bits, i + 26), OFFSETS["B"]):
            start = i
            break
    if start is None:
        return [], 0, 0

    groups = []
    good = 0
    total = 0
    i = start
    while i + 104 <= len(bits):
        group = []
        for j, names in enumerate([["A"], ["B"], ["C", "C'"], ["D"]]):
            data = read_block(get_word(bits, i + 26 * j), names)
            total += 1
            if data is not None:
                good += 1
            group.append(data)
        groups.append(group)
        i += 104
    return groups, good, total


# ---------- step 3: groups -> station info ----------

def to_char(code):
    if 32 <= code < 127:
        return chr(code)
    return "?"


def call_letters(pi):
    # US stations: the station code is the call letters counted in base 26.
    # K stations start at 0x1000 (KAAA), W stations start at 0x54A8 (WAAA).
    # Codes starting with A are a short form that gets expanded first.
    if pi >> 12 == 0xA:
        if (pi >> 8) & 0xF == 0xF:
            pi = (pi & 0xFF) << 8  # AFxy -> xy00
        else:
            pi = (((pi >> 8) & 0xF) << 12) | (pi & 0xFF)  # Axyz -> x0yz
    if 0x1000 <= pi <= 0x54A7:
        first = "K"
        n = pi - 0x1000
    elif 0x54A8 <= pi <= 0x994F:
        first = "W"
        n = pi - 0x54A8
    else:
        return None  # 3 letter call signs, Canada and Mexico work differently
    return first + chr(65 + n // 676) + chr(65 + (n // 26) % 26) + chr(65 + n % 26)


def program_type(pty):
    if pty is None:
        return "unknown"
    return PROGRAM_TYPES[pty]


def decode_clock(b, c, d):
    # The date is sent as days since Nov 17 1858 (the "modified julian day").
    # Returns (local time, hours from UTC), or None if it doesn't make sense.
    mjd = ((b & 3) << 15) | (c >> 1)
    hour = ((c & 1) << 4) | (d >> 12)
    minute = (d >> 6) & 63
    if hour > 23 or minute > 59 or mjd < 50000:
        return None
    offset = (d & 31) / 2  # local time offset, sent in half hours
    if (d >> 5) & 1:
        offset = -offset
    utc = datetime(1858, 11, 17) + timedelta(days=mjd, hours=hour, minutes=minute)
    return utc + timedelta(hours=offset), offset


def most_common(counts):
    if not counts:
        return None
    return max(counts, key=counts.get)


def decode_groups(groups):
    station = {"pi": None, "pty": None, "names": [], "texts": [], "clock": None}
    pi_counts = {}
    pty_counts = {}
    name = ["?"] * 8
    name_parts = set()
    text = [None] * 64
    text_flag = None

    for a, b, c, d in groups:
        if a is not None:
            pi_counts[a] = pi_counts.get(a, 0) + 1
        if b is None:
            continue  # without block B we don't know what kind of group this is
        group_type = b >> 12
        version_b = (b >> 11) & 1
        pty = (b >> 5) & 31
        pty_counts[pty] = pty_counts.get(pty, 0) + 1

        if group_type == 0 and d is not None:
            # station name, 8 letters, 2 per group
            part = b & 3
            name[part * 2] = to_char(d >> 8)
            name[part * 2 + 1] = to_char(d & 255)
            name_parts.add(part)
            if len(name_parts) == 4:
                # all 4 parts are fresh, so this is one whole name
                station["names"].append("".join(name))
                name_parts = set()

        elif group_type == 2:
            # radio text, up to 64 letters, 4 per group (2 per group in version B)
            flag = (b >> 4) & 1
            if flag != text_flag:
                # the station flips this flag when the text changes, so start over
                text = [None] * 64
                text_flag = flag
            part = b & 15
            if version_b == 0 and c is not None and d is not None:
                text[part * 4:part * 4 + 4] = [c >> 8, c & 255, d >> 8, d & 255]
            elif version_b == 1 and d is not None:
                text[part * 2:part * 2 + 2] = [d >> 8, d & 255]
            # the text ends at a carriage return (13), or at the max length
            end = 64 if version_b == 0 else 32
            if 13 in text:
                end = text.index(13)
            if None not in text[:end]:
                full_text = "".join(to_char(x) for x in text[:end]).strip()
                if full_text:
                    station["texts"].append(full_text)
                text = [None] * 64

        elif group_type == 4 and version_b == 0 and c is not None and d is not None:
            clock = decode_clock(b, c, d)
            if clock is not None:
                station["clock"] = clock

    station["pi"] = most_common(pi_counts)
    station["pty"] = most_common(pty_counts)
    return station


def decode_station(baseband, sample_rate):
    # everything in one go: FM baseband -> station info
    bits = get_bits(baseband, sample_rate)
    groups, good, total = get_groups(bits)
    station = decode_groups(groups)
    station["good_blocks"] = good
    station["total_blocks"] = total
    return station


def newest_song(texts):
    # The newest radio text that came through at least twice. A text that only
    # shows up once can be half the old text and half the new one (when the
    # station changes it in the middle), so those are skipped.
    for text in reversed(texts):
        if texts.count(text) >= 2:
            return text
    return None
