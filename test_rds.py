# tests for rds.py
# These build a fake FM station (audio tone + 19 kHz pilot + RDS data at 57 kHz)
# with data we choose, then check the decoder gets exactly that data back.

from datetime import datetime

import numpy as np
import pytest

import fm
import rds

RATE = 250000


def make_station(bits, rds_level=0.04, noise=0.02, angle=0.0, pilot=True):
    # differential coding: send a change for a 1, no change for a 0
    sent = [0]
    for bit in bits:
        sent.append(sent[-1] ^ bit)
    sent = np.array(sent)

    t = np.arange(int((len(sent) + 2) / 1187.5 * RATE)) / RATE
    pilot_phase = 2 * np.pi * 19000 * t + 0.7
    bit_number = np.floor(t * 1187.5).astype(int)
    level = np.where(sent[np.minimum(bit_number, len(sent) - 1)] == 1, 1.0, -1.0)
    level[bit_number >= len(sent)] = 0
    first_half = (t * 1187.5) % 1 < 0.5
    biphase = np.where(first_half, level, -level)  # each bit: pulse, then the flipped pulse

    baseband = 0.3 * np.sin(2 * np.pi * 1000 * t) + rds_level * biphase * np.cos(3 * pilot_phase + angle)
    if pilot:
        baseband = baseband + 0.09 * np.cos(pilot_phase)

    # FM modulate it, then add noise
    phase = 2 * np.pi * fm.MAX_DEVIATION * np.cumsum(baseband) / RATE
    rng = np.random.default_rng(0)
    return np.exp(1j * phase) + noise * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))


def as_string(bits):
    return "".join(str(int(b)) for b in bits)


# ---------- helpers that build real RDS groups ----------

def make_block(data, name):
    return (data << 10) | (rds.calc_check(data) ^ rds.OFFSETS[name])


def make_group(a, b, c, d):
    bits = []
    for word in [make_block(a, "A"), make_block(b, "B"), make_block(c, "C"), make_block(d, "D")]:
        for i in range(25, -1, -1):
            bits.append((word >> i) & 1)
    return bits


def name_groups(pi, pty, name):
    # group 0A: the station name, 2 letters per group, sent 3 times
    bits = []
    for _ in range(3):
        for part in range(4):
            letters = (ord(name[part * 2]) << 8) | ord(name[part * 2 + 1])
            bits += make_group(pi, (pty << 5) | part, 0xE0CD, letters)
    return bits


def text_groups(pi, pty, text, flag=0):
    # group 2A: radio text, 4 letters per group, ending with a carriage return
    text = text + "\r"
    while len(text) % 4 != 0:
        text = text + " "
    bits = []
    for part in range(len(text) // 4):
        c = [ord(x) for x in text[part * 4:part * 4 + 4]]
        b = (2 << 12) | (pty << 5) | (flag << 4) | part
        bits += make_group(pi, b, (c[0] << 8) | c[1], (c[2] << 8) | c[3])
    return bits


def clock_group(pi, pty, mjd, hour, minute, offset_half_hours):
    # group 4A: the clock (this one sends a negative offset, like US time zones)
    b = (4 << 12) | (pty << 5) | (mjd >> 15)
    c = ((mjd & 0x7FFF) << 1) | (hour >> 4)
    d = ((hour & 15) << 12) | (minute << 6) | (1 << 5) | offset_half_hours
    return make_group(pi, b, c, d)


PI = 0x54A8 + 676 * 4 + 26 * 8 + 13  # station code for "WEIN"


# ---------- the tests ----------

def test_bits_come_back_exactly():
    data = [int(x) for x in np.random.default_rng(42).integers(0, 2, 600)]
    # in phase with the pilot, 90 degrees off, and upside down
    for angle in [0.0, np.pi / 2, np.pi]:
        bits = rds.get_bits(fm.demodulate(make_station(data, angle=angle), RATE), RATE)
        assert as_string(data[2:-2]) in as_string(bits)


def test_mono_station_gives_a_clear_error():
    data = [int(x) for x in np.random.default_rng(42).integers(0, 2, 600)]
    with pytest.raises(ValueError):
        rds.get_bits(fm.demodulate(make_station(data, pilot=False), RATE), RATE)


def test_call_letters():
    assert rds.call_letters(PI) == "WEIN"
    assert rds.call_letters(0x1000) == "KAAA"
    assert rds.call_letters(0x54A7) == "KZZZ"
    assert rds.call_letters(0x54A8) == "WAAA"
    assert rds.call_letters(0x994F) == "WZZZ"
    assert rds.call_letters(0x8FC4) == "WWKA"  # what 92.3 Orlando actually sends


def test_whole_station_through_the_radio_chain():
    # MJD 61316 is 2026-10-03. 18:30 UTC with a -4 hour offset is 14:30 local.
    bits = name_groups(PI, 1, "NEWS 923")
    bits += text_groups(PI, 1, "HELLO FROM THE TEST SUITE")
    bits += clock_group(PI, 1, 61316, 18, 30, 8)
    station = rds.decode_station(fm.demodulate(make_station(bits), RATE), RATE)
    assert rds.call_letters(station["pi"]) == "WEIN"
    assert rds.program_type(station["pty"]) == "News"
    assert "NEWS 923" in station["names"]
    assert "HELLO FROM THE TEST SUITE" in station["texts"]
    assert station["clock"] == (datetime(2026, 10, 3, 14, 30), -4.0)
    assert station["good_blocks"] == station["total_blocks"]


def test_bad_bits_get_fixed():
    bits = name_groups(PI, 1, "NEWS 923") + text_groups(PI, 1, "HELLO FROM THE TEST SUITE")
    bits[5 * 104 + 30] ^= 1  # one wrong bit in block B of the 6th group
    bits[7 * 104 + 60] ^= 1  # two wrong bits next to each other in block C of the 8th group
    bits[7 * 104 + 61] ^= 1
    groups, good, total = rds.get_groups(bits)
    assert len(groups) == len(bits) // 104
    assert good == total
    assert "NEWS 923" in rds.decode_groups(groups)["names"]


def test_badly_broken_block_is_dropped_not_misread():
    bits = name_groups(PI, 1, "NEWS 923") + text_groups(PI, 1, "HELLO FROM THE TEST SUITE")
    for i in range(130, 140):  # 10 wrong bits in a row, too many to fix
        bits[i] ^= 1
    groups, good, total = rds.get_groups(bits)
    assert good < total
    for name in rds.decode_groups(groups)["names"]:
        assert name == "NEWS 923"


def test_c_and_c_prime_never_get_mixed_up():
    # The 3rd spot in a group can hold a C or a C' block, so both get tried there.
    # A good block of one kind must never get "fixed" into the other kind.
    for real, other in [("C", "C'"), ("C'", "C")]:
        word = make_block(0x1234, real)
        assert rds.read_block(word, ["C", "C'"]) == 0x1234
        assert rds.read_block(word, [other]) is None


def test_newest_song():
    assert rds.newest_song([]) is None
    assert rds.newest_song(["SONG 1"]) is None  # only seen once, could be half old half new
    assert rds.newest_song(["SONG 1", "SONG 1", "SONG 2"]) == "SONG 1"
    assert rds.newest_song(["SONG 1", "SONG 1", "SONG 2", "SONG 2"]) == "SONG 2"


def test_song_change_between_two_listens():
    # like live mode: the first listen has one song, the next one a new song
    first = name_groups(PI, 10, "COUNTRY ") + text_groups(PI, 10, "FIRST SONG - ARTIST ONE") * 3
    second = name_groups(PI, 10, "COUNTRY ") + text_groups(PI, 10, "SECOND SONG - ARTIST TWO", flag=1) * 3
    songs = []
    for bits in [first, second]:
        station = rds.decode_station(fm.demodulate(make_station(bits), RATE), RATE)
        songs.append(rds.newest_song(station["texts"]))
    assert songs == ["FIRST SONG - ARTIST ONE", "SECOND SONG - ARTIST TWO"]
