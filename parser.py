"""
parser.py — Ardupilot DataFlash binary log parser (.BIN format).

Binary format structure:
  Each message starts with two sync bytes: 0xA3 0x95
  Followed by: msg_type (1B), then payload of variable length.
  The special FMT (type=0x80) message defines all other message formats.

FMT payload layout (89 bytes total including sync+type):
  [0x80] type(B) length(B) name(4s) format(16s) columns(64s)

Format characters (Ardupilot DataFlash encoding):
  b/B = int8/uint8,  h/H = int16/uint16,  i/I = int32/uint32
  f   = float32,     d   = float64,        q/Q = int64/uint64
  n   = char[4],     N   = char[16],        Z  = char[64]
  c   = int16 * 0.01 (e.g. HDop)
  C   = uint16 * 0.01
  e   = int32 * 0.01 (e.g. altitude in cm -> metres)
  E   = uint32 * 0.01
  L   = int32 * 1e-7 (latitude/longitude in WGS-84 degrees)
  M   = uint8 (flight mode)
"""

import struct
import pandas as pd
from typing import Dict, List

HEAD1 = 0xA3
HEAD2 = 0x95
MSG_FMT_TYPE = 0x80

_FMT_MAP: Dict[str, tuple] = {
    'b': ('b', 1), 'B': ('B', 1),
    'h': ('h', 2), 'H': ('H', 2),
    'i': ('i', 4), 'I': ('I', 4),
    'f': ('f', 4), 'd': ('d', 8),
    'q': ('q', 8), 'Q': ('Q', 8),
    'n': ('4s', 4), 'N': ('16s', 16),
    'Z': ('64s', 64), 'z': ('64s', 64),
    'c': ('h', 2),
    'C': ('H', 2),
    'e': ('i', 4),
    'E': ('I', 4),
    'L': ('i', 4),
    'M': ('B', 1),
}

_SCALE: Dict[str, float] = {
    'c': 0.01, 'C': 0.01,
    'e': 0.01, 'E': 0.01,
    'L': 1e-7,
}


def _decode_value(fmt_char: str, raw_val):
    if isinstance(raw_val, bytes):
        return raw_val.rstrip(b'\x00').decode('ascii', errors='ignore').strip()
    if fmt_char in _SCALE:
        return raw_val * _SCALE[fmt_char]
    return raw_val


def parse_bin(filepath: str, target_messages: List[str]) -> Dict[str, List[dict]]:
    """
    Parse an Ardupilot .BIN file and extract specified message types.

    Returns dict mapping message name -> list of row dicts.
    TimeUS is always in microseconds (raw from log).
    """
    fmt_map: Dict[int, tuple] = {}
    results: Dict[str, List[dict]] = {name: [] for name in target_messages}

    with open(filepath, 'rb') as f:
        data = f.read()

    i = 0
    total = len(data)

    while i < total - 3:
        if data[i] != HEAD1 or data[i + 1] != HEAD2:
            i += 1
            continue

        msg_type = data[i + 2]

        if msg_type == MSG_FMT_TYPE:
            if i + 89 > total:
                break
            fmt_type = data[i + 3]
            fmt_len  = data[i + 4]
            name     = data[i + 5:i + 9].rstrip(b'\x00').decode('ascii', errors='ignore').strip()
            fmt_str  = data[i + 9:i + 25].rstrip(b'\x00').decode('ascii', errors='ignore').strip()
            cols_raw = data[i + 25:i + 89].rstrip(b'\x00').decode('ascii', errors='ignore').strip()
            cols     = [c.strip() for c in cols_raw.split(',')]
            fmt_map[fmt_type] = (name, fmt_str, cols, fmt_len)
            i += 89
            continue

        if msg_type in fmt_map:
            name, fmt_str, cols, msg_len = fmt_map[msg_type]

            if name in target_messages:
                payload = data[i + 3: i + msg_len]
                row: dict = {}
                offset = 0

                for fi, c in enumerate(fmt_str):
                    if c not in _FMT_MAP:
                        continue
                    struct_fmt, size = _FMT_MAP[c]
                    if offset + size > len(payload):
                        break
                    raw = struct.unpack_from('<' + struct_fmt, payload, offset)[0]
                    if fi < len(cols):
                        row[cols[fi]] = _decode_value(c, raw)
                    offset += size

                results[name].append(row)

            i += msg_len
            continue

        i += 1

    return results


def to_dataframes(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Parse a .BIN file and return GPS, IMU, and ATT as clean DataFrames.

    GPS: Lat (deg), Lng (deg), Alt (m MSL), Spd (m/s), VZ (m/s)
    IMU: AccX/Y/Z (m/s^2), GyrX/Y/Z (rad/s)
    ATT: Roll, Pitch, Yaw (degrees)

    All DataFrames get a 'time_s' column = seconds from first GPS record.
    """
    raw = parse_bin(filepath, ['GPS', 'IMU', 'ATT'])

    dfs = {}

    gps_rows = [r for r in raw['GPS']
                if abs(r.get('Lat', 0)) > 1 and r.get('Status', 0) >= 3]
    if gps_rows:
        df = pd.DataFrame(gps_rows)
        t0 = df['TimeUS'].iloc[0]
        df['time_s'] = (df['TimeUS'] - t0) / 1e6
        dfs['GPS'] = df.reset_index(drop=True)

    imu_rows = [r for r in raw['IMU'] if r.get('I', 0) == 0]
    if imu_rows:
        df = pd.DataFrame(imu_rows)
        t0 = dfs['GPS']['TimeUS'].iloc[0] if 'GPS' in dfs else df['TimeUS'].iloc[0]
        df['time_s'] = (df['TimeUS'] - t0) / 1e6
        dfs['IMU'] = df.reset_index(drop=True)

    if raw['ATT']:
        df = pd.DataFrame(raw['ATT'])
        t0 = dfs['GPS']['TimeUS'].iloc[0] if 'GPS' in dfs else df['TimeUS'].iloc[0]
        df['time_s'] = (df['TimeUS'] - t0) / 1e6
        dfs['ATT'] = df.reset_index(drop=True)

    return dfs


def get_sampling_info(dfs: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    """Return sampling frequency and units for each sensor stream."""
    units = {
        'GPS': {'Lat': 'deg', 'Lng': 'deg', 'Alt': 'm (MSL)', 'Spd': 'm/s', 'VZ': 'm/s'},
        'IMU': {'AccX': 'm/s^2', 'AccY': 'm/s^2', 'AccZ': 'm/s^2',
                'GyrX': 'rad/s', 'GyrY': 'rad/s', 'GyrZ': 'rad/s'},
        'ATT': {'Roll': 'deg', 'Pitch': 'deg', 'Yaw': 'deg'},
    }
    info = {}
    for name, df in dfs.items():
        if len(df) < 2:
            continue
        duration = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
        freq = len(df) / duration if duration > 0 else 0
        info[name] = {
            'records':    len(df),
            'freq_hz':    round(freq, 1),
            'duration_s': round(duration, 2),
            'units':      units.get(name, {}),
        }
    return info
