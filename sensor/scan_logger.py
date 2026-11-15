# ============================================================
# ECO-LOOP — Real sensor scan logger
# sensor/scan_logger.py
#
# Talks to the ESP32 (firmware/ecoloop_scanner) over USB and
# saves every reading of the AS7265x sensor into one CSV file.
#
# Try it without any hardware first:
#     python sensor/scan_logger.py --simulate
#
# With the ESP32 plugged in (find the COM port in Device Manager):
#     python sensor/scan_logger.py --port COM5 --operator Yugant
#
# Commands you type while it runs (also shown with "help"):
#     w            white reference (reflectance box, white tile in place)
#     l            lamp reference  (transmission box, reference block in place)
#     r A012 3     reflectance scan of apple A012, spot 3
#     t A012 1     transmission scan of apple A012, position 1
#     live         keep printing readings (to check the setup)
#     note text    save a note into the session log
#     q            quit
#
# Full instructions: docs/04_HARDWARE_SETUP.md and
#                    docs/05_DATA_COLLECTION_PROTOCOL.md
# ============================================================

import argparse
import csv
import os
import random
import statistics
import sys
import time
from datetime import datetime

WAVELENGTHS = [410, 435, 460, 485, 510, 535, 560, 585, 610,
               645, 680, 705, 730, 760, 810, 860, 900, 940]

GAIN_NAMES = {0: "1x", 1: "3.7x", 2: "16x", 3: "64x"}
RAW_MAX = 65535            # a raw value this high means the sensor is saturated

# Which lights are used for each kind of scan.
# Each entry: (name saved in the CSV, white, ir, uv, external lamp)
REFLECTANCE_FRAMES = [
    ("OFF",   0, 0, 0, 0),   # all lights off = how dark the box really is
    ("WHITE", 1, 0, 0, 0),
    ("IR",    0, 1, 0, 0),
    ("UV",    0, 0, 1, 0),
]
TRANSMISSION_FRAMES = [
    ("LAMP_OFF", 0, 0, 0, 0),
    ("LAMP_ON",  0, 0, 0, 1),
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "real_scans")

SCAN_COLUMNS = (
    ["timestamp", "session_id", "operator", "apple_id", "scan_type", "spot",
     "illumination", "gain", "cycles", "frames", "temp_c", "saturated",
     "worst_repeat_pct"]
    + [f"raw_{w}" for w in WAVELENGTHS]
    + [f"cal_{w}" for w in WAVELENGTHS]
    + ["note"]
)

APPLE_COLUMNS = ["apple_id", "variety", "bought_date", "source", "treatment",
                 "treatment_date", "diameter_mm", "weight_g", "notes"]

CUT_COLUMNS = ["apple_id", "cut_date", "rot_percent_equator", "rot_percent_axis",
               "rot_location", "class_label", "photo_files", "ph_measured", "notes"]


# ══════════════════════════════════════════════════════════════════════════
# SENSOR LINKS — the real one (USB serial) and a fake one for practice
# ══════════════════════════════════════════════════════════════════════════

class SerialLink:
    """Sends one text command to the ESP32 and returns its one-line reply."""

    def __init__(self, port, baud=115200):
        try:
            import serial
        except ImportError:
            sys.exit("pyserial is missing. Install it with:  pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=6)
        time.sleep(2.5)                 # the ESP32 restarts when the port opens
        self.ser.reset_input_buffer()

    def ask(self, command):
        self.ser.write((command + "\n").encode())
        reply = self.ser.readline().decode(errors="ignore").strip()
        if not reply:
            raise RuntimeError(f"No reply to '{command}'. Is the ESP32 running the scanner firmware?")
        if reply.startswith("ERR"):
            raise RuntimeError(f"ESP32 said: {reply}")
        return reply

    def close(self):
        self.ser.close()


class FakeLink:
    """Pretends to be the ESP32 so the workflow can be practised without hardware.
    The numbers are invented and must never be used as data."""

    def __init__(self):
        self.led = (0, 0, 0)
        self.lamp = 0
        self.gain = 2

    def ask(self, command):
        parts = command.upper().split(",")
        if parts[0] == "PING":
            return "OK,ECOLOOP_SCANNER,1"
        if parts[0] == "INFO":
            return "OK,INFO,0,0.0.0"
        if parts[0] == "CFG":
            self.gain = int(parts[1])
            return f"OK,CFG,{parts[1]},{parts[2]}"
        if parts[0] == "LED":
            self.led = tuple(int(p) for p in parts[1:4])
            return "OK," + ",".join(parts)
        if parts[0] == "LAMP":
            self.lamp = int(parts[1])
            return f"OK,LAMP,{parts[1]}"
        if parts[0] == "READ":
            time.sleep(0.3)             # a real reading takes about this long
            level = 40 + 2500 * self.led[0] + 900 * self.led[1] + 300 * self.led[2] + 600 * self.lamp
            raw = [max(0, int(random.gauss(level, level * 0.004 + 1))) for _ in WAVELENGTHS]
            cal = [round(r * 0.35, 3) for r in raw]
            return "DATA,27.0," + ",".join(map(str, raw)) + "," + ",".join(map(str, cal))
        return "ERR,unknown command"

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════════════
# MEASURING
# ══════════════════════════════════════════════════════════════════════════

def read_frames(link, frames, discard=1):
    """Takes several readings in a row and averages them.
    The first reading after changing lights is thrown away because it can
    still contain the old light level."""
    for _ in range(discard):
        link.ask("READ")

    temps, raws, cals = [], [], []
    for _ in range(frames):
        parts = link.ask("READ").split(",")
        if parts[0] != "DATA" or len(parts) != 38:
            raise RuntimeError(f"Unexpected reply from ESP32: {','.join(parts)[:80]}")
        temps.append(float(parts[1]))
        raws.append([float(v) for v in parts[2:20]])
        cals.append([float(v) for v in parts[20:38]])

    raw_mean = [statistics.mean(ch) for ch in zip(*raws)]
    cal_mean = [statistics.mean(ch) for ch in zip(*cals)]

    # How much the repeated readings disagree, as % of the signal (worst channel).
    worst_repeat = 0.0
    if frames > 1:
        for ch in zip(*raws):
            mean = statistics.mean(ch)
            if mean > 50:
                worst_repeat = max(worst_repeat, 100 * statistics.stdev(ch) / mean)

    return {
        "temp_c": round(statistics.mean(temps), 1),
        "raw": [round(v, 1) for v in raw_mean],
        "cal": [round(v, 3) for v in cal_mean],
        "saturated": int(any(max(ch) >= RAW_MAX for ch in zip(*raws))),
        "worst_repeat_pct": round(worst_repeat, 2),
    }


class Scanner:
    def __init__(self, link, args, out_dir):
        self.link = link
        self.args = args
        self.out_dir = out_dir
        self.session_id = datetime.now().strftime("S%Y%m%d_%H%M%S")
        self.scans_path = os.path.join(out_dir, "scans.csv")
        self.log_path = os.path.join(out_dir, "session_log.txt")
        self.rows_saved = 0
        self.have_white = False
        self.have_lampref = False

    # ── files ─────────────────────────────────────────────────────────────
    def prepare_files(self):
        os.makedirs(self.out_dir, exist_ok=True)
        for name, columns in (("scans.csv", SCAN_COLUMNS),
                              ("apples.csv", APPLE_COLUMNS),
                              ("cuts.csv", CUT_COLUMNS)):
            path = os.path.join(self.out_dir, name)
            if not os.path.exists(path):
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(columns)
        self.log(f"session started | operator={self.args.operator} | "
                 f"reflectance gain={GAIN_NAMES[self.args.ref_gain]} cycles={self.args.ref_cycles} | "
                 f"transmission gain={GAIN_NAMES[self.args.trans_gain]} cycles={self.args.trans_cycles}")

    def log(self, text):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} | {self.session_id} | {text}\n")

    def save_row(self, apple_id, scan_type, spot, illumination, gain, cycles, result):
        row = ([datetime.now().isoformat(timespec="seconds"), self.session_id,
                self.args.operator, apple_id, scan_type, spot, illumination,
                GAIN_NAMES[gain], cycles, self.args.frames, result["temp_c"],
                result["saturated"], result["worst_repeat_pct"]]
               + result["raw"] + result["cal"] + [""])
        with open(self.scans_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        self.rows_saved += 1

    # ── lights ────────────────────────────────────────────────────────────
    def set_lamp(self, on):
        if self.args.lamp == "auto":
            self.link.ask(f"LAMP,{1 if on else 0}")
            if on:
                time.sleep(self.args.lamp_warmup)
        else:
            input(f"      >> Switch the lamp {'ON' if on else 'OFF'}, then press Enter ")
            if on:
                time.sleep(self.args.lamp_warmup)

    # ── one complete scan (several light settings) ────────────────────────
    def run_scan(self, apple_id, scan_type, spot):
        transmission = scan_type in ("TRANS", "LAMP_REF")
        frames = TRANSMISSION_FRAMES if transmission else REFLECTANCE_FRAMES
        gain = self.args.trans_gain if transmission else self.args.ref_gain
        cycles = self.args.trans_cycles if transmission else self.args.ref_cycles

        self.link.ask(f"CFG,{gain},{cycles}")
        results = {}
        lamp_is_on = False
        try:
            for name, white, ir, uv, lamp in frames:
                if transmission and bool(lamp) != lamp_is_on:
                    self.set_lamp(bool(lamp))
                    lamp_is_on = bool(lamp)
                self.link.ask(f"LED,{white},{ir},{uv}")
                time.sleep(0.15)
                result = read_frames(self.link, self.args.frames)
                results[name] = result
                self.save_row(apple_id, scan_type, spot, name, gain, cycles, result)
        finally:
            self.link.ask("LED,0,0,0")
            if lamp_is_on:
                self.set_lamp(False)

        self.report(apple_id, scan_type, spot, results, transmission)

    def report(self, apple_id, scan_type, spot, results, transmission):
        """Prints a short quality check so problems are caught at the bench."""
        dark_name = "LAMP_OFF" if transmission else "OFF"
        dark = results[dark_name]["raw"]
        print(f"\n   {scan_type} | {apple_id} | spot {spot}   (raw counts, sensor {results[dark_name]['temp_c']} C)")
        print("   " + "light".ljust(9) + "".join(f"{w:>7}" for w in WAVELENGTHS))
        for name, result in results.items():
            print("   " + name.ljust(9) + "".join(f"{v:>7.0f}" for v in result["raw"]))

        warnings = []
        if max(dark) > 0.02 * RAW_MAX:
            warnings.append("lights-off reading is high: light is leaking into the box")
        for name, result in results.items():
            if result["saturated"]:
                warnings.append(f"{name}: SATURATED (65535). Lower the gain or cycles and rescan")
            if result["worst_repeat_pct"] > 3:
                warnings.append(f"{name}: repeated readings differ by {result['worst_repeat_pct']}% "
                                "(apple moved, or lamp not stable)")
        if transmission:
            lit = results["LAMP_ON"]["raw"]
            nir = [WAVELENGTHS.index(w) for w in (705, 730, 760, 810, 860)]
            signal = statistics.mean(lit[i] for i in nir)
            floor = statistics.mean(dark[i] for i in nir)
            ratio = signal / max(floor, 1.0)
            print(f"   light through the sample (705-860 nm): {signal:.0f} counts, "
                  f"{ratio:.1f}x the lamp-off level")
            if ratio < 10:
                warnings.append("very little light is getting through (below 10x the dark level)")
        for warning in warnings:
            print(f"   !! {warning}")
        if not warnings:
            print("   ok")

    # ── live view for setting up ──────────────────────────────────────────
    def live(self, white, ir, uv, seconds=20):
        self.link.ask(f"CFG,{self.args.ref_gain},{self.args.ref_cycles}")
        self.link.ask(f"LED,{white},{ir},{uv}")
        print("   " + "".join(f"{w:>7}" for w in WAVELENGTHS))
        end = time.time() + seconds
        try:
            while time.time() < end:
                result = read_frames(self.link, 1, discard=0)
                print("   " + "".join(f"{v:>7.0f}" for v in result["raw"]))
        except KeyboardInterrupt:
            pass
        finally:
            self.link.ask("LED,0,0,0")


# ══════════════════════════════════════════════════════════════════════════
# COMMAND LOOP
# ══════════════════════════════════════════════════════════════════════════

HELP = """
   w              white reference (reflectance box, white tile in place)
   l              lamp reference  (transmission box, reference block in place)
   r <apple> <n>  reflectance scan,  e.g.  r A012 3
   t <apple> <n>  transmission scan, e.g.  t A012 1
   live [w|ir|uv|off]   print readings for 20 s (default: white LED)
   note <text>    save a note in session_log.txt
   help           show this list
   q              quit
"""


def command_loop(scanner):
    print(HELP)
    while True:
        try:
            line = input("ecoloop> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        words = line.split()
        cmd = words[0].lower()

        try:
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "w":
                scanner.run_scan("WHITE_REF", "WHITE_REF", 0)
                scanner.have_white = True
            elif cmd == "l":
                scanner.run_scan("LAMP_REF", "LAMP_REF", 0)
                scanner.have_lampref = True
            elif cmd in ("r", "t"):
                if len(words) != 3 or not words[2].isdigit():
                    print("   use:  r <apple_id> <spot number>   for example  r A012 3")
                    continue
                if cmd == "r" and not scanner.have_white:
                    print("   !! no white reference yet in this session. Do 'w' first.")
                    continue
                if cmd == "t" and not scanner.have_lampref:
                    print("   !! no lamp reference yet in this session. Do 'l' first.")
                    continue
                scanner.run_scan(words[1].upper(), "REF" if cmd == "r" else "TRANS", int(words[2]))
            elif cmd == "live":
                which = words[1].lower() if len(words) > 1 else "w"
                lights = {"w": (1, 0, 0), "ir": (0, 1, 0), "uv": (0, 0, 1), "off": (0, 0, 0)}
                if which not in lights:
                    print("   use:  live w | live ir | live uv | live off")
                    continue
                scanner.live(*lights[which])
            elif cmd == "note":
                scanner.log("NOTE: " + line[4:].strip())
                print("   saved")
            else:
                print("   unknown command, type 'help'")
        except RuntimeError as error:
            print(f"   !! {error}")

    scanner.log(f"session ended | {scanner.rows_saved} rows saved")
    print(f"\n{scanner.rows_saved} rows saved to {scanner.scans_path}")


def main():
    parser = argparse.ArgumentParser(description="ECO-LOOP real sensor scan logger")
    parser.add_argument("--port", help="serial port of the ESP32, e.g. COM5")
    parser.add_argument("--simulate", action="store_true",
                        help="practise without hardware (invented numbers, saved in a separate folder)")
    parser.add_argument("--operator", default="unknown", help="who is scanning")
    parser.add_argument("--out", default=None, help="output folder (default: data/real_scans)")
    parser.add_argument("--frames", type=int, default=3, help="readings averaged per light setting")
    parser.add_argument("--ref-gain", type=int, default=2, choices=[0, 1, 2, 3],
                        help="reflectance gain: 0=1x 1=3.7x 2=16x 3=64x")
    parser.add_argument("--ref-cycles", type=int, default=49, help="reflectance integration cycles (1-255)")
    parser.add_argument("--trans-gain", type=int, default=3, choices=[0, 1, 2, 3],
                        help="transmission gain")
    parser.add_argument("--trans-cycles", type=int, default=255, help="transmission integration cycles (1-255)")
    parser.add_argument("--lamp", choices=["manual", "auto"], default="manual",
                        help="manual = you switch the lamp, auto = ESP32 pin 26 switches it")
    parser.add_argument("--lamp-warmup", type=float, default=3.0,
                        help="seconds to wait after the lamp turns on")
    args = parser.parse_args()

    if not args.simulate and not args.port:
        parser.error("give --port COMx, or use --simulate to practise without hardware")

    if args.simulate:
        link = FakeLink()
        out_dir = args.out or os.path.join(PROJECT_ROOT, "data", "practice_scans_NOT_REAL")
        print("\nSIMULATION: numbers are invented. Files go to", out_dir)
    else:
        link = SerialLink(args.port)
        out_dir = args.out or DEFAULT_OUT

    try:
        print("Sensor link:", link.ask("PING"), "|", link.ask("INFO"))
        scanner = Scanner(link, args, out_dir)
        scanner.prepare_files()
        print(f"Session {scanner.session_id} | saving to {scanner.scans_path}")
        command_loop(scanner)
    finally:
        link.close()


if __name__ == "__main__":
    main()
