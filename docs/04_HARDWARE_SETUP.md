# 04 — Hardware and Software Setup

Exact connections, parts, and steps to get from "sensor in a drawer" to "numbers in a CSV file".

**Status of the code:** the firmware and the logger are new. The logger was tested in practice mode
(no hardware). The firmware has not been compiled or run on a real board yet. Its function names were
checked against the SparkFun library source. Expect small fixes on the first evening.

---

## 1. ESP32 or Raspberry Pi?

**ESP32. Do not buy a Raspberry Pi.**

- During research, the ESP32 only passes sensor readings to the laptop. The laptop saves and analyses them.
- For the final device, the model will be a small formula (a few dozen numbers). An ESP32 runs that easily.
- The old Master Document said a Pi was needed to load big Random Forest files. With simple models that reason is gone.

Any common ESP32 board works: "ESP32 DevKit V1" or "ESP32-WROOM-32 DevKit" (30 or 38 pins).
The pin numbers below are for these boards. If yours is an ESP32-S3 or C3, tell me the exact model first,
because the default I2C pins differ.

## 2. Parts list

| Part | Have it? | Cost (Rs, approx.) | Note |
|---|---|---|---|
| AS7265x sensor board | Yes | 0 | |
| ESP32 dev board + USB data cable | Yes | 0 | The cable must carry data, not only charge. |
| 4 jumper wires, or a Qwiic cable | ? | 30 to 150 | Qwiic cable if your sensor board has the small white connector. |
| 12 V filament lamp: G4 halogen capsule 10 W or 20 W, with holder | No | 60 to 150 | Must be a **filament** lamp (halogen or old-style bulb). Not LED. |
| 12 V DC adapter, 2 A or more | No | 200 to 300 | 20 W lamp needs about 1.7 A. |
| On/off switch for the lamp | No | 20 | Manual switching is fine to start. |
| Box material: plywood or thick cardboard, matte black paint or black chart paper | No | 150 to 300 | |
| Black foam sheet (EVA) and black tape | No | 100 | For the rings that seal against the apple. |
| White reference: PTFE (Teflon) sheet, 3 mm or thicker | No | 100 to 200 | Or PTFE thread-seal tape wound 15+ layers flat on a card. |
| Lamp reference block: white plastic chopping board, cut and stacked 4 to 5 cm thick | No | 100 | Stays the same forever. Used to check the lamp. |
| Vernier caliper | ? | 150 to 300 | To measure apple diameter. |
| **Total new spend, without apples** | | **about 900 to 1,700** | |

Optional later: a logic-level MOSFET module so the ESP32 switches the lamp by itself (pin 26).

## 3. Wiring

Only four wires. **Power off (USB unplugged) while wiring.**

| AS7265x pin | ESP32 pin | Qwiic wire colour |
|---|---|---|
| 3V3 (or 3.3V) | **3V3** | Red |
| GND | **GND** | Black |
| SDA | **GPIO 21** | Blue |
| SCL | **GPIO 22** | Yellow |

```
        AS7265x board                      ESP32 DevKit
      +---------------+                 +----------------+
      |           3V3 |-----------------| 3V3            |
      |           GND |-----------------| GND            |
      |           SDA |-----------------| GPIO 21        |
      |           SCL |-----------------| GPIO 22        |
      +---------------+                 |            USB |==== laptop
                                        +----------------+
```

Rules:

1. **Never connect the sensor to VIN or 5V.** It is a 3.3 V part. 5 V will destroy it, and we only have one.
2. Connect GND first.
3. Do not use the RX/TX pins on the sensor board. Those are for a different mode.
4. If the board has bare holes instead of a connector, solder header pins. Loose wires give random errors.
5. Keep the wires short (under 20 cm).

## 4. Software on the laptop

### 4.1 Arduino IDE (to program the ESP32, one time)

1. Install Arduino IDE 2 from arduino.cc.
2. Open **Boards Manager**, search `esp32`, install **"esp32 by Espressif Systems"**.
   If it does not appear, add this in File > Preferences > Additional boards manager URLs, then search again:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
3. Open **Library Manager**, search `AS7265X`, install **"SparkFun Spectral Triad AS7265X"**.
4. Plug in the ESP32. Choose Tools > Board > **ESP32 Dev Module**, and Tools > Port > the new COM port.
   If no port appears, install the USB driver for your board's chip (CP210x or CH340; the name is printed
   on the small chip near the USB socket).
5. Open `firmware/ecoloop_scanner/ecoloop_scanner.ino` and press **Upload**.
   If it stays on "Connecting....", hold the **BOOT** button on the ESP32 until it starts.

### 4.2 Quick check in the Serial Monitor

1. Tools > Serial Monitor. Set **115200 baud** and line ending **Newline**.
2. Press the EN (reset) button on the ESP32. You should see: `OK,ECOLOOP_SCANNER,1`
3. Type `PING` and press Enter. Reply: `OK,ECOLOOP_SCANNER,1`
4. Type `LED,1,0,0`. The white LED on the sensor turns on. `LED,0,0,0` turns it off.
5. Type `READ`. Reply: `DATA,` followed by a temperature and 36 numbers.
6. **Close the Serial Monitor** when done. Only one program can use the port at a time.

If you see `ERR,sensor not found`: check the four wires, then press EN again.

### 4.3 Python (to record data)

In PowerShell, from `D:\ECOLOOP\ecoloop`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Practice without hardware (invented numbers, saved in a separate folder):

```powershell
python sensor/scan_logger.py --simulate --lamp auto
```

With the real sensor (replace COM5 with your port):

```powershell
python sensor/scan_logger.py --port COM5 --operator Yugant
```

### 4.4 Logger commands

| You type | What happens |
|---|---|
| `w` | White reference scan (white tile on the reflectance station). Needed once at the start of each session. |
| `l` | Lamp reference scan (reference block on the transmission station). Needed once per session. |
| `r A012 3` | Reflectance scan of apple A012, spot 3. Takes 4 readings: lights off, white, infrared, UV. |
| `t A012 1` | Transmission scan of apple A012, position 1. Takes 2 readings: lamp off, lamp on. |
| `live` | Prints readings for 20 seconds with the white LED on. Use it while adjusting the setup. |
| `note some text` | Saves a note with the time. |
| `q` | Quit. |

After each scan the logger prints the numbers and warns about three things:
light leaking in, saturation (a reading stuck at 65535), and readings that do not repeat.

Files it creates in `data/real_scans/`:

| File | Content |
|---|---|
| `scans.csv` | One row per reading. Filled automatically. |
| `apples.csv` | One row per apple. **You fill this by hand** (file 05). |
| `cuts.csv` | One row per apple you cut open. **You fill this by hand** (file 05). |
| `session_log.txt` | Start and end of sessions, settings, your notes. |

## 5. The reflectance station (Mode A)

Goal: the apple sits in exactly the same place every time, and no room light reaches the sensor.

```
            lid (black inside)
        +-----------------------+
        |         apple         |
        |        _______        |
        |       /       \       |
        |      |         |      |
        |       \_______/       |
   =====|====[ foam ring ]======|=====   black board with a hole
        |      [ sensor ]       |        sensor faces UP, about 10 mm below the apple skin
        +-----------------------+
```

1. Take a flat black board. Cut a hole just big enough for the three sensor windows and the LEDs.
2. Fix the sensor under the hole, facing up.
3. Glue a ring of black foam around the hole, about 10 mm high. The apple rests on this ring.
   The ring fixes the distance and blocks light from the sides.
4. Cover everything with a box painted matte black inside.
5. The same ring is used for the white tile: lay the tile on the ring, white side down.

The 10 mm distance is a starting value. Test 3 mm and 10 mm in the first week and keep the one that gives
the most repeatable readings (section 8, test 3). After that, **never change it**.

## 6. The transmission station (Mode B)

Goal: the sensor only sees light that went through the apple.

```
   lamp chamber            apple chamber
  +--------------+--------------------------+
  |              |          lid             |
  |   (lamp) ---->  hole + foam ring        |
  |    12 V      |      _______             |
  |              |     /       \            |
  |              ||===|  apple  |           |     the apple side presses on the ring
  |              |     \_______/            |
  +--------------+=====[ foam ring ]========+
                        [ sensor ]               sensor under the apple, facing up
```

1. Two chambers with a wall between them. The lamp sits in its own chamber.
2. The wall has a hole of about 25 mm with a foam ring. The apple's side presses against it.
   Light can only leave the lamp chamber by going into the apple.
3. The sensor sits under the apple with its own foam ring, the same as in Mode A.
4. Lamp at the side and sensor at the bottom is a 90° layout. A research group found this layout worked best
   for brown heart in apples. It also gives more light than placing the sensor directly opposite the lamp.
5. Lay the apple **on its side**, so the stem points sideways (not at the lamp and not at the sensor).
6. Keep the lamp 3 to 5 cm from the wall hole. Give the lamp chamber small air holes at the back,
   covered by a black baffle so no light escapes toward the sensor.

Both stations can share one sensor: build the sensor into a small block that slots into either station.
A simpler start is one box that does both jobs, since Mode A is just Mode B with the lamp off.

### Lamp safety

- Use only the 12 V lamp and adapter. **No mains-voltage wiring.**
- A halogen capsule gets very hot. Mount it in its ceramic holder on wood or metal, not on cardboard or foam.
  Never touch the glass.
- Switch the lamp on only for the few seconds of a scan. This protects the box and stops the apple from heating up.
- The UV LED on the sensor is weak, but do not look into it.

## 7. Choosing the sensor settings (once), then freezing them

The sensor has two settings: **gain** (1x, 3.7x, 16x, 64x) and **integration cycles** (1 to 255; more = longer exposure).
The logger's defaults are a starting guess:

| Mode | Gain | Cycles |
|---|---|---|
| Reflectance | 16x | 49 |
| Transmission | 64x | 255 |

How to tune:

1. Reflectance: scan the white tile. The brightest channel should read between about 20,000 and 50,000 raw counts.
   If it shows 65535 or the logger says SATURATED, lower the gain: add `--ref-gain 1` when starting the logger.
   If the brightest channel is under 5,000, raise it: `--ref-gain 3`.
2. Transmission: scan the reference block and a normal apple. Nothing may saturate. More signal is better.
3. Write the final settings on the box and in `session_log.txt` with a `note`.
4. **Use the same settings for every apple in the dataset.** Readings taken with different settings cannot be mixed.

## 8. First-evening tests (no apples needed for the first three)

These cost nothing and tell us if the sensor is healthy.

| # | Test | How | Good result |
|---|---|---|---|
| 1 | **Darkness** | Close the box, type `live off`. | All channels stay low and steady. If they jump when you move your hand near the box, light is leaking. |
| 2 | **Stability** | White tile in place. Do `w` ten times without touching anything. | Each channel changes by less than about 1 to 2% between scans. |
| 3 | **Re-placement** | Lift the white tile and put it back before each of ten `w` scans. | Changes stay under about 3%. If not, the holder needs to hold things more firmly. |
| 4 | **Which channels get light** | Look at one `w` scan. | Write down which channels are near the lights-off level under each LED. Those channels carry no information in Mode A. Expect the near-infrared ones to be weak under the white LED. |
| 5 | **Warm-up** | Scan the white tile every minute for 15 minutes from cold. | Find how many minutes it takes for readings to settle. Use that as the warm-up time from now on. |
| 6 | **One apple, same spot** | `r TEST 1` ten times, lifting and replacing the apple. | Changes under about 5%. This is the noise level any rot signal has to beat. |

Save the numbers from these tests. They go straight into the "methods" section of the paper.

## 9. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ERR,sensor not found` | Wiring, or SDA and SCL swapped | Check the four wires. Press EN. |
| No COM port appears | Charge-only cable, or missing driver | Try another cable. Install CP210x or CH340 driver. |
| Upload stuck at "Connecting..." | Board not in upload mode | Hold BOOT while it connects. |
| Logger says "No reply" | Serial Monitor still open, or wrong port | Close Arduino Serial Monitor. Check the port name. |
| Compile error naming a sensor function | Library version differs | Open the library's own example (File > Examples > SparkFun Spectral Triad AS7265X) and compare the names. |
| Reading stuck at 65535 | Saturated | Lower gain or cycles. |
| Readings near zero with LED on | Sensor facing the wrong way, or hole blocked | Check that the sensor windows see the sample. |
| Lights-off reading is high | Light leak | More black tape and foam. Test with a phone torch around the closed box. |
| Transmission reading barely above lamp-off | Not enough light through the apple | See Experiment 0 in file 05 for the options. |
