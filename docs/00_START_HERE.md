# ECO-LOOP — Start Here

Written 15 November 2026. This folder is the full rethink of the project, in simple words.
Read the files in order. Each one answers one question.

| File | The question it answers |
|---|---|
| `01_WHERE_WE_ARE_NOW.md` | What do we really have today? What is real and what is not? |
| `02_THE_SCIENCE_IN_SIMPLE_WORDS.md` | Can one cheap sensor see rot inside an apple? What about the other side? Was the IISER idea right? |
| `03_LITERATURE_AND_MARKET.md` | What have other researchers and companies already done? Where is the gap for us? |
| `04_HARDWARE_SETUP.md` | Exact wiring, parts, software setup, and how to build the scanning box. |
| `05_DATA_COLLECTION_PROTOCOL.md` | Exactly how to take readings, which apples, how to label them. |
| `06_ROADMAP_AND_SUCCESS.md` | End goal, why it is a real contribution, timeline, budget, chances of success, IISER and funding. |

The two older files (`EcoLoop_v2_Master_Document.docx` and the camera-ready paper) stay here as history.
Where they disagree with these new files, the new files are the current plan.

---

## The rethink in one page

**1. The idea is good. The path was wrong.**
We went to IISER to "calibrate" the cheap sensor against the Shimadzu. That is not needed.
The cheap sensor does not have to copy the Shimadzu. It only has to learn, from its own readings,
which apples were rotten when we cut them open. The knife is our ground truth, not the Shimadzu.
So we do not need Rs 1,000 per scan. We need apples, the sensor we already own, and a good routine.

**2. Your worry about depth is correct, and it has a known answer.**
Light bounced off the surface (reflectance) only tells you about the first few millimetres under the sensor.
It cannot see the core or the far side. The machines used in big packhouses solve this by shining a
lamp **through** the apple and reading the light that comes out the other side (transmission).
Rotten tissue blocks and colours that light. We can copy this method with a Rs 100 lamp.

**3. So the new design uses the same single sensor in two ways.**
- **Mode A, reflectance:** the sensor's own LEDs, a few spots around the apple. Good for rot near the skin.
- **Mode B, transmission:** a small 12 V lamp on one side, the sensor on another side. Good for rot hidden inside.

**4. Nobody seems to have tested this sensor this way.**
Our search found the AS7265x used for sugar, ripeness and shelf life, always by reflectance.
We found no study using it in transmission to find hidden rot. That is our opening (see file 03).

**5. ESP32 is enough. Do not buy a Raspberry Pi.**
The laptop saves the data during the research. The final model will be small enough to run on the ESP32.

**6. Some old parts of the plan are dropped or made smaller.**
- Synthetic data: only for testing software, never as a result.
- pH prediction: no longer a headline. It was never measured, and the literature says it is very hard.
- Shimadzu calibration: not needed.
- Compost advice: stays, as a small add-on driven by the rot class.

---

## What to do this week

1. Read files 01 and 02 as a team (about 30 minutes).
2. Wire the sensor to the ESP32 and upload the firmware (file 04, sections 2 to 4).
3. Run the practice mode once so everyone knows the commands:
   `python sensor/scan_logger.py --simulate`
4. Do the first-evening checks in file 04, section 8. They cost nothing.
5. Buy the lamp parts (about Rs 300 to 500) and do **Experiment 0** in file 05.
   This one experiment tells us if light from the lamp reaches the sensor through an apple.
   Everything else depends on that answer, so it comes first.

## New files in the project

| File | What it is |
|---|---|
| `firmware/ecoloop_scanner/ecoloop_scanner.ino` | Program for the ESP32. Reads the sensor when the laptop asks. |
| `sensor/scan_logger.py` | Program for the laptop. You type which apple you are scanning; it saves everything to CSV. |

Both are new and have **not been run on real hardware yet** (the logger was tested in practice mode only).
Expect to fix small things on the first evening.
