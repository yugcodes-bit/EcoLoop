# 01 — Where We Are Now

An honest picture of the project on 15 November 2026.
Nothing here is meant to discourage. We need a true starting point so the next steps are solid.

---

## 1. The story so far

1. The idea: a cheap box that tells if an apple is fresh, early rot or severe rot, without cutting it.
2. We had no money for the sensor at first. So we went to IISER Pune with apples.
3. The Shimadzu machine there cannot take a whole apple. We had to cut **thin slices**, with skin and without skin.
4. We got 15 scans. Then we built the software, a simulator, and wrote the paper using synthetic data.
5. Later we bought one AS7265x sensor. It has not been used yet.
6. Going back to IISER costs about Rs 1,000 per scan, which we cannot afford. The project stopped here.

## 2. What we have

| Thing | Status | Honest value |
|---|---|---|
| 15 IISER scans (300 to 1500 nm) | Real | Useful as background evidence. Too small to train anything. |
| Data cleaning and plotting scripts | Working | Fine. |
| Synthetic data + 6 trained models | Working | Only proves the software runs. Not evidence that detection works. |
| FastAPI server + compost engine | Working | Reusable later for a dashboard. |
| Browser simulator | Working | Good for demos. Not evidence. |
| AS7265x sensor | Owned, unused | The most important thing we have. |
| ESP32 | Owned | Enough for the whole project. |
| Paper (camera-ready) | Written | Describes architecture and pilot. See section 5. |

## 3. What the IISER data really shows

The scans are of **slices of apple tissue**, not whole apples. So they show this:

> Rotten apple tissue looks different from fresh apple tissue under Vis-NIR light.

They do **not** show this:

> We can detect rot from the outside of a whole apple.

The second statement is the actual goal. It has not been tested by us yet. This is the main gap.

What the numbers say (reflectance, with skin):

| Region | Fresh slices | Rotten slices |
|---|---|---|
| 700 to 1000 nm (inside our sensor's range) | about 43 to 50% | about 27 to 31% |

Rotten tissue is clearly darker in the near-infrared. That is a real and useful signal.
It supports the plan in file 02: rotten tissue swallows more light.

## 4. Facts about the dataset that everyone on the team should know

1. **Only 4 physical apples.** Two fresh, two rotten. The 15 scans are repeats of these.
   For machine learning, the sample size is 4, not 15.
2. **No early-rot apples at all**, and **no pH was measured**.
3. **The "absorbance" files are not separate measurements.** They are the reflectance values pushed through
   a formula (Kubelka-Munk). So there is one dataset, not two.
4. **`fresh_apple_3` and `fresh_apple_4` are the same file** copied twice.
5. **`rotten_apple_1` is a bad scan** (already excluded in the paper).
6. **`rotten_apple_5` looks like a fresh apple.** It is probably a healthy part of a rotten apple.
   Lesson: rot sits in one place. A reading from a healthy spot says "fresh" even if the apple is rotten elsewhere.
7. **`fresh_apple_1` is a bruised spot**, and it behaves differently from the other fresh scans.
8. **Which scan belongs to which apple is a guess** in the code (`build_master_dataset.py` says scans 1 to 4 are apple A).
   If anyone remembers the real order, write it down.

## 5. About the paper

The paper is honest that the machine learning results come from synthetic data. Still, a few points need care
in any future version or presentation:

- It says each apple was "scanned 15 different times". The data has 15 scans in total.
- It does not say that the samples were slices. A reader will assume whole apples.
- The "167% difference at 1450 nm" depends on the bruised scan and the duplicated file.
  Without them it is about 113%. It is also outside the range of our sensor.
- The pH numbers in the synthetic data were chosen by us (4.2, 3.5, 2.9). The "pH model" only learned the class.
- The "mystery apple" demo in the simulator always works because the simulator itself creates the rotten signal.

None of this is a disaster. The next paper simply has to be built on real sensor data.

## 6. Problems in the code to fix later

| Where | Problem |
|---|---|
| `sensor/read_sensor.py` | The "real sensor" part cannot work. The AS7265x does not allow direct register reads like that. Use the new `scan_logger.py` + ESP32 firmware instead. |
| `sensor/read_sensor.py` | Its fake readings do not match the data the model was trained on. |
| `models/train_models.py` | Train/test split mixes scans from the same apple. On real data this gives falsely high scores. |
| Master Document vs paper | One says Raspberry Pi is required, the other uses ESP32. ESP32 is the answer. |
| `.gitignore` | Ignores every `.csv`. Real scan data will not be saved in git. Back it up another way (see file 05). |

## 7. The one-sentence summary

We have a working software shell, a good idea, real evidence that rotten tissue looks different,
and a sensor in a drawer. We do **not** yet have a single reading from that sensor on a whole apple.
That is the next step, and it costs almost nothing.
