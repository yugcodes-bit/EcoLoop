# 05 — Data Collection Protocol

How to take readings so that the data is worth something. Do the experiments in order.
Each one ends with a decision, so we never spend weeks on something that cannot work.

---

## The rules that protect the data

1. **One apple, one ID.** Write the ID on the apple with a marker (A001, A002, ...). Never reuse an ID.
2. **Same settings for the whole dataset.** Same gain, cycles, distance, lamp, lamp position, box.
   If anything must change, start a new dataset and write a `note`.
3. **References first.** Every session starts with `w` (white tile) and `l` (lamp block). The logger insists on this.
4. **The cut is the truth.** An apple only gets a class label after it has been cut and photographed.
   Never label an apple by how it looks from outside.
5. **Never test on apples the model trained on.** All scans of one apple go together, either into training or into testing.
6. **Back up after every session.** Copy the whole `data/real_scans/` folder and the photos to Google Drive.
   Git ignores `.csv` files in this project, so git will not save them.
7. **Write down anything odd** with the `note` command: a dropped apple, a power cut, a different operator.

---

## Experiment 0 — Can the sensor see lamp light through an apple? (one evening)

This is the first big unknown. Everything about Mode B depends on it.

**You need:** the transmission station from file 04, 3 normal apples of different sizes, the reference block.

**Steps**

1. Warm up 5 minutes. Start the logger.
2. `l` with the reference block in place.
3. For each apple: `t E01 1`, turn it half a turn, `t E01 2`. Repeat each scan 3 times.
4. Look at the line the logger prints: *"light through the sample (705-860 nm): ... counts, ...x the lamp-off level"*.

**Decision**

| What you see | Meaning | Next step |
|---|---|---|
| 50x the lamp-off level or more, readings repeat within about 5% | Plenty of light | Go to Experiment 1. |
| Between 10x and 50x | Usable | Go to Experiment 1, but try the improvements below first. |
| Under 10x | Too weak | Try the improvements. If still under 10x, see the fallback. |
| A reading of 65535 | Too much light or a leak | Check for leaks around the apple first. Then lower gain. |

**Improvements, in this order**

1. Hunt for light leaks. With the lamp on and **no apple** but the hole blocked by black foam, the sensor must read
   the same as lamp-off. If it reads more, light is going around, not through.
2. Move the lamp closer to the hole (not closer than 2 cm).
3. Use a 20 W lamp instead of 10 W.
4. Put a small reflector (kitchen foil) behind the lamp.
5. Try a smaller apple.

**Fallback if transmission stays too weak:** use a shorter light path. Put the lamp ring and the sensor ring
on the same side, about 3 cm apart, with a foam wall between them (this is "interactance" in file 02).
It sees less deep than full transmission but deeper than plain reflectance. The project continues with that.

---

## Experiment 1 — Does rot change the readings? (about 2 weeks, 15 apples)

A small test before the big dataset. Apples: same variety, similar size, bought together, no visible damage.

### Groups

| Group | Apples | Treatment |
|---|---|---|
| Control | 5 | A clean needle prick only (no mold). This checks that the hole itself is not what we detect. |
| Side rot | 5 | Mold introduced 5 mm deep at one marked point on the side. |
| Core rot | 5 | Mold introduced into the core through the bottom (calyx) end. Nothing visible from the side. |

### How to start rot (inoculation)

1. Find one naturally moldy apple (blue-green mold) from a fruit vendor. This is the mold source.
2. Side rot: dip a clean toothpick in the moldy part. Prick 5 mm deep at a point marked on the apple's side.
3. Core rot: dip a long thick needle or thin skewer in the moldy part. Push it from the bottom dimple (calyx)
   straight up into the core, about 3 cm. Cover the hole with a small piece of tape.
4. Controls: do the same pricks with a clean tool.
5. Keep all apples at room temperature in a covered, ventilated carton, not touching each other.

Rot usually becomes clear within one to two weeks at room temperature. Speed varies, so scan every day.

**Safety:** moldy apples can carry a toxin (patulin). Do not eat any apple from these experiments.
Wash hands after handling. Anyone with a mold allergy or asthma should wear a mask or leave this step to others.
Throw apples away in a sealed bag. Clean the foam rings with a dry cloth if juice touches them.

### Daily routine (same time each day if possible)

1. Warm up, `w`, `l`.
2. For every apple:
   - Reflectance at 4 spots around the middle, a quarter turn apart: `r A001 1` ... `r A001 4`.
     Spot 1 is always the marked point. Draw small dots for spots 2, 3, 4.
   - Transmission at 2 positions: `t A001 1` (marked point toward the lamp), `t A001 2` (half a turn).
3. Take one outside photo of each apple next to its ID.
4. `note` anything unusual.

Scan every apple **before** inoculating it too (day 0). That day-0 scan is a known "fresh" reading of the same apple.

### Cutting schedule

Cut 1 apple from each group on days 4, 6, 8, 11 and 14 (adjust if rot is faster or slower).
Always scan first, then cut the same day. Follow "How to cut and label" below.

### Decision after Experiment 1

Plot each apple's readings against days. Compare inoculated apples with controls.

| What you see | Meaning | Next step |
|---|---|---|
| Side-rot apples separate from controls in reflectance at the marked spot, and core-rot apples separate in transmission | The method works | Go to Experiment 2. |
| Only large, late rot separates | It works for severe rot only | Go to Experiment 2, but the claim becomes "detects severe hidden rot". Still a result. |
| Controls drift as much as rotten apples | Normal ageing or setup noise is as big as the rot signal | Fix repeatability first (file 04, section 8). Do not start Experiment 2 yet. |
| Nothing separates | The signal is not there in this setup | Stop and rethink with the data in hand before buying more apples. |

---

## Experiment 2 — The main dataset (5 to 6 weeks, 120 apples)

### Plan

| Item | Number |
|---|---|
| Apples | 120, in 3 purchases of 40, each from a different vendor or week |
| Per purchase | 12 controls, 14 side rot, 14 core rot |
| Scans per apple per session | 4 reflectance spots + 2 transmission positions |
| Scan days | Day 0 (before treatment), then every 2 days until the apple is cut |
| Cut days | Spread from day 3 to day 16, so we get every stage from slight to severe |

Why three purchases: the third purchase is **never used for training**. The model is built on purchases 1 and 2
and tested on purchase 3. That is the only honest way to know if it works on apples it has not met.

Also add any naturally defective apples you can get (soft, bruised, old stock from a vendor).
Mark them as `natural` in `apples.csv`. Rot we start ourselves is not identical to natural rot,
so a few natural cases make the test more convincing.

### `apples.csv` — fill one row when an apple enters the study

| Column | Example | Note |
|---|---|---|
| apple_id | A037 | |
| variety | Royal Delicious | Ask the vendor. |
| bought_date | 2026-12-02 | |
| source | Market Yard, vendor 2 | |
| treatment | control / side / core / natural | |
| treatment_date | 2026-12-03 | |
| diameter_mm | 71 | Widest point, with the caliper. |
| weight_g | 168 | Optional. |
| notes | small scar near stem | |

### How to cut and label — `cuts.csv`

1. Scan the apple one last time. Then cut it within the hour.
2. **Cut 1:** across the middle (equator), through the marked point. Photograph the cut face from straight above,
   with a ruler and the ID card in the picture. Same lighting every time.
3. **Cut 2:** one half again, top to bottom through the core. Photograph.
4. Measure the rotten or brown area as a percentage of the whole cut face, for each cut.
   Use ImageJ (free): outline the brown area, outline the whole face, divide. A clear grid sheet and counting squares also works.
5. Fill one row:

| Column | Example | Note |
|---|---|---|
| apple_id | A037 | |
| cut_date | 2026-12-11 | |
| rot_percent_equator | 12 | From cut 1. |
| rot_percent_axis | 18 | From cut 2. |
| rot_location | core / side / both / none | |
| class_label | 0, 1 or 2 | See below. |
| photo_files | A037_c1.jpg; A037_c2.jpg | |
| ph_measured | | Leave empty unless a pH meter is used. |
| notes | rot reached the skin on one side | |

### Class labels (starting definition)

Take the larger of the two percentages.

| Class | Name | Rule |
|---|---|---|
| 0 | Fresh | 0%: no brown or rotten tissue inside |
| 1 | Early rot | More than 0% and up to 15% |
| 2 | Severe rot | More than 15% |

The 15% line is a first choice, not a law. Because we store the actual percentage, the line can be moved later
without re-doing any work. Report the percentage in the paper, not only the class.

Which scans get a label? Only scans from the day the apple was cut, plus the day-0 scan (label: fresh).
Scans from the days in between show the trend but have no exact label.

---

## What we will do with the data (short version)

1. **Clean:** subtract the lights-off reading. Divide reflectance by the white tile, transmission by the lamp block.
2. **Features:** channel ratios such as 730/810 and 705/810, plus the 18 normalised values and apple diameter.
3. **Per-apple summary:** for reflectance, the "worst" spot out of 4. For transmission, the average of the 2 positions.
4. **Models:** simple ones first: logistic regression, LDA, PLS-DA. Random Forest only as a comparison.
5. **Testing:** train on purchases 1 and 2, test on purchase 3. Inside training, cross-validate by apple, never by scan.
6. **Report:**
   - Confusion matrix for the 3 classes.
   - Reflectance alone vs transmission alone vs both together.
   - **Detection rate against rot percentage.** This curve shows the smallest rot the device can find.
     It is the most valuable figure of the whole project.

## Time and cost for the data

| Item | Estimate |
|---|---|
| One apple, one session (6 scans) | about 2 to 3 minutes |
| A session with 40 apples | about 2 hours |
| Apples for Experiment 1 (15) | Rs 300 to 500 |
| Apples for Experiment 2 (120) | Rs 2,500 to 4,000, less at wholesale rates |
