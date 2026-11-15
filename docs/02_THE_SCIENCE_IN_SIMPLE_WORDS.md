# 02 — The Science in Simple Words

This file answers the hard questions:
- Is this even possible with one cheap sensor?
- Light does not go deep. What if the other side of the apple is rotten?
- Was our IISER "calibration" idea right?

Links to the research papers behind each claim are in file 03.

---

## 1. What the sensor does

The AS7265x is 18 tiny light meters in one package. Each one only sees one colour of light:

```
410 435 460 485 510 535 | 560 585 610 645 680 705 | 730 760 810 860 900 940   (nanometres)
<------ violet to green ------><---- yellow to deep red ----><---- near-infrared ---->
```

It does not take a picture. It gives 18 numbers: "how much light of each colour reached me".
A laboratory machine like the Shimadzu gives about 1,200 numbers. We get 18. That is the price of being cheap.

The board also has three small LEDs: white, infrared (875 nm) and ultraviolet (405 nm).

## 2. What happens to light inside an apple

An apple is not like a mirror and not like glass. It is like **thick fog**.
Light that enters does not go straight. It bounces off cell walls thousands of times and wanders around.
Along the way, parts of it are absorbed by pigments and water.

Two things decide what comes out:

- **Scattering** (how much the light bounces). It depends on structure: firm cells, air gaps, juice.
- **Absorption** (how much light is eaten). It depends on chemistry: chlorophyll, brown pigments, water.

When tissue rots, both change:

| What rot does to the tissue | What it does to the light |
|---|---|
| Flesh turns brown | Brown pigments eat light below about 750 nm (visible and deep red). |
| Cell walls break down, tissue becomes wet and soft | Less bouncing. Light is no longer returned the same way. |
| Chlorophyll breaks down | The dip near 680 nm changes. |
| Water moves or is lost | Small change near 940 nm. |

A very old and simple trick from the literature: compare light at about **720 nm** with light at about **810 nm**.
Browning eats the 720 light much more than the 810 light. So the ratio 720/810 drops in a damaged apple.
Our sensor has channels at 705, 730 and 810 nm. That is the main reason 18 channels can be enough.

## 3. Three ways to measure, and how deep each one sees

```
 (A) REFLECTANCE            (B) INTERACTANCE           (C) TRANSMISSION

   light   sensor            light      sensor            lamp
     \      /                  |   wall   |                 |
      \    /                   |   ####   |               __v__
    ___\__/___               __v___####___^__            /     \
   /          \             /   \________/   \          | apple |
  |   apple    |           |      apple       |          \_____/
   \__________/             \________________/              |
                                                            v
                                                          sensor

  sees: skin + 2-5 mm        sees: about 1-2 cm           sees: the whole path
                                                           through the fruit
```

- **(A) Reflectance.** Light and sensor on the same side, a gap between sensor and apple.
  Most of what returns has only touched the skin and the first few millimetres.
  One well-known study measured about 4 mm of useful depth at 700 to 900 nm.
- **(B) Interactance.** Same side, but a wall blocks the direct shine, so the sensor only gets light
  that travelled a short way inside the flesh. Sees deeper than (A), roughly a centimetre or two.
- **(C) Transmission.** Lamp on one side, sensor on another side. Every bit of light that reaches the sensor
  has crossed the inside of the apple. Only a very small fraction of the light survives the trip,
  so it needs a strong lamp, a sensitive setting and a box with no light leaks.

Light itself can travel far inside an apple. Other measurements found a little light (about 1%) still present
25 mm deep. The limit of reflectance is not that light cannot go deep. It is that the light which *comes back
to the same side* has mostly stayed shallow.

## 4. Your question: "What if the other side is rotten?"

You are right. With reflectance alone:

- Rot on the far side: **not seen**.
- Rot in the core: **not seen**.
- Rot under the sensor, near the skin: seen.

Our own IISER data shows this problem. One scan of a rotten apple (`rotten_apple_5`) looked fresh,
most likely because that slice came from a healthy part.

There are two fixes, and we will use both.

**Fix 1: scan several spots.** Turn the apple and read 4 to 6 spots around it. If any spot looks rotten,
the apple is rotten. This catches rot near the skin on any side. It still misses the core.

**Fix 2: transmission.** Shine a lamp through the apple. Rot anywhere on the light's path, including the core,
changes what comes out. This is what commercial sorting machines do for internal defects (file 03, section 5).
Researchers who compared the methods on apples found transmission best for internal browning.

One honest limit: light takes many paths, so a very small rotten patch only changes the signal a little.
Even expensive machines struggle when less than about 5% of the fruit is affected.

## 5. So is it possible with just one sensor?

**Yes, with conditions.** One sensor is enough because we move the light, not the sensor:

| Mode | Light source | What it is good at |
|---|---|---|
| A. Reflectance, 4 to 6 spots | The LEDs on the board | Rot and bruises near the skin, any side. |
| B. Transmission, 2 positions | A small 12 V filament lamp | Rot hidden inside: core, far side, deep flesh. |

What needs to be true for this to work, and how sure we are:

| Condition | How sure | How we check |
|---|---|---|
| The sensor gives steady, repeatable numbers | Very likely | First evening tests (file 04). |
| Enough lamp light gets through an apple for the sensor to measure it | Likely, not proven | **Experiment 0** (file 05). This is the first big unknown. |
| Rot changes the numbers more than normal apple-to-apple differences do | Likely for large rot, unknown for early rot | Experiment 1, then the main dataset. |

Why we think the light level will be enough: a research group built a handheld moldy-core detector with a
tiny Hamamatsu chip spectrometer and a small microcontroller, and it worked in transmission. Each of our
sensor's channels collects light over a wider colour band than one pixel of that spectrometer.
That is a reason for hope, not a proof. We test it first.

A note on the LEDs on the board: the white LED gives almost no light above about 750 nm, and the infrared LED
only gives light near 875 nm. So in Mode A some near-infrared channels (730, 760, 940) may get very little light.
The first test with the white tile will show this. The filament lamp does not have this problem,
because a hot filament gives light at all 18 wavelengths.

## 6. What this system will not be able to do

Being clear about limits makes the project stronger, not weaker.

| Not realistic | Why |
|---|---|
| Finding a tiny rot spot (a few mm) deep inside | The change in light is smaller than normal variation between apples. |
| Measuring pH to ±0.3 | A published study got only weak pH prediction even with imaging. Rot can also push pH up or down depending on the fungus. We never measured pH. |
| Working on a new apple variety without new training data | Skin colour and flesh differ between varieties. |
| Matching the Shimadzu's numbers | Different machine, different geometry. Not needed. |
| Working in open room light | The sensor would read the room, not the apple. The dark box is required. |

## 7. Things that confuse the sensor (and what we do about them)

| Confuser | Effect | What we do |
|---|---|---|
| Skin colour (red vs green side) | Changes everything below about 650 nm | Rely mostly on 680 nm and above. Scan several spots. |
| Apple size | A bigger apple lets less light through, and can look "rotten" | Measure the diameter of every apple. Use ratios of channels, not raw brightness. |
| Lamp brightness changing | Looks like a change in the apple | Scan a reference block at the start of every session. Use ratios. |
| Sensor warming up | Readings drift | Log the temperature (the firmware does). Warm up 5 minutes. |
| Apple placed differently | Different path through the fruit | Fixed holder, marked positions, repeat scans. |
| Light leaking into the box | Adds a false signal | Lights-off reading before every scan (the logger does this automatically). |

## 8. The IISER idea: was it right?

**The thinking was reasonable. It just is not needed, and it cannot work the way we did it.**

What we planned has a name in the field: *calibration transfer*. A model is built on an expensive "master"
instrument, and a mathematical bridge lets a cheap "slave" instrument use that model.

Why it does not fit our case:

1. **Calibration transfer needs the same sample measured the same way on both machines.**
   The Shimadzu measured thin slices. Our sensor will measure whole apples. A slice and a whole apple are
   different objects to light. No formula connects them reliably.
2. **It is only useful if a big, good model already exists on the master machine.**
   We have 4 apples on the Shimadzu. There is nothing to transfer.
3. **It costs Rs 1,000 per scan.** A dataset of 100 apples would cost a lakh. The direct way costs a few thousand.
4. **The Shimadzu is not the truth we care about.** The truth is: was this apple rotten inside, yes or no,
   and how much? A knife and a camera answer that for free.

The better path is direct:

```
   OLD IDEA:   apple -> Shimadzu -> formula -> cheap sensor -> answer        (expensive, indirect)

   NEW PLAN:   whole apple -> cheap sensor reading
                           -> cut it open, photograph, measure the rot        (the truth)
               do this for 100+ apples -> the model learns directly
```

This is also how the published low-cost sensor studies work. They train the cheap sensor on its own readings.

## 9. Was the IISER trip a waste? No.

The IISER data still does three useful jobs:

1. **It shows the basic physics is real**: rotten tissue reflects much less near-infrared than fresh tissue.
2. **It shows where skin matters.** With and without skin, the curves differ mostly in the visible range,
   much less in the near-infrared. That supports relying on the 680 to 940 nm channels.
3. **It is good material for the "motivation" part of a paper and for a funding pitch**, described honestly
   as tissue-level evidence from a reference instrument.

When might IISER be worth it again? Later, and small: once our own device works, a handful of reference scans
could add credibility to a paper. That would be perhaps 6 to 10 scans, not hundreds. See file 06.

## 10. Summary

- Reflectance sees shallow. Transmission sees through. Use both with the same sensor.
- 18 channels are few, but internal rot mostly shows up as a simple change between about 700 and 810 nm,
  and we have channels there.
- The sensor learns from cut-open apples, not from the Shimadzu.
- The first real unknown is light level in transmission. One evening of testing answers it.
