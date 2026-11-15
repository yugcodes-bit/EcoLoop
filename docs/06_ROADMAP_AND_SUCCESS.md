# 06 — Roadmap, Contribution, and Chances of Success

Where we are going, why it is worth doing, how long it takes, what it costs,
how likely it is to work, and what to do about IISER and funding.

---

## 1. The end goal

> **A box costing under Rs 10,000 in parts. You place a whole apple inside, close the lid, and in a few seconds
> it shows Fresh, Early Rot or Severe Rot. The apple is not cut or damaged.
> And we can state, with real test data, how small a rotten patch it can find.**

What "done" looks like:

1. A working box: one AS7265x, one ESP32, one small lamp, a display or three LEDs.
2. A dataset of 120+ real apples: sensor readings, cut photos, rot percentage. Shared openly.
3. Test results on apples the model never saw during training.
4. A paper built only on real sensor data.
5. A short demo video: scan, show the result, cut the apple, show the inside.

## 2. The research questions

| # | Question | Why it matters |
|---|---|---|
| RQ1 | Can an 18-channel chip sensor detect rot hidden inside a whole apple when light is sent through the fruit? | Not answered in the literature we found. |
| RQ2 | How does shining through (transmission) compare with same-side reading (reflectance) on the same apples with this sensor? | Tells builders which design to use. |
| RQ3 | What is the smallest amount of rot it can detect reliably? | An honest limit that others can cite. |
| RQ4 | Can the final model run on the ESP32 itself? | Shows it is a real low-cost device, not a laptop demo. |

## 3. Why this is a real contribution

| Contribution | Why reviewers will care |
|---|---|
| First test (as far as we found) of this sensor class in transmission for hidden rot | Packhouse machines use transmission with costly spectrometers. Nobody has shown what a chip costing a few thousand rupees can do. |
| A detection-limit curve (detection rate vs rot percentage) | Most low-cost papers report one accuracy number. A limit curve is more honest and more useful. |
| Open dataset with cut-open truth | Low-cost sensor data with destructive ground truth is rare. Others can build on it. |
| Reflectance vs transmission on the same apples, same sensor | A clean comparison that guides future designs. |
| Indian context: hidden core rot in Delicious-type apples | A documented local problem (file 03, section 7). |
| Very low cost and on-device result | Fits small traders, cold stores, farmer groups. |

**It stays a good paper even if accuracy is modest.** "A cheap sensor finds rot above X% and misses it below X%"
is a publishable finding. What is not publishable is synthetic data. That is the difference from the last round.

## 4. What changes from the old plan

| Old plan | New plan | Reason |
|---|---|---|
| Calibrate the sensor against the Shimadzu | Train the sensor directly on cut-open apples | File 02, section 8. |
| Reflectance at one spot | Reflectance at 4 spots + transmission at 2 positions | Rot is local and can be hidden deep. |
| Synthetic data for the results | Synthetic data only for testing software | Results must be real. |
| pH prediction as a main feature | Dropped as a headline. Optional small side study if someone buys a pH meter. | Never measured, and hard even for better instruments. |
| Compost recipe from predicted pH | Compost advice from the rot class (and measured pH if available) | Keeps the "loop" idea without resting on an unproven number. |
| Raspberry Pi | ESP32 | Enough, and cheaper. |
| Random Forest as the main model | Simple linear models first | Small data, and they fit on a microcontroller. |
| Random train/test split of scans | Split by apple, final test on a separate purchase | Avoids fooling ourselves. |

The existing FastAPI server, compost engine and simulator are not wasted. They become the optional
dashboard layer once real models exist.

## 5. Phases and decision points

Times assume a student team working part-time.

| Phase | Weeks | What happens | Decision at the end |
|---|---|---|---|
| 0. Bring-up | 1 | Wire, upload, first-evening tests (file 04). | Sensor steady and repeatable? |
| 1. Build + Experiment 0 | 1 | Build the box and lamp. Test light through an apple. | Enough light? If not, switch to the shorter-path layout. |
| 2. Experiment 1 | 2 to 3 | 15 apples, daily scans, staged cutting. | Does rot separate from controls? Continue, narrow the claim, or stop and rethink. |
| 3. Main dataset | 5 to 6 | 120 apples in 3 purchases. | Enough apples in each class? |
| 4. Models | 2 | Clean data, simple models, test on purchase 3, detection-limit curve. | Which mode and which channels matter? |
| 5. Device | 1 to 2 | Put the model on the ESP32, add a display, record the demo. | |
| 6. Paper | 2 to 3 | Write, with the dataset and code released. | |
| **Total** | **about 14 to 18 weeks** | | |

The important design choice: **money and time are spent in small steps, and each step can stop the project
cheaply.** After week 2 we know if the light level works. After week 5 we know if rot is visible to the sensor.
Only then do we buy 120 apples.

## 6. Budget

| Item | Rs (approx.) |
|---|---|
| Lamp, adapter, box, foam, references, wires, caliper | 900 to 1,700 |
| Apples, Experiment 1 | 300 to 500 |
| Apples, Experiment 2 | 2,500 to 4,000 |
| Small display or LEDs for the final device | 150 to 300 |
| Spare money for mistakes | 500 |
| **Total** | **about 4,500 to 7,000** |

Compare: 120 apples at IISER would cost about Rs 1,20,000 at Rs 1,000 per scan, for one scan each.

## 7. Chances of success

These are my judgement from the literature and from our own data. They are estimates, not calculations.
Experiment 0 and Experiment 1 will replace them with facts.

| Outcome | Chance | Why |
|---|---|---|
| Sensor works and gives repeatable numbers | about 95% | Many groups have used it without trouble. |
| Enough lamp light gets through an apple (or the short-path fallback works) | about 80% | A handheld device with a chip spectrometer managed it. Our sensor has wider channels. Not yet shown for this chip. |
| Telling fresh apples from **severe** rot, any side, whole apple | about 75 to 85% | Large rot changes the light strongly. One sorting-machine maker's patents describe doing it with just two light bands. |
| Three classes, with early rot (around 5 to 15%) found most of the time | about 45 to 55% | Early rot is hard even for research rigs. Middle classes are always the weakest. |
| Finding very small rot (under 5%) | about 20% | Expensive equipment also struggles here. |
| pH within ±0.3 | about 10% | Not pursued. |
| A publishable paper from real data | about 80%, if Experiment 2 is completed properly | The result is useful whether accuracy is high or modest. |

In plain words: **a working fresh-vs-rotten device is likely. A reliable early-rot detector is a coin toss.
A solid, honest paper is very likely if the data collection is done carefully.**

## 8. Main risks and what we do about them

| Risk | What we do |
|---|---|
| Not enough light in transmission | Stronger lamp, closer lamp, reflector, shorter-path layout (file 05, Experiment 0). |
| Apple size confuses the model | Record diameter. Use channel ratios. Keep apples in a similar size range at first. |
| Rot we start ourselves differs from natural rot | Include natural cases. State this limit in the paper. |
| The only sensor breaks | Handle with care, 3.3 V only, fixed mount. Consider a second one if any money comes in (it also lets us test if two units agree). |
| Results only hold for one variety | Say so. One variety done well is better than three done badly. A second variety is future work. |
| Team loses momentum again | Short phases with clear decisions. One fixed scanning slot per day. One person owns the data backup. |
| Lamp or sensor drifts over weeks | References every session. Log temperature. Check the reference block readings over time. |

## 9. IISER and funding

### Should we go back to IISER now? No.

We do not need the Shimadzu for the device to work (file 02, section 8). Paying Rs 1,000 per scan now buys nothing
the knife does not give us for free.

### When does IISER (or any lab) make sense?

After Phase 4, when we have our own results. Then a **small** reference study adds credibility:

- 6 to 10 samples measured on a laboratory instrument, to show that the tissue-level spectra match what our
  sensor's channels suggest. Cost: Rs 6,000 to 10,000, or free if a lab is interested.

### How to get a lab or a professor interested

Nobody gives equipment time to an idea. People give time to **evidence plus a clear ask**.
After Experiment 1 we can walk in with:

1. A working box on the table.
2. One page with real plots: rotten vs fresh apples, from our own sensor.
3. A specific, small request: "ten scans" or "one afternoon", not "access to the lab".
4. Something for them: co-authorship or acknowledgement, an open dataset, a student project that is already running.

Places to try, cheapest first:

- Our own college: physics, chemistry or instrumentation labs may have a UV-Vis or a fibre spectrometer.
  Our guide can ask.
- Other Pune institutes with food, agriculture or chemistry departments.
- IISER, with results in hand.

### Funding options to look into

I have not checked current deadlines or rules for any of these. Treat this as a list to investigate.

| Option | Note |
|---|---|
| College research or project grant | The whole remaining budget is under Rs 7,000. This is the most realistic first source. |
| NIDHI-PRAYAS (Department of Science and Technology) | A prototype grant for individual innovators, given through centres. Venture Center in Pune is listed as one. Sources give different ceilings (Rs 10 lakh in older pages, more in newer ones). Needs a physical prototype idea with commercial potential. Worth a look only after Experiment 1. |
| Hackathons and student innovation contests | A working box and a cut-the-apple demo present very well. |
| Student paper and project awards | After the dataset exists. |

### The pitch in five sentences

1. Hidden rot in apples cannot be seen from outside, and it is common in the Delicious apples sold across India.
2. Packhouse machines find it by shining light through the fruit, but they are industrial sorting lines, far out of reach for small traders.
3. We built the same idea with a single chip sensor and an ESP32 for under Rs 10,000.
4. Here is our own data from real apples, with the cut-open photos, showing what it can and cannot detect.
5. We need (a small, specific thing) to finish the test on 120 apples and publish the dataset openly.

Sentence 4 is the one we cannot say today. Experiment 1 is how we earn it.

## 10. The existing paper

If it can still be edited, three corrections are worth making: the samples were slices, there were 15 scans in total,
and the 1450 nm figure should be recalculated without the duplicate file (see file 01, section 5).

If it is already final, leave it. It is honestly framed as an architecture and pilot paper.
The next paper will be the real-data one and can cite it as the starting point.

## 11. Next three actions

1. Wire the sensor and run the first-evening tests (file 04, sections 3, 4 and 8).
2. Buy the lamp parts and build the box (file 04, sections 2 and 6).
3. Run Experiment 0 (file 05). Bring the numbers back, and we decide the next step from them.
