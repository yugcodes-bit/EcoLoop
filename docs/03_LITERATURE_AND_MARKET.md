# 03 — Literature Survey and Market Systems

What others have already done, in simple words, and where the gap is for ECO-LOOP.

**How this survey was made, and its limits.** It comes from web searches on 15 November 2026.
For most papers only the abstract or a summary was read, not the full text.
Before citing any paper, one of us must open it and check the numbers.
Our claim of a "gap" (section 6) is based on not finding something. A deeper search is needed before
writing "first" in a paper.

---

## 1. How deep does light go in an apple?

| Study | What they found | Why it matters to us |
|---|---|---|
| Lammertyn et al., 2000, *Postharvest Biology and Technology* ('Jonagold' apples) | Useful depth for reflectance-type measurement: up to about 4 mm at 700 to 900 nm, 2 to 3 mm at 900 to 1900 nm. | Reflectance sees only shallow tissue. This is the worry the team raised. |
| Work cited in an NIR review ([IntechOpen chapter](https://intechopen.com/chapters/56041)) | Light intensity inside an apple falls to about 1% at 25 mm depth (700 to 900 nm). | Light does travel deep. A through-the-apple measurement is possible but the signal is weak. |
| Fraser, 2001, PhD thesis, University of Waikato ([PDF](https://researchcommons.waikato.ac.nz/bitstreams/b5bcfa18-3b3f-4289-a22b-0d95cb40158c/download)) | Measured light levels inside fruit with a probe. Transmission mode performed better than reflectance. | Supports adding a transmission mode. |
| [Depth-resolved imaging study, *Foods* 2023](https://mdpi.com/2304-8158/12/9/1783) | Skin cannot be ignored in Vis-NIR measurements of apple. | Skin colour is a confuser, mostly in visible light. |

The two depth numbers (4 mm and 25 mm) are not in conflict. They measure different things:
how deep the *returned* light has been, versus how far *any* light gets.

## 2. Finding hidden defects in whole apples (laboratory and packing-line research)

| Study | Defect | Method | Result (as reported) | Lesson for us |
|---|---|---|---|---|
| Upchurch et al., 1997 | Internal breakdown | Light through the apple | Damaged apples absorb more light below 750 nm. Classified using the ratio of 720 nm to 810 nm. | Two wavelengths were enough. We have 705, 730 and 810. |
| Clark, McGlone, Jordan, 2003 | Brownheart in 'Braeburn' | Transmission NIR, 4 layouts compared | Best: lamp and detector at right angles on the equator, average of opposite sides. Several views needed because browning is uneven. | Use a 90° layout and scan from two sides. |
| McGlone et al., 2005 | Brownheart, on a moving line | Transmission, 650 to 950 nm | R² about 0.9 when two readings around the fruit were averaged. | The useful range (650 to 950 nm) sits inside our sensor's range. |
| [Khatiwada et al., 2016](https://agris.fao.org/search/en/records/65def42f0f3e94b9e5d342a5) | Flesh browning | Compared interactance, partial and full transmission | Full transmission was best. Over 95% correct on a separate test set. | Transmission beats same-side measurement for internal browning. |
| [Huang, Lu, Chen, 2020](https://www.sciencedirect.com/science/article/abs/pii/S0925521419309305) | Internal defects, 430 'Honeycrisp' apples | Six light fibres around the fruit | About 89 to 93% accuracy depending on orientation. | Around 90% is what a good research rig gets. Sets a realistic ceiling. |
| Shenderey et al., 2010, *Food and Bioprocess Technology* (DOI 10.1007/s11947-009-0256-1) | Moldy core in 'Red Delicious' | Transmission with mini-spectrometers | High accuracy, but unsure below about 5% moldy tissue. | Very small rot is hard even for good equipment. |
| [Online moldy core study, *Agriculture* 2022](https://www.mdpi.com/2077-0472/12/4/489) and [2024 study](https://link.springer.com/10.1186/s40538-024-00588-8) | Moldy core | Transmission on a conveyor | Around 97 to 99% for healthy vs moldy in their sample sets. | Shows the method is mature with full spectrometers. |
| [2016 study, IJABE](https://ijabe.org/index.php/ijabe/article/view/2235) | Moldy core and its type | Transmission, 200 to 1025 nm | About 97% for presence, about 81% for type. | Presence is much easier than grading. Our 3 classes will be harder than 2. |
| 2019 study (DOI 10.1016/j.infrared.2019.05.015) | Moldy core | Transmission | Big healthy apples get called diseased; small diseased apples get called healthy. | **Apple size must be recorded and corrected for.** |

## 3. Low-cost and portable devices for hidden defects

| Study | Hardware | Result | Lesson for us |
|---|---|---|---|
| [Liu et al., 2024, *Computers and Electronics in Agriculture*](https://doi.org/10.1016/j.compag.2024.108898) | Handheld moldy-core detector: Hamamatsu C12880MA chip spectrometer + STM32 microcontroller, transmission | 97.3% on their test set. Helped most with apples under 6% mold. | The closest work to ours. Their sensor is a real spectrometer (hundreds of channels) and costs far more than ours. Nobody seems to have asked: can an 18-channel chip do it? |
| [Portable pear browning instrument, *Horticulturae* 2023](https://doi.org/10.3390/horticulturae9080944) | Portable commercial instrument | Browning near the core was hard to detect because depth depends on fruit size and shape. | Core defects are the hardest case. Be honest about this. |

## 4. What has been done with our exact sensor (AS7265x)

| Study | Fruit and task | Method | Result |
|---|---|---|---|
| [Tran and Fukuzawa, 2020, *Sensors*](https://www.mdpi.com/1424-8220/20/20/5883) | Apple, sugar content | Same-side measurement, compared with a USB2000+ spectrometer | R² 0.861 |
| [Noguera et al., 2022, *Agriculture*](https://mdpi.com/2077-0472/13/1/4) | Red grapes, ripening | Handheld device, field use | R² 0.70 for sugar, 0.67 for acidity |
| Mohammed et al., 2023, *Sustainability* | Dates, shelf life | AS7265x + TinyML | Already cited in our paper |
| Banana study, *Jurnal Keteknikan Pertanian* | Natural vs carbide-ripened banana | AS7265x + LDA | Clean separation in their samples |
| [Carbide ripening preprint (arXiv)](https://arxiv.org/pdf/2608.13073) | Climacteric fruits | AS7265x + **ESP32** in a closed chamber lined with matte black felt | Same parts as us. Useful build ideas. |

What all of these have in common: they measure from the same side (reflectance or interactance),
and they predict a property of the whole fruit (sugar, ripeness). None looks for a hidden rotten region.

Two more useful lessons from this group:
- Simple models (linear regression, LDA) are what worked. Nobody needed deep learning.
- Acidity was predicted worse than sugar (grapes: 0.67). Another study (Kavuncuoğlu 2023, in our paper)
  got only R ≤ 0.383 for apple pH. This is why pH should not be our headline.

## 5. What is on the market

| System | Type | How it finds internal defects | Rough cost |
|---|---|---|---|
| [TOMRA Inspectra²](https://www.tomra.com/food/machines/inspectra2) and Spectrim | Packhouse sorting line | Near-infrared light **transmitted through every fruit**. Lists core rot, internal browning and water core. | Industrial line equipment. Far beyond a small trader. |
| MAF Roda internal-defect systems | Packhouse sorting line | Their patents describe a simple "index of defectiveness" from the **ratio of two transmitted light bands**. | Industrial. |
| Greefa, Aweta, Unitec and similar | Packhouse sorting lines | Internal-quality modules of the same kind. | Industrial. (Not checked in this search.) |
| [Felix Instruments F-750](https://cid-inc.com/static/media/uploads/brochures/f750_web.pdf) | Handheld meter | Same-side NIR spectrometer. Made for sugar, dry matter, colour. Not for hidden rot. | Exact price not found. A simpler sister model was quoted at about USD 3,500. |
| Atago, Sunforest, Kubota handheld NIR meters | Handheld meters | Sugar and ripeness. | Not checked in this search. |
| Shimadzu UV-3600i Plus | Laboratory spectrophotometer | Needs cut samples. Not a sorting tool. | Rs 25 to 40 lakh. |

Three things to take from the market:

1. **Industry already trusts transmission for hidden rot.** We are not betting on an unknown idea.
2. **One major maker gets by with a ratio of two light bands.** That is strong support for
   "a few well-chosen channels can be enough".
3. **There is nothing in between.** Either a sorting line for big packhouses or a handheld sugar meter.
   A fruit vendor, small cold store, juice maker or farmer group in India has no cheap tool for hidden rot.

## 6. The gap, and what ECO-LOOP can claim

| Question | What we found |
|---|---|
| Has transmission been used for hidden apple rot? | Yes, many times, with spectrometers. |
| Has a cheap portable transmission device been built? | Yes, at least one, with a chip spectrometer. |
| Has the AS7265x been used on fruit? | Yes, for sugar, ripeness, shelf life, by same-side measurement. |
| Has the AS7265x (or a similar 18-channel chip) been tested in **transmission** for **hidden rot**? | **We did not find any.** |
| Is there an open dataset of low-cost sensor readings + cut-open truth for Indian-market apples? | We did not find any. |

So the honest research question is:

> **How much of what a packhouse machine does for hidden rot can an 18-channel sensor costing a few thousand rupees do?
> Which rot size can it detect, and which can it not?**

This is a good question because either answer is useful. If it works, it is a very cheap tool.
If it only works for large rot, that limit is still a result people can cite.

## 7. Why this matters in India

- Mouldy core and core rot are a known problem in Himachal apples. A survey in Shimla district reported
  around 16 to 18% incidence, with *Alternaria alternata* as the usual cause
  ([thesis, UHF Nauni](https://krishikosh.egranth.ac.in/items/b3ef44e4-7be7-43a5-825c-5e09d4220814);
  [*Indian Phytopathology* paper](https://epubs.icar.org.in/index.php/IPPJ/article/view/68954)).
- It is worst in Delicious-type apples with an open calyx tube. These are the most common apples in Indian markets.
- The fruit looks fine from outside. This is exactly the case where a camera or the eye fails.

This gives the paper a concrete Indian problem, not just a general "food waste" statement.

## 8. Papers from our current reference list that stay useful

Nicolaï et al. 2007 (the main review of NIR for fruit), Walsh et al. 2020 (thirty years of commercial NIR),
Tran and Fukuzawa 2020, Mohammed et al. 2023, Zhang et al. 2021 (watercore by full transmission),
Kavuncuoğlu et al. 2023 (pH is hard), Rynk et al. 1992 (compost).

One loose end: the Master Document says "a 2022 paper already did AS7265x + Random Forest on fruit binary
classification". We could not identify which paper that is. Someone should find it.

## 9. Reading list for the team (in this order)

1. Nicolaï et al., 2007: the big review. Read sections on measurement modes.
2. Khatiwada et al., 2016: comparison of interactance and transmission for browning.
3. Clark et al., 2003: how lamp and detector positions were chosen.
4. Liu et al., 2024: the handheld moldy-core device. Our nearest neighbour.
5. Tran and Fukuzawa, 2020: how they handled the AS7265x.
6. The carbide-ripening preprint: an ESP32 + AS7265x chamber build.
