# ECO-LOOP

Low-cost, non-destructive apple rot detection (Fresh / Early Rot / Severe Rot)
with an AS7265x 18-channel Vis-NIR sensor and an ESP32.

**Start with [`docs/00_START_HERE.md`](docs/00_START_HERE.md).** It explains the current plan and links to everything else.

| Folder | What is in it |
|---|---|
| `docs/` | Project plan, science, literature survey, hardware setup, data protocol, roadmap |
| `firmware/ecoloop_scanner/` | ESP32 program that reads the sensor |
| `sensor/scan_logger.py` | Laptop program that records real scans to CSV |
| `iiser_processing/` | Cleaned pilot scans from the Shimadzu at IISER Pune |
| `data/`, `models/`, `api/`, `ecoloop_simulator/` | Earlier software pipeline built on synthetic data (for demos and software testing only) |
