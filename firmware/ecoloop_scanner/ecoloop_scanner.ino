/* ============================================================
   ECO-LOOP — ESP32 scanner firmware
   firmware/ecoloop_scanner/ecoloop_scanner.ino

   What this does
   --------------
   The ESP32 is a simple "bridge". It does NOT decide anything.
   The laptop sends short text commands over USB, the ESP32
   talks to the AS7265x sensor and sends the 18 readings back.
   All the logic (what to scan, saving to CSV) lives in
   sensor/scan_logger.py on the laptop.

   Wiring (ESP32 DevKit  <->  SparkFun AS7265x)
   --------------------------------------------
     3V3     -> 3V3   (NEVER 5V / VIN — it will kill the sensor)
     GND     -> GND
     GPIO 21 -> SDA
     GPIO 22 -> SCL
   Optional: GPIO 26 -> signal pin of a MOSFET/relay module
             that switches the 12 V lamp (transmission mode).

   Library needed (Arduino IDE -> Library Manager):
     "SparkFun Spectral Triad AS7265X"

   Serial: 115200 baud. Every command ends with a newline.

   Commands                      Reply
   ---------------------------   ---------------------------------
   PING                          OK,ECOLOOP_SCANNER,1
   INFO                          OK,INFO,<hw>,<fwMajor>.<fwPatch>.<fwBuild>
   CFG,<gain 0-3>,<cycles 1-255> OK,CFG,<gain>,<cycles>
        gain: 0=1x 1=3.7x 2=16x 3=64x
        integration time = cycles x 2.8 ms (x2 for all 18 channels)
   LED,<white>,<ir>,<uv>         OK,LED,<white>,<ir>,<uv>     (each 0 or 1)
   LAMP,<0|1>                    OK,LAMP,<0|1>
   READ                          DATA,<tempC>,<18 raw>,<18 calibrated>
   anything else                 ERR,<message>

   The 18 values are always in wavelength order:
   410,435,460,485,510,535,560,585,610,645,680,705,730,760,810,860,900,940
   ============================================================ */

#include <Wire.h>
#include "SparkFun_AS7265X.h"

const int PIN_SDA  = 21;
const int PIN_SCL  = 22;
const int PIN_LAMP = 26;      // optional lamp switch (MOSFET / relay module)

AS7265X sensor;
bool sensorOk = false;
uint8_t currentCycles = 49;

// ---------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  pinMode(PIN_LAMP, OUTPUT);
  digitalWrite(PIN_LAMP, LOW);

  Wire.begin(PIN_SDA, PIN_SCL);
  delay(200);

  sensorOk = sensor.begin();
  if (sensorOk) {
    sensor.disableIndicator();          // the blue status LED would leak light into the box
    allLedsOff();
    applyConfig(2, 49);                 // default: gain 16x, 49 cycles
    Serial.println("OK,ECOLOOP_SCANNER,1");
  } else {
    Serial.println("ERR,sensor not found - check wiring (3V3, GND, SDA=21, SCL=22)");
  }
}

// ---------------------------------------------------------------
void allLedsOff() {
  sensor.disableBulb(AS7265x_LED_WHITE);
  sensor.disableBulb(AS7265x_LED_IR);
  sensor.disableBulb(AS7265x_LED_UV);
}

void applyConfig(int gain, int cycles) {
  gain   = constrain(gain, 0, 3);
  cycles = constrain(cycles, 1, 255);
  sensor.setGain((uint8_t)gain);
  sensor.setIntegrationCycles((uint8_t)cycles);
  currentCycles = (uint8_t)cycles;
}

// Take one full 18-channel measurement and print it.
void doRead() {
  sensor.takeMeasurements();

  // The library's own timeout is a little short for long integration
  // times, so we wait here until the sensor really has fresh data.
  unsigned long extraWait = (unsigned long)currentCycles * 2.8 * 3 + 300;
  unsigned long t0 = millis();
  while (!sensor.dataAvailable() && (millis() - t0 < extraWait)) {
    delay(5);
  }

  uint16_t raw[18] = {
    sensor.getA(), sensor.getB(), sensor.getC(), sensor.getD(), sensor.getE(), sensor.getF(),
    sensor.getG(), sensor.getH(), sensor.getR(), sensor.getI(), sensor.getS(), sensor.getJ(),
    sensor.getT(), sensor.getU(), sensor.getV(), sensor.getW(), sensor.getK(), sensor.getL()
  };
  float cal[18] = {
    sensor.getCalibratedA(), sensor.getCalibratedB(), sensor.getCalibratedC(),
    sensor.getCalibratedD(), sensor.getCalibratedE(), sensor.getCalibratedF(),
    sensor.getCalibratedG(), sensor.getCalibratedH(), sensor.getCalibratedR(),
    sensor.getCalibratedI(), sensor.getCalibratedS(), sensor.getCalibratedJ(),
    sensor.getCalibratedT(), sensor.getCalibratedU(), sensor.getCalibratedV(),
    sensor.getCalibratedW(), sensor.getCalibratedK(), sensor.getCalibratedL()
  };

  Serial.print("DATA,");
  Serial.print(sensor.getTemperatureAverage(), 1);
  for (int i = 0; i < 18; i++) { Serial.print(","); Serial.print(raw[i]); }
  for (int i = 0; i < 18; i++) { Serial.print(","); Serial.print(cal[i], 3); }
  Serial.println();
}

// Read the n-th comma separated integer after the command word.
// "CFG,3,255" -> field 1 = 3, field 2 = 255. Returns -1 if missing.
int field(const String &line, int n) {
  int start = 0;
  for (int i = 0; i < n; i++) {
    start = line.indexOf(',', start);
    if (start < 0) return -1;
    start++;
  }
  int end = line.indexOf(',', start);
  String part = (end < 0) ? line.substring(start) : line.substring(start, end);
  part.trim();
  if (part.length() == 0) return -1;
  return part.toInt();
}

// ---------------------------------------------------------------
void handle(String line) {
  line.trim();
  if (line.length() == 0) return;
  line.toUpperCase();

  if (line == "PING") {
    Serial.println(sensorOk ? "OK,ECOLOOP_SCANNER,1" : "ERR,sensor not found");
    return;
  }
  if (line.startsWith("LAMP")) {
    int on = field(line, 1);
    digitalWrite(PIN_LAMP, on == 1 ? HIGH : LOW);
    Serial.print("OK,LAMP,"); Serial.println(on == 1 ? 1 : 0);
    return;
  }
  if (!sensorOk) {
    Serial.println("ERR,sensor not found - check wiring and press the EN/RESET button");
    return;
  }

  if (line == "INFO") {
    Serial.print("OK,INFO,");
    Serial.print(sensor.getHardwareVersion());       Serial.print(",");
    Serial.print(sensor.getMajorFirmwareVersion());  Serial.print(".");
    Serial.print(sensor.getPatchFirmwareVersion());  Serial.print(".");
    Serial.println(sensor.getBuildFirmwareVersion());
  } else if (line.startsWith("CFG")) {
    int gain = field(line, 1), cycles = field(line, 2);
    if (gain < 0 || cycles < 0) { Serial.println("ERR,use CFG,<gain 0-3>,<cycles 1-255>"); return; }
    applyConfig(gain, cycles);
    Serial.print("OK,CFG,"); Serial.print(constrain(gain, 0, 3));
    Serial.print(",");       Serial.println(constrain(cycles, 1, 255));
  } else if (line.startsWith("LED")) {
    int w = field(line, 1), ir = field(line, 2), uv = field(line, 3);
    if (w < 0 || ir < 0 || uv < 0) { Serial.println("ERR,use LED,<white>,<ir>,<uv>"); return; }
    if (w == 1)  sensor.enableBulb(AS7265x_LED_WHITE); else sensor.disableBulb(AS7265x_LED_WHITE);
    if (ir == 1) sensor.enableBulb(AS7265x_LED_IR);    else sensor.disableBulb(AS7265x_LED_IR);
    if (uv == 1) sensor.enableBulb(AS7265x_LED_UV);    else sensor.disableBulb(AS7265x_LED_UV);
    Serial.print("OK,LED,"); Serial.print(w == 1); Serial.print(",");
    Serial.print(ir == 1);   Serial.print(",");    Serial.println(uv == 1);
  } else if (line == "READ") {
    doRead();
  } else {
    Serial.println("ERR,unknown command");
  }
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handle(line);
  }
}
