/*
  ═══════════════════════════════════════════════════════════════════════════
  cpx_fog_logger.ino  —  Circuit Playground Express
  Parkinson's gait-freeze ON-BOARD DATA LOGGER  ·  ENGF0031 Scenario 2
  ═══════════════════════════════════════════════════════════════════════════

  PURPOSE — untethered ground-truth capture for the Accuracy Worksheet.
  No ESP32, no laptop during the walk. The wearer follows an OLED-guided
  still/walk/freeze script; the board logs raw accelerometer samples + the
  scripted phase label to its 2 MB SPI flash. Afterwards, plug the CPX into
  the laptop and a one-line command dumps the whole capture as CSV, which
  drops straight into analyze_worksheet.py for real sensitivity/specificity.

  WHY THIS SHAPE
    · 64 Hz accel, read through the SAME CircuitPlayground.motionX/Y/Z() →
      milli-g path the detector uses, so the units/axes match the pipeline.
    · The phase column is an INCREMENTING SEGMENT INDEX (0,1,2,…) — exactly
      what analyze_worksheet.py expects (it maps idx→name via --labels and
      scores freezes via --freeze-phases). NOT a fixed still/walk/freeze code.
    · Logging is RAW (no FAT filesystem, no TinyUSB) → tiny RAM footprint on
      the SAMD21's 32 KB. 7 bytes/sample (int16 ax,ay,az + uint8 phase); a
      4.3-min run ≈ 116 KB, well inside the reserved region.

  ── OPERATE ──────────────────────────────────────────────────────────────
    1. Power from a battery/USB pack. Boots to READY (switch in OFF position).
    2. Stand still, then FLIP THE SLIDE SWITCH to start. The board erases the
       log (~3-8 s, shown on OLED), counts down 5 s, then walks you through:
         STILL 20s · WALK 40s · FREEZE 20s · WALK 40s · FREEZE 20s ·
         WALK 40s · FREEZE 20s · WALK 40s · STILL 20s          (≈ 4.3 min)
       Follow the BIG label on the OLED; it auto-labels each segment.
    3. At the end it shows "DONE — N samples". Flip the switch back to OFF.
       (Flipping OFF early stops and saves what you have.)
    4. Take the device off, plug the CPX into the laptop, and run the dump:
         python cpx_dump.py walk1            (saves accuracy_figs/captures/walk1.csv)
       then analyse (the dump prints the exact flags too):
         python analyze_worksheet.py accuracy_figs/captures/walk1.csv \
            --labels still,walk,freeze,walk,freeze,walk,freeze,walk,still \
            --freeze-phases 2,4,6 --threshold 1.815

  ── HARDWARE ──
    Board : Circuit Playground Express (onboard LIS3DH accel + 2 MB SPI flash)
    OLED  : 128x64 on the EXTERNAL I2C bus (SDA=A5, SCL=A4, 3.3 V / GND)
    Slide switch : start (one side) / stop (other side)
    NeoPixels    : blue=erasing · yellow=get-ready · teal=STILL · green=WALK ·
                   red=FREEZE · white=DONE
  ═══════════════════════════════════════════════════════════════════════════
*/

#include <Adafruit_CircuitPlayground.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SPIFlash.h>
#include <math.h>

// ── OLED is OPTIONAL and OFF by default ──────────────────────────────────────
//   The external OLED shares the CPX's I2C bus with the ONBOARD accelerometer.
//   Flaky/half-seated OLED wiring stalls that bus, and the SAMD Wire driver has
//   no timeout — so a sensor read inside the 64 Hz loop can block forever and
//   freeze a recording (symptom: NeoPixels stuck on one colour, no USB reply,
//   board won't even auto-reset). Running NeoPixel-only removes that failure
//   entirely; the ring colours + the HTML downloader's status display fully
//   replace the screen. Set to 1 ONLY once you trust the OLED wiring.
#define USE_OLED 0

// ── OLED driver select (same panel as cpx_fog_standalone) ───────────────────
#define OLED_SH1106 1            // 1 = SH1106 (1.3"),  0 = SSD1306 (0.96")
#if OLED_SH1106
  #include <Adafruit_SH110X.h>
  #define OLED_WHITE SH110X_WHITE
  #define OLED_BLACK SH110X_BLACK
#else
  #include <Adafruit_SSD1306.h>
  #define OLED_WHITE SSD1306_WHITE
  #define OLED_BLACK SSD1306_BLACK
#endif

// ── sampling (must match the detector / pipeline) ──
#define FS            64
const uint32_t SAMPLE_US = 1000000UL / FS;
const float    MS2_TO_MG = 1000.0f / 9.80665f;     // m/s² → milli-g (wire units)

// ── the scripted protocol (auto-labelled). Segment index = the phase column.
//    Keep --labels / --freeze-phases below in sync if you edit this. ──
struct Phase { const char *name; uint16_t secs; bool isFreeze; };
const Phase SCHED[] = {
  { "STILL",  20, false },   // 0
  { "WALK",   40, false },   // 1
  { "FREEZE", 20, true  },   // 2
  { "WALK",   40, false },   // 3
  { "FREEZE", 20, true  },   // 4
  { "WALK",   40, false },   // 5
  { "FREEZE", 20, true  },   // 6
  { "WALK",   40, false },   // 7
  { "STILL",  20, false },   // 8
};
const uint8_t N_SEG = sizeof(SCHED) / sizeof(SCHED[0]);

// ── SPI flash raw log layout ──
// Sector 0 (4 KB) = metadata; data starts at 4096. We pre-erase ERASE_BLOCKS
// 64 KB blocks up front (covers meta + data) so there are NO erases — and thus
// no timing gaps — during sampling.
Adafruit_FlashTransport_SPI flashTransport(EXTERNAL_FLASH_USE_CS, EXTERNAL_FLASH_USE_SPI);
Adafruit_SPIFlash flash(&flashTransport);
bool flashOK = false;

const uint32_t MAGIC        = 0xCADCE100UL;
const uint32_t META_ADDR    = 0;
const uint32_t DATA_ADDR    = 4096;
const uint8_t  REC_BYTES    = 7;                    // int16 x,y,z + uint8 phase
const uint8_t  ERASE_BLOCKS = 4;                    // 4 × 64 KB = 256 KB
const uint32_t DATA_LIMIT   = (uint32_t)ERASE_BLOCKS * 65536UL;
const uint32_t MAX_SAMPLES  = (DATA_LIMIT - DATA_ADDR) / REC_BYTES;  // ≈ 36 864

// ── run state ──
enum Mode { IDLE, ERASING, COUNTDOWN, RUNNING, DONE, FLASH_ERR };
Mode      mode = IDLE;
uint32_t  writeAddr   = DATA_ADDR;
uint32_t  sampleCount = 0;
uint8_t   segIdx      = 0;
uint32_t  segEndMs    = 0;
uint32_t  cdEndMs     = 0;
uint32_t  nextSampleUs = 0;
bool      prevSw      = false;
bool      full        = false;

uint8_t   pageBuf[REC_BYTES * 36];                  // 252 B = 36 records
uint16_t  pageLen = 0;

// last-stored session (for the READY/STATUS screens)
uint32_t  storedSamples = 0;

// serial command line (used while idle / done)
char      cmd[16];
uint8_t   cmdLen = 0;

// ── OLED on the external I2C bus (SDA=A5, SCL=A4) ──
#define OLED_ADDR 0x3C
#if OLED_SH1106
Adafruit_SH1106G oled(128, 64, &Wire, -1);
#else
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
#endif
bool      oledOK = false;
uint8_t   oledAddrFound = 0;
uint32_t  lastOledMs = 0;

// ───────────────────────────────────────────────────────────────────────────
//  helpers
// ───────────────────────────────────────────────────────────────────────────
void setPixels(uint8_t r, uint8_t g, uint8_t b) {
  for (uint8_t i = 0; i < 10; i++) CircuitPlayground.setPixelColor(i, r, g, b);
}

bool oledBegin(uint8_t addr) {
#if OLED_SH1106
  return oled.begin(addr, true);
#else
  return oled.begin(SSD1306_SWITCHCAPVCC, addr, true, false);
#endif
}

void printLower(const char *s) {
  for (const char *p = s; *p; p++) Serial.print((char)tolower(*p));
}

// Read the stored sample count from the metadata sector (0 if none / invalid).
uint32_t readStoredCount() {
  if (!flashOK) return 0;
  uint8_t m[12];
  flash.readBuffer(META_ADDR, m, 12);
  uint32_t magic = (uint32_t)m[0] | ((uint32_t)m[1] << 8) | ((uint32_t)m[2] << 16) | ((uint32_t)m[3] << 24);
  if (magic != MAGIC) return 0;
  return (uint32_t)m[4] | ((uint32_t)m[5] << 8) | ((uint32_t)m[6] << 16) | ((uint32_t)m[7] << 24);
}

void flushPage() {
  if (pageLen == 0 || !flashOK) { pageLen = 0; return; }
  flash.writeBuffer(writeAddr, pageBuf, pageLen);
  writeAddr += pageLen;
  pageLen = 0;
}

void pushRecord(int16_t x, int16_t y, int16_t z, uint8_t ph) {
  if (full) return;
  if (writeAddr + pageLen + REC_BYTES > DATA_LIMIT) { full = true; return; }
  pageBuf[pageLen++] = x & 0xFF;  pageBuf[pageLen++] = (x >> 8) & 0xFF;
  pageBuf[pageLen++] = y & 0xFF;  pageBuf[pageLen++] = (y >> 8) & 0xFF;
  pageBuf[pageLen++] = z & 0xFF;  pageBuf[pageLen++] = (z >> 8) & 0xFF;
  pageBuf[pageLen++] = ph;
  sampleCount++;
  if (sampleCount >= MAX_SAMPLES) full = true;
  if (pageLen >= sizeof(pageBuf)) flushPage();
}

void writeMeta() {
  if (!flashOK) return;
  uint8_t m[12] = {0};
  m[0] = MAGIC & 0xFF;  m[1] = (MAGIC >> 8) & 0xFF;  m[2] = (MAGIC >> 16) & 0xFF;  m[3] = (MAGIC >> 24) & 0xFF;
  m[4] = sampleCount & 0xFF; m[5] = (sampleCount >> 8) & 0xFF;
  m[6] = (sampleCount >> 16) & 0xFF; m[7] = (sampleCount >> 24) & 0xFF;
  m[8] = FS & 0xFF; m[9] = (FS >> 8) & 0xFF;
  m[10] = N_SEG; m[11] = 1;
  flash.writeBuffer(META_ADDR, m, 12);   // sector 0 was erased at start → write-once
  flash.waitUntilReady();
}

void finalize() {
  flushPage();
  writeMeta();
  storedSamples = sampleCount;
}

// ───────────────────────────────────────────────────────────────────────────
//  OLED screens
// ───────────────────────────────────────────────────────────────────────────
void oledHeader(const char *right) {
  oled.setTextSize(1); oled.setTextColor(OLED_WHITE);
  oled.setCursor(0, 0);  oled.print(F("CADENCE log"));
  if (right) { oled.setCursor(92, 0); oled.print(right); }
  oled.drawLine(0, 10, 127, 10, OLED_WHITE);
}

void drawReady() {
  if (!oledOK) return;
  oled.clearDisplay();
  oledHeader("READY");
  oled.setTextSize(1);
  oled.setCursor(0, 16); oled.print(F("Flip switch to START"));
  oled.setCursor(0, 30); oled.print(F("Stored: ")); oled.print(storedSamples); oled.print(F(" smp"));
  oled.setCursor(0, 44); oled.print(F("USB: run cpx_dump.py"));
  oled.setCursor(0, 54); oled.print(F("to pull the CSV off"));
  oled.display();
}

void drawErasing(uint8_t done, uint8_t total) {
  if (!oledOK) return;
  oled.clearDisplay();
  oledHeader(nullptr);
  oled.setTextSize(2); oled.setCursor(2, 22); oled.print(F("ERASING"));
  oled.setTextSize(1); oled.setCursor(2, 48);
  oled.print(F("block ")); oled.print(done); oled.print('/'); oled.print(total);
  oled.display();
}

void drawCountdown(int secsLeft) {
  if (!oledOK) return;
  oled.clearDisplay();
  oledHeader(nullptr);
  oled.setTextSize(1); oled.setCursor(2, 14); oled.print(F("GET READY"));
  oled.setTextSize(4); oled.setCursor(54, 24); oled.print(secsLeft);
  oled.setTextSize(1); oled.setCursor(2, 56);
  oled.print(F("first: ")); oled.print(SCHED[0].name);
  oled.display();
}

void drawRunning(int secsLeft) {
  if (!oledOK) return;
  oled.clearDisplay();
  oledHeader(nullptr);
  bool freeze = SCHED[segIdx].isFreeze;
  if (freeze) { oled.fillRect(0, 13, 128, 22, OLED_WHITE); oled.setTextColor(OLED_BLACK); }
  else        { oled.setTextColor(OLED_WHITE); }
  oled.setTextSize(3); oled.setCursor(2, 15); oled.print(SCHED[segIdx].name);
  oled.setTextColor(OLED_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 40); oled.print(secsLeft); oled.print(F("s  seg "));
  oled.print(segIdx); oled.print('/'); oled.print(N_SEG - 1);
  // anticipate the next phase in the last 3 s
  if (secsLeft <= 3 && segIdx + 1 < N_SEG) {
    oled.setCursor(0, 50); oled.print(F("NEXT: ")); oled.print(SCHED[segIdx + 1].name);
  } else {
    oled.setCursor(0, 50); oled.print(F("smp ")); oled.print(sampleCount);
  }
  oled.display();
}

void drawDone() {
  if (!oledOK) return;
  oled.clearDisplay();
  oledHeader("DONE");
  oled.setTextSize(2); oled.setCursor(2, 18); oled.print(F("DONE"));
  oled.setTextSize(1);
  oled.setCursor(0, 40); oled.print(sampleCount); oled.print(F(" samples"));
  oled.setCursor(0, 52); oled.print(F("switch OFF, then dump"));
  oled.display();
}

void drawFlashErr() {
  if (!oledOK) return;
  oled.clearDisplay();
  oledHeader(nullptr);
  oled.setTextSize(2); oled.setCursor(2, 20); oled.print(F("FLASH"));
  oled.setCursor(2, 40); oled.print(F("FAIL"));
  oled.display();
}

// ───────────────────────────────────────────────────────────────────────────
//  serial dump (transfer mode) — prints the whole capture as CSV
// ───────────────────────────────────────────────────────────────────────────
void doStatus() {
  uint32_t c = readStoredCount();
  Serial.print(F("#STATUS samples=")); Serial.print(c);
  Serial.print(F(" rate_hz=")); Serial.print(FS);
  Serial.print(F(" flash_kb=")); Serial.print(flashOK ? flash.size() / 1024 : 0);
  Serial.print(F(" max_samples=")); Serial.println(MAX_SAMPLES);
}

void doDump() {
  uint32_t cnt = readStoredCount();
  if (!flashOK) { Serial.println(F("#ERR flash not ready")); return; }
  if (cnt == 0) { Serial.println(F("#ERR no recording stored (record one first)")); return; }

  Serial.println(F("# CADENCE on-board capture (CPX SPI-flash logger)"));
  Serial.print(F("# samples=")); Serial.print(cnt);
  Serial.print(F("  rate_hz=")); Serial.print(FS);
  Serial.print(F("  duration_s=")); Serial.println(cnt / (float)FS, 2);
  Serial.print(F("# protocol: "));
  for (uint8_t i = 0; i < N_SEG; i++) { Serial.print(SCHED[i].name); if (i < N_SEG - 1) Serial.print('>'); }
  Serial.println();
  Serial.print(F("# analyze: --labels "));
  for (uint8_t i = 0; i < N_SEG; i++) { printLower(SCHED[i].name); if (i < N_SEG - 1) Serial.print(','); }
  Serial.print(F(" --freeze-phases "));
  bool first = true;
  for (uint8_t i = 0; i < N_SEG; i++) if (SCHED[i].isFreeze) { if (!first) Serial.print(','); Serial.print(i); first = false; }
  Serial.println();
  Serial.println(F("idx,t_s,ax_mg,ay_mg,az_mg,phase"));

  uint8_t buf[REC_BYTES * 36];
  uint32_t addr = DATA_ADDR;
  uint32_t i = 0;
  while (i < cnt) {
    uint32_t n = (cnt - i < 36) ? (cnt - i) : 36;
    flash.readBuffer(addr, buf, n * REC_BYTES);
    addr += n * REC_BYTES;
    for (uint32_t k = 0; k < n; k++) {
      const uint8_t *r = &buf[k * REC_BYTES];
      int16_t x = (int16_t)((uint16_t)r[0] | ((uint16_t)r[1] << 8));
      int16_t y = (int16_t)((uint16_t)r[2] | ((uint16_t)r[3] << 8));
      int16_t z = (int16_t)((uint16_t)r[4] | ((uint16_t)r[5] << 8));
      uint8_t ph = r[6];
      uint32_t idx = i + k;
      Serial.print(idx);              Serial.print(',');
      Serial.print(idx / (float)FS, 4); Serial.print(',');
      Serial.print(x);                Serial.print(',');
      Serial.print(y);                Serial.print(',');
      Serial.print(z);                Serial.print(',');
      Serial.println(ph);
    }
    i += n;
  }
  Serial.print(F("#END,")); Serial.println(cnt);
}

void handleCmd() {
  if (cmdLen == 0) return;
  if      (!strcmp(cmd, "DUMP"))   doDump();
  else if (!strcmp(cmd, "STATUS")) doStatus();
  else if (!strcmp(cmd, "ERASE"))  { if (flashOK) { flash.eraseSector(0); storedSamples = 0; } Serial.println(F("#OK erased")); }
  else                             Serial.println(F("#? cmds: DUMP STATUS ERASE"));
}

void pollSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { cmd[cmdLen] = 0; handleCmd(); cmdLen = 0; }
    else if (cmdLen < sizeof(cmd) - 1) { cmd[cmdLen++] = toupper(c); }
  }
}

// ───────────────────────────────────────────────────────────────────────────
void startErase() {
  setPixels(0, 0, 40);
  for (uint8_t blk = 0; blk < ERASE_BLOCKS; blk++) {
    drawErasing(blk, ERASE_BLOCKS);
    if (flashOK) { flash.eraseBlock(blk); flash.waitUntilReady(); }
  }
  writeAddr = DATA_ADDR; sampleCount = 0; pageLen = 0; segIdx = 0; full = false;
}

// ───────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);                 // USB CDC — used for the dump
  CircuitPlayground.begin();
  CircuitPlayground.setAccelRange(LIS3DH_RANGE_8_G);  // ±8 g: ankle heel-strike exceeds ±2 g; record the un-clipped signal the detector expects
  CircuitPlayground.setBrightness(35);
  setPixels(0, 0, 40);

  // I2C bus (shared: onboard accel + the optional external OLED on SDA=A5/SCL=A4)
  Wire.begin();
  Wire.setClock(400000);
#if defined(WIRE_HAS_TIMEOUT)
  // never let a glitchy bus block a sensor read forever; auto-reset on timeout
  Wire.setWireTimeout(3000 /* us */, true);
#endif
#if USE_OLED
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0 && (a == 0x3C || a == 0x3D)) oledAddrFound = a;
  }
  if (oledAddrFound) oledOK = oledBegin(oledAddrFound);
#endif

  // SPI flash
  flashOK = flash.begin();
  if (flashOK) storedSamples = readStoredCount();

  prevSw = CircuitPlayground.slideSwitch();
  mode = flashOK ? IDLE : FLASH_ERR;
  if (mode == IDLE) drawReady(); else drawFlashErr();
  nextSampleUs = micros();
}

void loop() {
  bool sw = CircuitPlayground.slideSwitch();
  bool swChanged = (sw != prevSw);
  prevSw = sw;

  switch (mode) {

    case FLASH_ERR:
      setPixels(40, 0, 0);
      pollSerial();                       // still answer STATUS over USB
      break;

    case IDLE:
      setPixels(0, 0, 6);
      pollSerial();                       // DUMP / STATUS / ERASE over USB
      if (swChanged) {                    // any flip from the rest position = start
        mode = ERASING;
      } else if (millis() - lastOledMs > 800) { lastOledMs = millis(); drawReady(); }
      break;

    case ERASING:
      startErase();
      cdEndMs = millis() + 5000;
      mode = COUNTDOWN;
      break;

    case COUNTDOWN: {
      if (swChanged) { mode = IDLE; drawReady(); break; }   // aborted
      int secsLeft = (int)((cdEndMs - millis() + 999) / 1000);
      if ((int32_t)(millis() - cdEndMs) >= 0) {
        segIdx = 0;
        segEndMs = millis() + (uint32_t)SCHED[0].secs * 1000UL;
        nextSampleUs = micros();
        mode = RUNNING;
      } else {
        setPixels(40, 30, 0);
        if (millis() - lastOledMs > 200) { lastOledMs = millis(); drawCountdown(secsLeft); }
      }
      break;
    }

    case RUNNING: {
      if (swChanged) { finalize(); mode = DONE; drawDone(); break; }  // manual stop

      // steady 64 Hz sampling (catch up if a flash write nudged us late).
      // Cap the catch-up: a slow/stalled read can never spin this loop forever,
      // so segment-advance, USB and pixels always keep making forward progress.
      uint32_t now = micros();
      uint8_t  caught = 0;
      while ((int32_t)(now - nextSampleUs) >= 0 && caught < 8) {
        nextSampleUs += SAMPLE_US;
        float ax = CircuitPlayground.motionX();
        float ay = CircuitPlayground.motionY();
        float az = CircuitPlayground.motionZ();
        pushRecord((int16_t)(ax * MS2_TO_MG), (int16_t)(ay * MS2_TO_MG),
                   (int16_t)(az * MS2_TO_MG), segIdx);
        caught++;
        now = micros();
      }
      // if still far behind after the cap, resync (drop a few samples, never spiral)
      if ((int32_t)(now - nextSampleUs) >= (int32_t)SAMPLE_US) nextSampleUs = now;

      // advance the scripted phase
      if ((int32_t)(millis() - segEndMs) >= 0) {
        segIdx++;
        if (segIdx >= N_SEG || full) { finalize(); mode = DONE; drawDone(); break; }
        segEndMs = millis() + (uint32_t)SCHED[segIdx].secs * 1000UL;
      }

      // NeoPixel + OLED feedback (OLED ~4 Hz so it never starves sampling)
      bool freeze = SCHED[segIdx].isFreeze;
      if      (freeze)                          setPixels(40, 0, 0);
      else if (strcmp(SCHED[segIdx].name, "STILL") == 0) setPixels(0, 6, 8);
      else                                      setPixels(0, 30, 0);
      if (millis() - lastOledMs > 250) {
        lastOledMs = millis();
        int secsLeft = (int)((segEndMs - millis() + 999) / 1000);
        drawRunning(secsLeft);
      }
      break;
    }

    case DONE:
      setPixels(30, 30, 30);
      pollSerial();                       // allow dump straight away if plugged in
      if (swChanged) { mode = IDLE; drawReady(); }   // flip back to re-arm
      break;
  }
}
