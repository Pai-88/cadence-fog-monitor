/*
  ═══════════════════════════════════════════════════════════════════════════
  cpx_fog_freelog.ino  —  Circuit Playground Express
  Parkinson's gait-freeze FREE-FORM RECORDER + ON-DEVICE INFERENCE
  ENGF0031 Scenario 2
  ═══════════════════════════════════════════════════════════════════════════

  PURPOSE — untethered FREE-FORM capture with LIVE on-board classification.
  Unlike cpx_fog_logger (which scripts the wearer through STILL/WALK/FREEZE
  segments and stores the scripted segment index), this build NEVER tells the
  wearer what to do. The wearer moves freely; the board only OBSERVES. It runs
  the exact classical Freeze-Index detector from cpx_fog_standalone on-device,
  infers STILL / WALKING / FREEZE once per 2 s hop, and:
      · logs every 64 Hz accelerometer sample to its 2 MB SPI flash, and
      · stores the DEVICE-INFERRED STATE CODE in the per-sample phase byte
        (0 = still, 1 = walk, 2 = freeze) — so the recording carries the
        board's OWN classification, not a scripted label.
  Afterwards, plug the CPX into the laptop and cpx_dump.py pulls the whole
  capture as CSV (unchanged parser) for offline analysis.

  WHY THIS SHAPE
    · 64 Hz accel, read through the SAME CircuitPlayground.motionX/Y/Z() →
      milli-g path the detector uses, so units/axes match the pipeline. The
      inference itself runs on the float m/s² values exactly as the standalone.
    · The phase column is the ON-DEVICE INFERRED STATE (0/1/2), NOT a scripted
      segment index. cpx_dump.py is parser-compatible: same data columns and
      "#END,<count>" trailer; only the header comment lines changed.
    · Logging is RAW (no FAT filesystem, no TinyUSB) → tiny RAM footprint on
      the SAMD21's 32 KB. 7 bytes/sample (int16 ax,ay,az + uint8 phase).

  ── OPERATE ──────────────────────────────────────────────────────────────
    1. Power from a battery/USB pack. Boots to READY (switch in OFF position).
    2. FLIP THE SLIDE SWITCH to start. The board erases the log (~3-8 s, blue),
       then runs a brief still-floor CALIBRATION over the first window (~4 s)
       — sensor baseline only, NOT a movement instruction; just hold steady —
       then RUNNING. While RUNNING the OLED/ring report what the board THINKS
       is happening (STILL / WALKING / FREEZE). It is feedback, never a cue.
    3. Flip the switch back to OFF to stop and save. It shows "DONE — N smp".
    4. Plug the CPX into the laptop and run the dump:
         python cpx_dump.py walk1   (saves accuracy_figs/captures/walk1.csv)
       The phase column is the device-inferred state (0=still 1=walk 2=freeze).

  ── HARDWARE ──
    Board : Circuit Playground Express (onboard LIS3DH accel + 2 MB SPI flash)
    OLED  : 128x64 on the EXTERNAL I2C bus (SDA=A5, SCL=A4, 3.3 V / GND) — optional
    Slide switch : start (one side) / stop (other side)
    Button A     : hold still + press to manually recalibrate the still-floor
    NeoPixels    : blue=erasing/calibrating · teal=STILL · green=WALKING ·
                   red=FREEZE · white=DONE
    NOTE: the vibration motor is NEVER driven here — this is a monitor/recorder,
          the cue is disabled.
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
//   freeze a recording. NeoPixel-only removes that failure entirely; set to 1
//   ONLY once you trust the OLED wiring.
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
#define WINDOW        256          // analysis window, samples (4.0 s)
#define HOP           128          // decision cadence, samples (2.0 s)
const uint32_t SAMPLE_US = 1000000UL / FS;
const float    MS2_TO_MG = 1000.0f / 9.80665f;     // m/s² → milli-g (wire units)

// ╔═══════════════════════════════════════════════════════════════════════╗
// ║  ON-DEVICE INFERENCE — ported verbatim from cpx_fog_standalone.ino     ║
// ║  EXPORTED FROM COLAB (export_for_device, tuned at 64 Hz)              ║
// ╚═══════════════════════════════════════════════════════════════════════╝
const float FI_THRESHOLD = 1.815f;

// Butterworth band-pass, order 2 → two second-order sections each.
// Section coefficients are {b0, b1, b2, a1, a2}  (a0 normalised to 1).
struct Sos { float b0, b1, b2, a1, a2; };

const uint8_t N_SOS = 2;
const Sos FREEZE_SOS[N_SOS] = {            // 3–8 Hz
  { 0.04427971f,  0.08855942f, 0.04427971f, -1.24067070f, 0.62897179f },
  { 1.00000000f, -2.00000000f, 1.00000000f, -1.69751871f, 0.79495806f },
};
const Sos LOCO_SOS[N_SOS] = {              // 0.5–3 Hz
  { 0.01278734f,  0.02557469f, 0.01278734f, -1.69059402f, 0.75072617f },
  { 1.00000000f, -2.00000000f, 1.00000000f, -1.93848688f, 0.94143123f },
};
// ── end exported block ─────────────────────────────────────────────────────

// ── detection / debounce tuning (verbatim from standalone) ──
const uint8_t  CONFIRM_ON  = 2;       // consecutive freeze windows → commit FREEZE
const uint8_t  CONFIRM_OFF = 2;       // consecutive clear  windows → leave FREEZE

// ── movement-energy gate (rejects quiet standing; Bachlin 2010) ──
// freeze ⟺ FI > FI_THRESHOLD AND (Pf+Pl) > still_floor.  still_floor is in the
// same (m/s²)²·window units as the band powers; auto-calibrated at start over
// one still window (floor = STILL_MARGIN × resting band-power), clamped.
float          still_floor  = 50.0f;  // (re)set by calibration; sane mid-band default (±8 g)
const float    STILL_MARGIN = 4.0f;   // floor = margin × resting energy
// In real m/s²·band-power units a still baseline is ~1, while walking sits in the
// hundreds to tens-of-thousands. The clamp keeps a genuinely-still calibration in
// a sane window: ABOVE still-noise (~1) yet well BELOW any real-motion energy.
// If the wearer was clearly moving during calibration (rest > 200, see CALIBRATE)
// the floor is NOT derived from that bad window — it falls back to the 12.0f default.
const float    MIN_FLOOR    = 30.0f;  // floor stays ABOVE ±8 g still-noise (quantisation ~16)
const float    MAX_FLOOR    = 90.0f;  // … yet well BELOW real-motion energy (hundreds+)

// ── ring buffer of the most-recent magnitude samples (for the detector) ──
float    ring[WINDOW];
uint16_t ringHead   = 0;               // index of oldest sample when full
uint16_t filled     = 0;
uint16_t sinceHop   = 0;

// ── inference runtime state ──
enum GaitState { GS_STILL = 0, GS_WALK = 1, GS_FREEZE = 2 };  // == phase byte codes
float     last_fi     = 0.0f;
float     last_energy = 0.0f;          // Pf+Pl on the last window (gate input)
float     g_pf = 0.0f, g_pl = 0.0f;    // last band powers (filled by computeFI)
GaitState g_state    = GS_WALK;        // instantaneous class (per-hop)
GaitState committed  = GS_WALK;        // debounced state stored in the phase byte
bool      inFreeze   = false;          // debounced FREEZE latch
uint8_t   on_count   = 0, off_count = 0;
bool      haveDecision = false;        // false until the first window decision
uint16_t  freezeEpisodes = 0;          // confirmed freeze onsets (debounced)

// ── calibration state ──
bool      calibrating = false;         // still-floor calibration in progress
uint16_t  calib_left  = 0;             // samples left to collect while calibrating
bool      btnA_prev   = false;         // button-A edge detector

// ── SPI flash raw log layout ──
// Sector 0 (4 KB) = metadata; data starts at 4096. Pre-erase ERASE_BLOCKS
// 64 KB blocks up front so there are NO erases — and no timing gaps — while
// sampling.
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
enum Mode { IDLE, ERASING, CALIBRATE, RUNNING, DONE, FLASH_ERR };
Mode      mode = IDLE;
uint32_t  writeAddr    = DATA_ADDR;
uint32_t  sampleCount  = 0;
uint32_t  runStartMs   = 0;
uint32_t  nextSampleUs = 0;
bool      prevSw       = false;          // last DEBOUNCED switch level
bool      full         = false;

// ── slide-switch debounce ──
// A mechanical bounce must not prematurely stop a RUNNING capture (truncated
// file) or spuriously trigger the destructive ERASE from IDLE. Treat the switch
// as changed only once a new level has held stable for SW_DEBOUNCE_MS.
const uint32_t SW_DEBOUNCE_MS = 25;
bool      swCandidate  = false;          // most recent raw level seen
uint32_t  swCandidateMs = 0;             // when that raw level was first seen

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
//  inference helpers — ported verbatim from cpx_fog_standalone.ino
// ───────────────────────────────────────────────────────────────────────────

// Run a band through the SOS cascade (Direct-Form-II-Transposed) from zero
// initial state and return Σ y² over the window — i.e. the band power.
float runBand(const Sos *sos, const float *x, uint16_t n) {
  float z1[N_SOS] = {0}, z2[N_SOS] = {0};
  float sumsq = 0.0f;
  for (uint16_t i = 0; i < n; i++) {
    float in = x[i];
    for (uint8_t s = 0; s < N_SOS; s++) {
      float out = sos[s].b0 * in + z1[s];
      z1[s] = sos[s].b1 * in - sos[s].a1 * out + z2[s];
      z2[s] = sos[s].b2 * in - sos[s].a2 * out;
      in = out;
    }
    sumsq += in * in;
  }
  return sumsq;
}

float computeFI() {
  static float w[WINDOW];               // ordered, mean-removed window
  float mean = 0.0f;
  for (uint16_t i = 0; i < WINDOW; i++) {
    w[i] = ring[(ringHead + i) % WINDOW];   // head = oldest when buffer is full
    mean += w[i];
  }
  mean /= WINDOW;
  for (uint16_t i = 0; i < WINDOW; i++) w[i] -= mean;
  g_pf = runBand(FREEZE_SOS, w, WINDOW);   // 3–8 Hz band power
  g_pl = runBand(LOCO_SOS,   w, WINDOW);   // 0.5–3 Hz band power
  return g_pf / (g_pl + 1e-9f);
}

// ───────────────────────────────────────────────────────────────────────────
//  flash log helpers — kept verbatim from cpx_fog_logger.ino
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
  m[10] = 3; m[11] = 1;   // 3 inferred state codes (still/walk/freeze)
  flash.writeBuffer(META_ADDR, m, 12);   // sector 0 was erased at start → write-once
  flash.waitUntilReady();
}

void finalize() {
  flushPage();
  writeMeta();
  storedSamples = sampleCount;
}

// ───────────────────────────────────────────────────────────────────────────
//  OLED screens — feedback only, NEVER prescriptive
// ───────────────────────────────────────────────────────────────────────────
const char *stateWord(GaitState s) {
  return (s == GS_STILL) ? "STILL" : (s == GS_FREEZE) ? "FREEZE" : "WALKING";
}

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

// Sensor baseline calibration screen — NOT a movement instruction.
void drawCalibrating() {
  if (!oledOK) return;
  oled.clearDisplay();
  oledHeader(nullptr);
  oled.setTextSize(2); oled.setCursor(2, 18); oled.print(F("calibr."));
  oled.setTextSize(1); oled.setCursor(2, 44);
  oled.print(F("baseline... hold steady"));
  oled.display();
}

// RUNNING screen: REC indicator, elapsed mm:ss, sample count, and the big
// inferred-state word framed as what the device THINKS — never an instruction.
void drawRunning() {
  if (!oledOK) return;
  oled.clearDisplay();
  oledHeader(nullptr);

  // REC indicator + elapsed mm:ss
  uint32_t el = (millis() - runStartMs) / 1000UL;
  oled.setTextSize(1); oled.setTextColor(OLED_WHITE);
  oled.setCursor(0, 0); oled.print(F("REC "));
  oled.fillCircle(28, 3, 3, OLED_WHITE);
  oled.setCursor(92, 0);
  if (el / 60 < 10) oled.print('0');
  oled.print(el / 60); oled.print(':');
  if (el % 60 < 10) oled.print('0');
  oled.print(el % 60);

  // big inferred-state word (board's verdict). Invert block on FREEZE.
  bool alert = (committed == GS_FREEZE);
  if (alert) { oled.fillRect(0, 13, 128, 22, OLED_WHITE); oled.setTextColor(OLED_BLACK); }
  else       { oled.setTextColor(OLED_WHITE); }
  oled.setTextSize(3); oled.setCursor(2, 15); oled.print(stateWord(committed));
  oled.setTextColor(OLED_WHITE);

  // "thinks:" framing + sample count (not an instruction)
  oled.setTextSize(1);
  oled.setCursor(0, 40); oled.print(F("device thinks (FI ")); oled.print(haveDecision ? last_fi : 0.0f, 1); oled.print(')');
  oled.setCursor(0, 52); oled.print(F("smp ")); oled.print(sampleCount);
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
//  cpx_dump.py parser-compatible: same columns + "#END,<count>" trailer.
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

  Serial.println(F("# CADENCE free-form capture (CPX SPI-flash logger, on-device inference)"));
  Serial.print(F("# samples=")); Serial.print(cnt);
  Serial.print(F("  rate_hz=")); Serial.print(FS);
  Serial.print(F("  duration_s=")); Serial.println(cnt / (float)FS, 2);
  Serial.println(F("# phase = device-inferred state: 0=still  1=walk  2=freeze"));
  Serial.println(F("# analyze: --labels still,walk,freeze --freeze-phases 2"));
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
  writeAddr = DATA_ADDR; sampleCount = 0; pageLen = 0; full = false;
  storedSamples = 0;   // on-flash recording is wiped; don't show a stale count if aborted
}

// Reset the detector pipeline + start a fresh still-floor calibration window.
void beginCalibration() {
  ringHead = 0; filled = 0; sinceHop = 0;
  on_count = off_count = 0;
  inFreeze = false; haveDecision = false;
  g_state = GS_WALK; committed = GS_WALK;
  freezeEpisodes = 0;
  last_fi = 0.0f; last_energy = 0.0f;
  calibrating = true; calib_left = WINDOW;
}

// Mid-run recalibration: recompute still_floor from the ALREADY-WARM 256-sample
// window WITHOUT zeroing filled/ring, so detection resumes immediately (no ~8 s
// blind re-warm). Uses the same motion-rejection fallback as the boot path.
void recalibrateWarm() {
  if (filled < WINDOW) { beginCalibration(); return; }  // not warm yet → normal cal
  computeFI();                            // fills g_pf, g_pl on the warm window
  float rest = g_pf + g_pl;
  if (rest > 200.0f) {                    // clearly moving → don't trust this window
    still_floor = 50.0f;                  // fall back to the safe default
  } else {
    still_floor = STILL_MARGIN * rest;
    if (still_floor < MIN_FLOOR) still_floor = MIN_FLOOR;
    if (still_floor > MAX_FLOOR) still_floor = MAX_FLOOR;
  }
  Serial.print(F("recalibrated(warm): rest_energy=")); Serial.print(rest, 3);
  Serial.print(F("  -> still_floor=")); Serial.println(still_floor, 3);
  sinceHop = 0;                           // realign the hop boundary to the recal
}

// Map the per-hop instantaneous + debounced decision into the committed state
// that gets stored in the phase byte. FREEZE is debounced both ways (2 windows);
// STILL vs WALKING follow the gate immediately when not in a confirmed freeze.
void runDecision() {
  last_fi = computeFI();                 // fills g_pf, g_pl
  last_energy = g_pf + g_pl;             // movement energy (both bands)
  haveDecision = true;

  bool moving = last_energy > still_floor;            // gate: real motion?
  bool freezeNow = moving && (last_fi > FI_THRESHOLD);
  g_state = !moving ? GS_STILL : (freezeNow ? GS_FREEZE : GS_WALK);

  if (freezeNow) { on_count++;  off_count = 0; }      // STILL & WALK both clear
  else           { off_count++; on_count  = 0; }

  if (!inFreeze && on_count >= CONFIRM_ON) {
    inFreeze = true;  freezeEpisodes++;               // confirmed freeze onset
  } else if (inFreeze && off_count >= CONFIRM_OFF) {
    inFreeze = false;
  }

  // committed state stored in the phase byte: a confirmed freeze overrides;
  // otherwise the immediate gated still/walk class.
  committed = inFreeze ? GS_FREEZE : g_state;
}

void showRunningPixels() {
  switch (committed) {
    case GS_FREEZE: setPixels(40, 0, 0);  break;   // red
    case GS_STILL:  setPixels(0, 6, 8);   break;   // teal
    default:        setPixels(0, 30, 0);  break;   // green (walking)
  }
}

// Debounced slide-switch level. Reads the raw switch, requires a new level to
// hold for SW_DEBOUNCE_MS before accepting it, and returns the accepted level.
// Bounces shorter than the window are ignored entirely.
bool debouncedSwitch() {
  bool raw = CircuitPlayground.slideSwitch();
  if (raw != swCandidate) {              // raw moved → restart the stability timer
    swCandidate   = raw;
    swCandidateMs = millis();
  } else if (raw != prevSw && (millis() - swCandidateMs) >= SW_DEBOUNCE_MS) {
    prevSw = raw;                        // candidate held long enough → accept it
  }
  return prevSw;
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
  swCandidate = prevSw; swCandidateMs = millis();   // seed switch debounce
  mode = flashOK ? IDLE : FLASH_ERR;
  if (mode == IDLE) drawReady(); else drawFlashErr();
  nextSampleUs = micros();
}

void loop() {
  bool before    = prevSw;              // last accepted (debounced) level
  bool sw        = debouncedSwitch();   // updates prevSw once a level is stable
  bool swChanged = (sw != before);      // true only on a debounced transition

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
      beginCalibration();                 // start the still-floor baseline window
      drawCalibrating();
      nextSampleUs = micros();
      mode = CALIBRATE;
      break;

    case CALIBRATE: {
      if (swChanged) { mode = IDLE; drawReady(); break; }   // aborted before RUNNING

      setPixels(0, 0, 40);                // blue = calibrating (sensor baseline)

      // sample the still window at a steady 64 Hz; do NOT log these — this is
      // sensor baseline calibration, not part of the recording.
      uint32_t now = micros();
      uint8_t  caught = 0;
      while ((int32_t)(now - nextSampleUs) >= 0 && caught < 8) {
        nextSampleUs += SAMPLE_US;
        float ax = CircuitPlayground.motionX();
        float ay = CircuitPlayground.motionY();
        float az = CircuitPlayground.motionZ();
        float mag = sqrtf(ax * ax + ay * ay + az * az);
        ring[ringHead] = mag;
        ringHead = (ringHead + 1) % WINDOW;
        if (filled < WINDOW) filled++;
        if (calib_left > 0) calib_left--;
        caught++;
        now = micros();
      }
      if ((int32_t)(now - nextSampleUs) >= (int32_t)SAMPLE_US) nextSampleUs = now;

      // window full → derive still_floor, then go RUNNING
      if (calib_left == 0 && filled >= WINDOW) {
        computeFI();                       // fills g_pf, g_pl
        float rest = g_pf + g_pl;
        // Motion-rejection for free-form use: the wearer may be moving when they
        // flip the switch. A still baseline is ~1; walking is hundreds+. If the
        // window is clearly motion (rest > 200) don't commit a bad floor — fall
        // back to the safe default and carry on (never block/prompt the wearer).
        if (rest > 200.0f) {
          still_floor = 50.0f;             // default floor; ignore the moving window
        } else {
          still_floor = STILL_MARGIN * rest;
          if (still_floor < MIN_FLOOR) still_floor = MIN_FLOOR;
          if (still_floor > MAX_FLOOR) still_floor = MAX_FLOOR;
        }
        Serial.print(F("calibrated: rest_energy=")); Serial.print(rest, 3);
        Serial.print(F("  -> still_floor=")); Serial.println(still_floor, 3);

        // start a clean recording: keep the warmed ring buffer, reset hop timer
        sinceHop = 0;
        committed = GS_WALK;
        runStartMs = millis();
        nextSampleUs = micros();
        mode = RUNNING;
      }
      break;
    }

    case RUNNING: {
      if (swChanged) { finalize(); mode = DONE; drawDone(); break; }  // manual stop

      // manual recalibrate: hold still + press button A (like the standalone).
      // Mid-run we recompute the floor from the WARM ring in place — detection is
      // never blinded, so the capture keeps RUNNING without a re-warm gap.
      bool btnA = CircuitPlayground.leftButton();
      if (btnA && !btnA_prev) {
        recalibrateWarm();                 // keep filled/ring; refresh still_floor only
      }
      btnA_prev = btnA;

      // steady 64 Hz sampling (catch up if a flash write nudged us late), capped
      // so a slow read can never spin this loop forever. The per-sample loop ONLY
      // reads/logs/advances the ring + counts hops; the heavy per-hop inference
      // (computeFI ~10-18 ms soft-float) runs ONCE after the loop so it never
      // compounds with sample catch-up.
      uint32_t now = micros();
      uint8_t  caught = 0;
      bool     hopDue = false;
      while ((int32_t)(now - nextSampleUs) >= 0 && caught < 8) {
        nextSampleUs += SAMPLE_US;
        float ax = CircuitPlayground.motionX();
        float ay = CircuitPlayground.motionY();
        float az = CircuitPlayground.motionZ();

        // feed the detector ring buffer (same path as the standalone)
        float mag = sqrtf(ax * ax + ay * ay + az * az);
        ring[ringHead] = mag;
        ringHead = (ringHead + 1) % WINDOW;
        if (filled < WINDOW) filled++;
        sinceHop++;

        // log the sample with the CURRENT device-inferred (committed) state
        pushRecord((int16_t)(ax * MS2_TO_MG), (int16_t)(ay * MS2_TO_MG),
                   (int16_t)(az * MS2_TO_MG), (uint8_t)committed);

        // note a hop boundary; defer the inference until after the catch-up loop
        if (filled >= WINDOW && sinceHop >= HOP) hopDue = true;

        caught++;
        now = micros();
      }
      if ((int32_t)(now - nextSampleUs) >= (int32_t)SAMPLE_US) nextSampleUs = now;

      // once per hop (every 128 samples / 2 s) update the inferred state — run
      // exactly once here; the new verdict applies to subsequent samples, as before.
      if (hopDue) {
        sinceHop = 0;
        runDecision();
      }

      if (full) { finalize(); mode = DONE; drawDone(); break; }

      // LIVE FEEDBACK ONLY — what the device THINKS, never a cue
      showRunningPixels();
      if (millis() - lastOledMs > 250) {
        lastOledMs = millis();
        drawRunning();
      }
      break;
    }

    case DONE:
      setPixels(30, 30, 30);              // white
      pollSerial();                       // allow dump straight away if plugged in
      if (swChanged) { mode = IDLE; drawReady(); }   // flip back to re-arm
      break;
  }
}
