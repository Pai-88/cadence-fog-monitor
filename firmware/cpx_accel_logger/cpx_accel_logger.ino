/*
  ═══════════════════════════════════════════════════════════════════════════
  cpx_accel_logger.ino  —  Circuit Playground Express
  ENGF0031 Accuracy Worksheet  ·  offline data-collection logger
  Cadence — Parkinson's wearable  ·  Scenario 2
  ═══════════════════════════════════════════════════════════════════════════

  PURPOSE
    A standalone logger for the *Accuracy Worksheet* (Tasks 2-3). It is NOT the
    live streamer — it records the onboard accelerometer to RAM while you run a
    test protocol, then dumps the whole capture as CSV over USB so you can paste
    it into a file and plot it (MATLAB, Appendix C). The board needs no ESP32 or
    laptop for this — just USB power and the Serial Monitor.

    SITE-AGNOSTIC. The logger just records raw 3-axis accel; what you do with
    the trace depends on where you strap it:
      · ANKLE  -> freeze-of-gait capture  (analyze_worksheet.py, Freeze Index)
      · WRIST  -> rest-tremor capture     (analyze_tremor.py, 4-6 Hz band power)
    Same firmware, same 64 Hz, same CSV. Pick the protocol for your site (see
    worksheet/wrist_capture_protocol.txt for the wrist run).

  WHY 64 Hz AND NOT THE WORKSHEET'S "every 200 ms"
    The features of interest are narrow-band tremor: freeze-of-gait shows up as
    3-8 Hz trembling (the "freeze band", Moore et al. 2008) and Parkinsonian
    rest tremor as 4-6 Hz (the wrist band). By Nyquist you must sample faster
    than 2x the highest frequency of interest, i.e. > 16 Hz, or that band is
    invisible / aliased. 200 ms logging = 5 Hz, which CANNOT see either band. We
    log at 64 Hz to match the rest of the pipeline (fog/config.py: SAMPLE_RATE =
    64) and the Daphnet dataset, so what you see here is what the detector sees.
    THIS IS A SENSOR PARAMETER DEFINED BY THE ANALYSIS — cite it in Task 5.

  RAM BUDGET  (this is itself a real prototype limitation — discuss in Task 4)
    The CPX (SAMD21) has only 32 KB of SRAM. We store 3 x int16 = 6 bytes per
    sample and DERIVE time from the sample index (fixed interval), so we don't
    waste 4 bytes/row on a timestamp like the worksheet's Appendix B does.
        MAX_SAMPLES (3000) x 6 bytes = 18 KB   ->   3000 / 64 Hz = 46.9 s
    So one capture is ~47 s. Run the protocol as TWO short captures (see the
    Task 1 protocol) rather than one long one. If you'd rather have one longer
    capture, you can: (a) raise MAX_SAMPLES until it stops compiling/running, or
    (b) store magnitude only (1 x int16) for 3x the duration at the cost of the
    per-axis plot.

  CONTROLS
    RIGHT button (B) : 1st press  -> START logging (phase 0).
                       each press -> mark a NEW phase boundary (phase++), so you
                                     can shade phases on the Task 3 plot. The
                                     NeoPixels light up to show the phase number.
    LEFT  button (A) : STOP logging and dump the CSV over USB.
    Buffer full      : logging auto-stops and the ring flashes red; press LEFT
                       to dump.
    After a dump     : press RIGHT to START A FRESH CAPTURE (e.g. Capture 2) —
                       no hardware reset needed. Copy Capture 1's CSV out of the
                       Serial Monitor FIRST; starting again clears the RAM buffer.

  HOW TO USE
    1. Upload over USB. Open the Serial Monitor at 115200 baud.
    2. Strap the board at the capture site: ANKLE for freeze-of-gait (matches
       the FoG detector / Daphnet), WRIST for rest tremor (like a watch face,
       forearm supported). Battery optional — USB is fine for a tethered run.
    3. Press RIGHT to start; press RIGHT again at each protocol phase change.
    4. Press LEFT to finish -> CSV prints between DATA START / DATA END.
    5. Select-all, copy, paste into a text editor, save as  capture1.csv .
    6. Repeat for capture 2. Plot with Appendix C (colour by the `phase` column).

  CSV COLUMNS
    idx, t_s, ax_mg, ay_mg, az_mg, phase
      idx   sample number (0..)            t_s   seconds = idx / 64
      a*_mg acceleration in milli-g        phase 0,1,2,... set by RIGHT button
  ═══════════════════════════════════════════════════════════════════════════
*/

#include <Adafruit_CircuitPlayground.h>

// ── Config ─────────────────────────────────────────────────
const uint32_t USB_BAUD     = 115200;
const uint16_t SAMPLE_HZ    = 64;                         // matches fog/config.py
const uint32_t SAMPLE_US    = 1000000UL / SAMPLE_HZ;      // 15625 µs
const uint16_t MAX_SAMPLES  = 3000;                       // 18 KB RAM -> ~46.9 s
const uint8_t  MAX_PHASES    = 12;                        // phase markers we keep
const uint16_t DEBOUNCE_MS   = 250;                       // button debounce
const float    MS2_TO_MG     = 1000.0 / 9.80665;          // m/s² → milli-g

// ── Capture buffers ────────────────────────────────────────
int16_t  ax_mg[MAX_SAMPLES];
int16_t  ay_mg[MAX_SAMPLES];
int16_t  az_mg[MAX_SAMPLES];
uint16_t n = 0;                                           // samples stored

uint16_t phase_start[MAX_PHASES];                         // sample idx of each phase
uint8_t  n_phases = 0;

// ── State ──────────────────────────────────────────────────
enum Mode { IDLE, LOGGING, DONE };
Mode     mode = IDLE;
uint32_t next_sample_us = 0;
uint32_t last_btn_ms    = 0;
bool     prev_left = false, prev_right = false;
bool     full = false;                                    // buffer-full latch

// ── NeoPixel helpers ───────────────────────────────────────
void ringClear() {
  for (uint8_t i = 0; i < 10; i++) CircuitPlayground.setPixelColor(i, 0, 0, 0);
}
void ringShowPhase(uint8_t p) {            // light up (p+1) green pixels
  ringClear();
  for (uint8_t i = 0; i <= p && i < 10; i++) CircuitPlayground.setPixelColor(i, 0, 50, 0);
}
void ringFlash(uint8_t r, uint8_t g, uint8_t b) {
  for (uint8_t i = 0; i < 10; i++) CircuitPlayground.setPixelColor(i, r, g, b);
}

// ── Edge-detected, debounced button reads ──────────────────
bool pressed(bool now, bool &prev) {
  bool edge = now && !prev;
  prev = now;
  if (edge && (millis() - last_btn_ms) > DEBOUNCE_MS) {
    last_btn_ms = millis();
    return true;
  }
  return false;
}

void startPhase() {
  if (mode == IDLE || mode == DONE) {                      // first press → begin logging
    mode = LOGGING;
    n = 0;
    n_phases = 0;
    full = false;                          // clear the buffer-full latch for the new run
    next_sample_us = micros();
  }
  if (mode == LOGGING && n_phases < MAX_PHASES) {
    phase_start[n_phases] = n;             // this phase begins at sample n
    ringShowPhase(n_phases);
    n_phases++;
  }
}

// Which phase is sample i in? (walk the phase_start table)
uint8_t phaseOf(uint16_t i) {
  uint8_t p = 0;
  for (uint8_t k = 0; k < n_phases; k++)
    if (i >= phase_start[k]) p = k;
  return p;
}

void dumpCSV() {
  Serial.println(F("--- LOGGING FINISHED ---"));
  Serial.print(F("# samples=")); Serial.print(n);
  Serial.print(F("  rate_hz=")); Serial.print(SAMPLE_HZ);
  Serial.print(F("  duration_s=")); Serial.println(n / (float)SAMPLE_HZ, 2);
  Serial.print(F("# phase boundaries (sample idx):"));
  for (uint8_t k = 0; k < n_phases; k++) { Serial.print(' '); Serial.print(phase_start[k]); }
  Serial.println();
  Serial.println(F("=== DATA START ==="));
  Serial.println(F("idx,t_s,ax_mg,ay_mg,az_mg,phase"));
  for (uint16_t i = 0; i < n; i++) {
    Serial.print(i);                Serial.print(',');
    Serial.print(i / (float)SAMPLE_HZ, 4); Serial.print(',');
    Serial.print(ax_mg[i]);         Serial.print(',');
    Serial.print(ay_mg[i]);         Serial.print(',');
    Serial.print(az_mg[i]);         Serial.print(',');
    Serial.println(phaseOf(i));
  }
  Serial.println(F("=== DATA END ==="));
}

void setup() {
  Serial.begin(USB_BAUD);
  CircuitPlayground.begin();
  CircuitPlayground.setBrightness(35);
  // ±8 g range (the library default is ±2 g). At the ANKLE, heel-strike
  // transients during ordinary walking exceed 2 g, so ±2 g CLIPS the
  // locomotion band — which inflates the Freeze Index and fakes "freezes".
  // ±8 g keeps walking un-clipped; resolution (~4 mg) is still far finer than
  // anything in the freeze band. (A real sensor parameter — cite it in Task 4.)
  CircuitPlayground.setAccelRange(LIS3DH_RANGE_8_G);
  ringFlash(0, 0, 40);                     // idle = dim blue "ready"
}

void loop() {
  bool L = CircuitPlayground.leftButton();
  bool R = CircuitPlayground.rightButton();

  // RIGHT → start / next phase
  if (pressed(R, prev_right)) startPhase();

  // LEFT → stop & dump
  if (pressed(L, prev_left) && mode == LOGGING) {
    mode = DONE;
    ringFlash(40, 0, 0);
    dumpCSV();
  }

  // Sampling (non-blocking, fixed 64 Hz). Stays in LOGGING when full so LEFT can
  // still dump; we just stop storing and latch the red "full" signal once.
  if (mode == LOGGING && !full) {
    uint32_t now = micros();
    if ((int32_t)(now - next_sample_us) >= 0) {
      next_sample_us += SAMPLE_US;
      if (n < MAX_SAMPLES) {
        ax_mg[n] = (int16_t)(CircuitPlayground.motionX() * MS2_TO_MG);
        ay_mg[n] = (int16_t)(CircuitPlayground.motionY() * MS2_TO_MG);
        az_mg[n] = (int16_t)(CircuitPlayground.motionZ() * MS2_TO_MG);
        n++;
      } else {                             // buffer full → stop storing, await LEFT
        full = true;
        ringFlash(40, 0, 0);
      }
    }
  }
}
