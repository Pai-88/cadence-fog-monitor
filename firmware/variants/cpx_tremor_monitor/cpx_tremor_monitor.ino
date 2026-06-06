/*
  ═══════════════════════════════════════════════════════════════════════════
  cpx_tremor_monitor.ino  —  Circuit Playground Express
  Parkinson's WRIST resting-tremor monitor  ·  ENGF0031 Scenario 2
  ═══════════════════════════════════════════════════════════════════════════

  The WRIST pivot of cpx_fog_standalone.ino. Where the freezing-of-gait build
  thresholds the Freeze Index of an ANKLE window and drives a vibrotactile CUE
  (a closed loop), this MONITORS resting tremor on the WRIST and only INDICATES
  it — there is no motor and no cue. It is the on-device twin of
  pi/tremor_detector.py:

      accel → 4–6 Hz tremor-band power (per RAW axis, summed) → rest gate
            → debounce/hysteresis → "tremor present" flag + severity on NeoPixels

  WHY PER-AXIS, NOT MAGNITUDE.  The accel magnitude squares the signal, so a
  tremor oscillating perpendicular to gravity partly cancels (its 4–6 Hz term
  collapses to a 2× harmonic outside the band) and slow voluntary motion folds
  in as a harmonic. Band-passing each RAW axis independently and summing the
  4–6 Hz power is orientation-robust — identical to fog.dsp.tremor_power_axes,
  the feature the worksheet and the live dashboard use. (The FoG build can use
  the magnitude because an ankle strap's orientation is constrained; a wrist's
  is not.)

  MONITOR, NOT THERAPY.  This asserts a "tremor present" flag and shows the
  tremor-power trend for the clinician (medication titration / ON–OFF tracking).
  There is nothing to actuate — the FoG build's A1 vibration motor is simply
  absent here.

  ── HARDWARE ──
    Board : Circuit Playground Express (onboard LIS3DH 3-axis accelerometer)
    Worn  : like a watch face on the back of the WRIST, forearm supported — a
            resting tremor is a tremor of a limb AT REST.
    Motor : NONE. (No NPN transistor, no flyback diode — nothing to switch.)
    Slide switch : telemetry enable — narrate to the ESP32 hub, or monitor
                   silently (LEDs still work either way).
    Button A     : hold the wrist at REST and press to CALIBRATE the tremor
                   threshold to this wearer (see below).
    NeoPixels    : white  = uncalibrated (press A at rest) ·
                   blue   = calibrating ·
                   green  = at rest, no tremor ·
                   dim teal = limb being moved (rest gate holding) ·
                   amber bar = TREMOR present, lit pixels ∝ severity.

  ── THE THRESHOLD IS REAL, NOT A PASTED-IN NUMBER ──
    TREMOR_THRESHOLD defaults to 0 = "uncalibrated", and while uncalibrated the
    monitor asserts NOTHING (exactly like the placeholder in pi/tremor_detector.py
    and the calibration gate in pi/dashboard_server.py). Two honest ways to set it:
      • press button A while holding the wrist at rest — the board measures the
        wearer's OWN resting 4–6 Hz power and sets the threshold a margin above it
        (threshold = TREMOR_MARGIN × resting power); or
      • paste the value analyze_tremor.py suggests from a real wrist capture.
    Either way the number comes from measured data, never from thin air.

  ── OPTIONAL DISPLAY LINK  (Serial1 → ESP32 hub; the board needs no host) ──
    Same wiring and wire-format as the FoG builds, so the existing hub/dashboard
    render it unchanged:
        per sample    ax,ay,az\n          int16 milli-g, 64 Hz   → waveform panel
        per decision  #STATE,TREMOR\n      REST/MOVE/TREMOR,power → status panel
    One-way and optional: with no hub wired the bytes just leave pad A7 and the
    on-board monitoring is unaffected.
        CPX A7 (Serial1 TX) ──▶ ESP32 GPIO16 (RX2) ;  common GND.
  ═══════════════════════════════════════════════════════════════════════════
*/

#include <Adafruit_CircuitPlayground.h>
#include <math.h>

#define FS     64            // sample rate (Hz) — must match the SOS design
#define WINDOW 256           // analysis window, samples (4.0 s)
#define HOP    128           // decision cadence, samples (2.0 s)

// Butterworth band-pass, order 2 → two second-order sections.
// Section coefficients are {b0, b1, b2, a1, a2}  (a0 normalised to 1).
struct Sos { float b0, b1, b2, a1, a2; };
const uint8_t N_SOS = 2;

// 4–6 Hz tremor band — the Parkinsonian resting-tremor frequency. Computed with
// scipy.signal.butter(2, [4/32, 6/32], btype='band', output='sos'); the same
// procedure reproduces the FoG build's 3–8 Hz / 0.5–3 Hz sections exactly.
const Sos TREMOR_SOS[N_SOS] = {
  {  0.00844269f,  0.01688539f,  0.00844269f, -1.57431617f,  0.85437745f },
  {  1.00000000f, -2.00000000f,  1.00000000f, -1.72565146f,  0.88666542f },
};
// 0.5–3 Hz locomotor band (identical to the FoG build) — the rest-gate input.
const Sos LOCO_SOS[N_SOS] = {
  {  0.01278734f,  0.02557469f,  0.01278734f, -1.69059402f,  0.75072617f },
  {  1.00000000f, -2.00000000f,  1.00000000f, -1.93848688f,  0.94143123f },
};

// ── detector tuning (mirrors pi/tremor_detector.py) ──
float TREMOR_THRESHOLD = 0.0f;        // PLACEHOLDER: 0 = uncalibrated ⇒ assert nothing.
                                      //   Set by button-A calibration or analyze_tremor.py.
float         REST_CEILING  = 1.0e9f; // reject windows whose 0.5–3 Hz power exceeds this
                                      //   (gross voluntary movement). Lenient until button-A
                                      //   calibration sets it from the resting baseline; a
                                      //   rest-only tremor threshold alone does NOT guarantee
                                      //   movement leakage stays below it, so the gate matters.
const uint8_t CONFIRM_ON    = 2;      // consecutive tremor windows → assert "present"
const uint8_t CONFIRM_OFF   = 2;      // consecutive clear windows → release (hysteresis)

// ── on-board calibration ──
const float TREMOR_MARGIN  = 4.0f;    // threshold = margin × resting 4–6 Hz power
const float MIN_THRESHOLD  = 1.0e-3f; // floor so the threshold never collapses to 0
const float LOCO_MARGIN    = 12.0f;   // rest-gate trips at margin × resting 0.5–3 Hz power;
                                      //   gates out gross voluntary movement (refine on data)
const float MIN_LOCO_CEIL  = 1.0e-2f; // floor so the gate never collapses to 0

// ── units / link ──
const uint32_t SAMPLE_US = 1000000UL / FS;
const uint32_t LINK_BAUD = 115200;                 // 8N1, matches the hub
const float    MS2_TO_MG = 1000.0f / 9.80665f;     // m/s² → milli-g (wire units)

// ── ring buffers: the three RAW axes (orientation-robust power needs the axes,
//    not the magnitude). 3 × 256 floats ≈ 3 KB of the SAMD21's 32 KB. ──
float    ring_x[WINDOW], ring_y[WINDOW], ring_z[WINDOW];
uint16_t head = 0, filled = 0, since_hop = 0;

// ── runtime state ──
enum MonState { REST, MOVE, TREMOR };
MonState g_state        = REST;       // instantaneous class (LEDs / serial)
bool     tremor_present = false;      // debounced, latched "tremor present" flag
uint8_t  on_count = 0, off_count = 0;
float    g_tremor = 0.0f, g_loco = 0.0f;
uint32_t next_sample_us = 0;
bool     calibrating = false;
uint16_t calib_left  = 0;
bool     btnA_prev   = false;

// Run a band through the SOS cascade (Direct-Form-II-Transposed) from zero state
// and return Σ y² over the window — i.e. the band power.
float runBand(const Sos *sos, const float *x, uint16_t n) {
  float z1[N_SOS] = {0}, z2[N_SOS] = {0}, sumsq = 0.0f;
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

// Copy the ordered (oldest→newest) window from a ring, subtract its mean → w.
void orderedWindow(const float *ring, float *w) {
  float mean = 0.0f;
  for (uint16_t i = 0; i < WINDOW; i++) { w[i] = ring[(head + i) % WINDOW]; mean += w[i]; }
  mean /= WINDOW;
  for (uint16_t i = 0; i < WINDOW; i++) w[i] -= mean;
}

// Fill g_tremor (Σ 4–6 Hz power over the 3 raw axes — tremor_power_axes) and
// g_loco (0.5–3 Hz power on the magnitude — the rest-gate "is the limb moving?").
void computeTremor() {
  static float w[WINDOW];
  orderedWindow(ring_x, w); float tp  = runBand(TREMOR_SOS, w, WINDOW);
  orderedWindow(ring_y, w);       tp += runBand(TREMOR_SOS, w, WINDOW);
  orderedWindow(ring_z, w);       tp += runBand(TREMOR_SOS, w, WINDOW);
  g_tremor = tp;

  float mean = 0.0f;                                   // magnitude window for the gate
  for (uint16_t i = 0; i < WINDOW; i++) {
    uint16_t k = (head + i) % WINDOW;
    w[i] = sqrtf(ring_x[k]*ring_x[k] + ring_y[k]*ring_y[k] + ring_z[k]*ring_z[k]);
    mean += w[i];
  }
  mean /= WINDOW;
  for (uint16_t i = 0; i < WINDOW; i++) w[i] -= mean;
  g_loco = runBand(LOCO_SOS, w, WINDOW);
}

void showState() {
  // white = uncalibrated · blue = calibrating · dim teal = limb moving (gate) ·
  // amber bar = tremor present (lit ∝ severity) · green = at rest, no tremor
  uint8_t r = 0, g = 0, b = 0;
  int lit = 10;
  if (calibrating)                 { r = 0;  g = 0;  b = 40; }
  else if (TREMOR_THRESHOLD <= 0)  { r = 12; g = 12; b = 12; }      // uncalibrated
  else if (g_state == MOVE)        { r = 0;  g = 4;  b = 6;  }      // gate holding
  else if (tremor_present)         {                                // tremor present
    r = 90; g = 40; b = 0;
    float ratio = g_tremor / (TREMOR_THRESHOLD + 1e-9f);
    lit = (int)(ratio * 2.0f);
    if (lit < 1) lit = 1; if (lit > 10) lit = 10;
  } else                           { r = 0;  g = 18; b = 0;  }      // at rest, clear
  bool bar = (!calibrating && TREMOR_THRESHOLD > 0 && tremor_present);
  for (uint8_t i = 0; i < 10; i++) {
    if (bar && i >= lit) CircuitPlayground.setPixelColor(i, 0, 0, 0);
    else                 CircuitPlayground.setPixelColor(i, r, g, b);
  }
}

void setup() {
  Serial.begin(115200);                  // USB monitor (debug)
  Serial1.begin(LINK_BAUD);              // optional link to the ESP32 hub
  CircuitPlayground.begin();
  CircuitPlayground.setBrightness(40);
  showState();
  next_sample_us = micros();
}

void loop() {
  uint32_t now = micros();
  bool stream = CircuitPlayground.slideSwitch();    // telemetry enable

  // ── button A: calibrate the tremor threshold — hold wrist at rest, press ──
  bool btnA = CircuitPlayground.leftButton();
  if (btnA && !btnA_prev && !calibrating) {
    calibrating = true;  calib_left = WINDOW;        // collect one fresh rest window
    tremor_present = false;  on_count = off_count = 0;
    Serial.println("calibrating tremor threshold: hold the wrist at REST ~4 s ...");
    showState();
  }
  btnA_prev = btnA;

  // ── sample accel at a steady FS, no drift ──
  if ((int32_t)(now - next_sample_us) >= 0) {
    next_sample_us += SAMPLE_US;

    float ax = CircuitPlayground.motionX();          // m/s²
    float ay = CircuitPlayground.motionY();
    float az = CircuitPlayground.motionZ();

    // optional: narrate the raw stream to the hub (waveform panel) — display only
    if (stream) {
      Serial1.print((int16_t)(ax * MS2_TO_MG)); Serial1.print(',');
      Serial1.print((int16_t)(ay * MS2_TO_MG)); Serial1.print(',');
      Serial1.println((int16_t)(az * MS2_TO_MG));
    }

    ring_x[head] = ax; ring_y[head] = ay; ring_z[head] = az;
    head = (head + 1) % WINDOW;
    if (filled < WINDOW) filled++;
    since_hop++;

    // ── calibration: measure resting 4–6 Hz power over one still window ──
    if (calibrating) {
      if (calib_left > 0) calib_left--;
      if (calib_left == 0 && filled >= WINDOW) {
        computeTremor();                              // fills g_tremor, g_loco
        float thr = TREMOR_MARGIN * g_tremor;
        if (thr < MIN_THRESHOLD) thr = MIN_THRESHOLD;
        TREMOR_THRESHOLD = thr;
        float ceil = LOCO_MARGIN * g_loco;            // calibrate the rest-gate from the same
        if (ceil < MIN_LOCO_CEIL) ceil = MIN_LOCO_CEIL; //  window, so gross movement is gated
        REST_CEILING = ceil;                          //  out even when the threshold is modest
        calibrating = false;  since_hop = 0;
        Serial.print("calibrated: rest_tremor="); Serial.print(g_tremor, 4);
        Serial.print(" rest_loco=");              Serial.print(g_loco, 4);
        Serial.print("  -> THRESHOLD=");          Serial.print(TREMOR_THRESHOLD, 4);
        Serial.print(" REST_CEILING=");           Serial.println(REST_CEILING, 4);
        next_sample_us = micros() + SAMPLE_US;        // re-anchor clock after heavy compute
        showState();
      }
      return;                                         // no detection while calibrating
    }

    // ── every HOP samples (window full): run the gated, debounced detector ──
    if (filled >= WINDOW && since_hop >= HOP) {
      since_hop = 0;
      computeTremor();                                // fills g_tremor, g_loco

      bool moving = g_loco > REST_CEILING;            // rest gate: limb being moved?
      bool raw = !moving && (TREMOR_THRESHOLD > 0.0f) && (g_tremor > TREMOR_THRESHOLD);
      g_state = moving ? MOVE : (raw ? TREMOR : REST);

      if (raw) { on_count++;  off_count = 0; }        // REST & MOVE both clear
      else     { off_count++; on_count  = 0; }
      if (!tremor_present && on_count >= CONFIRM_ON)        tremor_present = true;
      else if (tremor_present && off_count >= CONFIRM_OFF)  tremor_present = false;

      showState();

      // optional: one telemetry line per decision (hub status panel)
      if (stream) {
        Serial1.print('#');
        Serial1.print(g_state == REST ? "REST" : g_state == MOVE ? "MOVE" : "TREMOR");
        Serial1.print(',');
        Serial1.println(g_tremor, 3);
      }

      Serial.print("tremor=");  Serial.print(g_tremor, 3);
      Serial.print(" loco=");   Serial.print(g_loco, 1);
      Serial.print(" thr=");    Serial.print(TREMOR_THRESHOLD, 3);
      Serial.print(g_state == REST ? "  REST" : g_state == MOVE ? "  MOVE" : "  TREMOR");
      if (tremor_present)       Serial.print(" [PRESENT]");
      if (TREMOR_THRESHOLD <= 0) Serial.print("  (uncalibrated — press A at rest)");
      Serial.println();

      // computeTremor + the serial dump above take tens of ms on the M0+
      // (soft-float, no FPU; ~14–43 ms measured by op-count). Re-anchor the
      // sample clock to NOW so we resume on the 64 Hz grid instead of firing a
      // burst of catch-up samples — bounds the jitter to one hop-boundary gap.
      next_sample_us = micros() + SAMPLE_US;
    }
  }
}
