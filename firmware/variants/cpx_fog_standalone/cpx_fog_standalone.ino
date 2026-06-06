/*
  ═══════════════════════════════════════════════════════════════════════════
  cpx_fog_standalone.ino  —  Circuit Playground Express
  Parkinson's gait-freeze MONITOR garment  ·  ENGF0031 Scenario 2
  ═══════════════════════════════════════════════════════════════════════════

  STANDALONE — sensing + detection run ON THE BOARD, no laptop:
      accel  →  freeze-of-gait detection  →  on-body report (LEDs + OLED)

  MONITOR BUILD: the coin-motor vibrotactile cue is DISABLED (CUE_ENABLED 0) —
  Cadence REPORTS freezes (red NeoPixels + big category on the on-body OLED
  driven HERE) instead of cueing. Flip CUE_ENABLED to 1 for the closed loop.

  Why not the CNN?  The ATSAMD21 is a 48 MHz Cortex-M0+ with 32 KB RAM and no
  FPU — it cannot host the PyTorch CNN (the weights alone don't fit in RAM).
  So we deploy the clinically-validated classical detector instead:

      Freeze Index (Moore 2008) = power(3–8 Hz) / power(0.5–3 Hz)
                                  on the accelerometer magnitude.

  The CNN stays OFFLINE as the benchmark in the report (LOSO sensitivity /
  specificity). The numbers in the "EXPORTED FROM COLAB" block below — the two
  band-pass filters and the decision threshold — are tuned in the Colab
  pipeline on the SAME band-pass filter this board runs, so they transfer
  exactly. Re-run export_for_device() on the real Daphnet data and paste.

  ── HARDWARE ──  (identical wiring to cpx_fog_streamer)
    Board : Circuit Playground Express (onboard LIS3DH 3-axis accelerometer)
    Motor : coin vibration motor on pad A1 (NPN transistor) — PRESENT but UNUSED
            in the monitor build (CUE_ENABLED 0); kept wired so the closed-loop
            cue can be restored without rework:
                A1 ──[1 kΩ]── base (2N2222 / BC547)
                motor(+) ── VOUT (3.3 V)
                motor(−) ── collector ;  emitter ── GND
                1N4148 flyback diode across the motor (cathode → VOUT)
            (A0 is avoided — onboard speaker/DAC pin.)
    Slide switch : master cue enable (only meaningful if CUE_ENABLED 1).
    Button A     : hold still and press to calibrate the movement-energy floor.
    NeoPixels    : green = walking/monitoring, red = FREEZE alert,
                   dim teal = standing still (gate holding),
                   blue = calibrating the still-floor.
    OLED         : 128x64 SSD1306 on the EXTERNAL I2C bus (SDA = pad A5,
                   SCL = pad A4, 3.3 V / GND) — on-body readout, see below.

  ── SIGNAL CHAIN  (every HOP samples, over the last WINDOW samples) ──
    xyz → magnitude → subtract window mean (kills gravity/DC)
        → band-pass 3–8 Hz, Σ squares  = Pf
        → band-pass 0.5–3 Hz, Σ squares = Pl
    FI = Pf / Pl.
    MOVEMENT GATE (Bachlin 2010, two-threshold): FI is a ratio, so quiet
    standing (noise/noise) can spike it into a false "freeze". We therefore
    require real motion before trusting FI:
        freeze  ⟺  FI > FI_THRESHOLD  AND  (Pf+Pl) > still_floor
    This rejects standing/sitting still. Press button A while standing to
    calibrate still_floor to the wearer.  Decision debounced both ways.

  ── ON-BODY OLED  (128x64 SSD1306, driven HERE on the CPX) ──────────────────
    The on-body readout is wired to the CPX's EXTERNAL I2C bus (Wire):
        OLED SDA ── pad A5    ·    OLED SCL ── pad A4    ·    3.3 V / GND
    That public `Wire` bus lives on SERCOM5 — a SEPARATE peripheral from the
    onboard accelerometer (Wire1 / SERCOM1) and the UART link (Serial1 /
    SERCOM4) — so the screen can never disturb sampling or detection. It shows
    the board's OWN verdict: big freeze category, the Freeze Index, the
    movement-energy gate (energy / floor) and a freeze-episode count. Redrawn
    ONLY once per decision (every HOP = 2 s), so the ~25 ms I2C frame write
    (Wire at 400 kHz) perturbs the 64 Hz sampling by under 1 %.

  ── TELEMETRY LINK  (Serial1 → ESP32 hub; the board still runs without it) ───
    For the Wi-Fi dashboard + HR/SpO2 the CPX ALSO narrates over its hardware
    UART (Serial1, pad A7) to the ESP32 hub, which serves the browser dashboard
    and adds HR/SpO2 from its own MAX30102 (the hub has NO OLED):
        per sample    ax,ay,az\n     int16 milli-g, 64 Hz    → waveform panel
        per decision  #STATE,FI\n    e.g. #WALKING,1.82       → freeze category
    One-way: if no hub is wired the bytes just leave the pad and on-board
    detection (and the on-body OLED) are unaffected — neither depends on it.
        CPX A7 (Serial1 TX) ──▶ ESP32 GPIO16 (RX2) ;  common GND.
  ═══════════════════════════════════════════════════════════════════════════
*/

#include <Adafruit_CircuitPlayground.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
// ── OLED driver select ──────────────────────────────────────────────────────
// 1.3" 128x64 modules are almost always SH1106; 0.96" ones are SSD1306. Both
// ACK at 0x3C and the SSD1306 driver "begins" OK on either, but an SH1106 stays
// BLANK under SSD1306 (different column addressing). Set to match the panel.
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
#include <math.h>

// ╔═══════════════════════════════════════════════════════════════════════╗
// ║  EXPORTED FROM COLAB  (export_for_device, tuned at 64 Hz)              ║
// ╚═══════════════════════════════════════════════════════════════════════╝
#define FS        64           // sample rate — must match the export
#define WINDOW    256          // analysis window, samples (4.0 s)
#define HOP       128          // decision cadence, samples (2.0 s)
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

// ── MONITOR BUILD: the coin-motor cue is DISABLED ───────────────────────────
// Cadence is a freeze-of-gait MONITOR, not a closed-loop cueing device: it
// senses, detects and REPORTS (red NeoPixels + the on-body OLED here, plus
// #STATE,FI telemetry to the ESP32 hub's dashboard) but never drives the
// vibration motor. Set this to 1 only to restore the closed-loop cue.
#define CUE_ENABLED 0

// ── detection / alert tuning ──
const uint8_t  CONFIRM_ON  = 2;       // consecutive freeze windows → freeze ALERT on
const uint8_t  CONFIRM_OFF = 2;       // consecutive clear  windows → freeze ALERT off
const float    CUE_HZ      = 2.0f;    // (only used if CUE_ENABLED) cue pulse rate

// ── movement-energy gate (rejects quiet standing; Bachlin 2010) ──
// freeze ⟺ FI > FI_THRESHOLD AND (Pf+Pl) > still_floor.  still_floor is in the
// same (m/s²)²·window units as the band powers; the default is only a starting
// point — press button A while standing to calibrate it to THIS board + wearer
// (floor = STILL_MARGIN × resting band-power). Below the floor there is no gait
// to "freeze", so we never cue (this is what stops standing-still false alarms).
float          still_floor  = 12.0f;  // (re)set by calibration; sane mid-band default
const float    STILL_MARGIN = 4.0f;   // floor = margin × resting energy
// Clamp band tightened to [8,20] after bench measurement: a dead-still board on
// this hardware reads E≈2.5 (steady) with brief boot/handling transients up to
// ~5, while ANY deliberate motion (gentle 2 Hz rocking → vigorous shake) drives
// E into the tens-to-hundreds. So any floor in [8,20] is correct: above still
// noise (never false-WALKING) yet far below real gait (motion always registers).
// The old [0.5,50] band was too wide — the boot auto-calibration could land
// below the still-noise level (floor≈0.5 → still leaked through as WALKING) or
// pinned to the 50 ceiling. Tightening both ends makes every calibration outcome
// land in the usable band regardless of boot/handling transients.
const float    MIN_FLOOR    = 8.0f;   // floor stays ABOVE measured still-noise (~5)
const float    MAX_FLOOR    = 20.0f;  // … yet well BELOW real-motion energy (tens+)

// ── hardware ──
const uint8_t  MOTOR_PIN   = A1;      // via NPN transistor
const uint32_t SAMPLE_US   = 1000000UL / FS;
const uint32_t CUE_HALF_US = (uint32_t)(1000000.0 / CUE_HZ / 2.0);

// ── optional UART telemetry to the ESP32 display hub (Serial1, pad A7) ──
const uint32_t LINK_BAUD = 115200;                 // 8N1, matches the hub
const float    MS2_TO_MG = 1000.0f / 9.80665f;     // m/s² → milli-g (wire units)

// ── ring buffer of the most-recent magnitude samples ──
float    ring[WINDOW];
uint16_t head      = 0;               // index of oldest sample when full
uint16_t filled    = 0;
uint16_t since_hop = 0;

// ── runtime state ──
enum GaitState { STILL, WALKING, FREEZE };
bool      cueing    = false;
bool      motor_on  = false;
uint8_t   on_count  = 0, off_count = 0;
uint32_t  next_sample_us = 0, next_cue_us = 0;
float     last_fi     = 0.0f;
float     last_energy = 0.0f;          // Pf+Pl on the last window (gate input)
float     g_pf = 0.0f, g_pl = 0.0f;    // last band powers (filled by computeFI)
GaitState g_state = WALKING;           // instantaneous class (for LEDs / serial)
bool      calibrating = false;         // button-A still-floor calibration
uint16_t  calib_left  = 0;             // samples left to collect while calibrating
bool      btnA_prev   = false;         // button-A edge detector

// ── on-body OLED (SSD1306 128x64 on the CPX EXTERNAL I2C: SDA=A5, SCL=A4) ──
// Same `Wire` bus the soldered display sits on; SEPARATE SERCOM from the
// onboard accelerometer (Wire1) and the UART hub link (Serial1).
#define   OLED_ADDR 0x3C               // a few SSD1306 modules are 0x3D
#if OLED_SH1106
Adafruit_SH1106G oled(128, 64, &Wire, -1);   // 1.3" panel, no dedicated reset pin
#else
Adafruit_SSD1306 oled(128, 64, &Wire, -1);   // 0.96" panel, no dedicated reset pin
#endif
bool      oledOK         = false;      // true once the panel ACKs at begin()
bool      haveDecision   = false;      // false until the first window decision
uint16_t  freezeEpisodes = 0;          // confirmed freeze onsets (debounced)
// boot-time I2C scan results (printed from loop() so the USB host catches them)
uint8_t   i2cAddrs[8];                  // up to 8 devices found on the ext bus
uint8_t   i2cCount       = 0;
uint8_t   oledAddrFound  = 0;          // 0x3C / 0x3D if a panel ACKed, else 0
const uint32_t OLED_PROBE_MS = 1000;   // self-heal cadence: re-probe the panel
                                       // this often so a loose wire / power glitch
                                       // recovers WITHOUT a board reset

// Run a band through the SOS cascade (Direct-Form-II-Transposed) from zero
// initial state and return Σ y² over the window — i.e. the band power
// (proportionality cancels in the FI ratio).
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
    w[i] = ring[(head + i) % WINDOW];    // head = oldest when buffer is full
    mean += w[i];
  }
  mean /= WINDOW;
  for (uint16_t i = 0; i < WINDOW; i++) w[i] -= mean;
  g_pf = runBand(FREEZE_SOS, w, WINDOW);   // 3–8 Hz band power
  g_pl = runBand(LOCO_SOS,   w, WINDOW);   // 0.5–3 Hz band power
  return g_pf / (g_pl + 1e-9f);
}

void setMotor(bool on) {
  // Hard safety gate: in the monitor build (CUE_ENABLED 0) the motor pin is
  // ALWAYS held low, whatever the detector logic requests.
  motor_on = (CUE_ENABLED && on);
  digitalWrite(MOTOR_PIN, motor_on ? HIGH : LOW);
}

void showState() {
  // blue = calibrating · red = FREEZE alert (steady in the monitor build) ·
  // dim teal = standing still (gate holding) · green = walking/monitoring
  uint8_t r = 0, g = 0, b = 0;
  if (calibrating)           { r = 0;                              g = 0;  b = 40; }
  else if (cueing)           { r = CUE_ENABLED ? (motor_on ? 90 : 15) : 70; g = 0; b = 0; }
  else if (g_state == STILL) { r = 0;                              g = 4;  b = 6;  }
  else                       { r = 0;                              g = 22; b = 0;  }
  for (uint8_t i = 0; i < 10; i++)
    CircuitPlayground.setPixelColor(i, r, g, b);
}

// Render the on-body readout. Called ONLY at the per-decision cadence (every
// HOP = 2 s) and at the two calibration transitions — NEVER per sample — so
// the ~25 ms I2C frame write (Wire @ 400 kHz) costs under 1 % of sampling.
// The big label tracks the SAME debounced verdict as the red NeoPixels
// (cueing), so the screen, the LEDs and the freeze count never disagree.
void drawOled() {
  oled.clearDisplay();

  // ── title strip ──
  oled.setTextSize(1);
  oled.setTextColor(OLED_WHITE);
  oled.setCursor(0, 0);   oled.print(F("CADENCE"));
  oled.setCursor(80, 0);  oled.print(F("monitor"));
  oled.drawLine(0, 10, 127, 10, OLED_WHITE);

  // ── big category (inverted block when in a confirmed freeze) ──
  const char *label;
  bool alert = false;
  if (calibrating)            label = "CALIB";
  else if (!haveDecision)     label = "WAIT";
  else if (cueing)          { label = "FREEZE"; alert = true; }
  else if (g_state == STILL)  label = "STILL";
  else                        label = "WALKING";

  if (alert) {
    oled.fillRect(0, 13, 128, 19, OLED_WHITE);
    oled.setTextColor(OLED_BLACK);
  } else {
    oled.setTextColor(OLED_WHITE);
  }
  oled.setTextSize(2);
  oled.setCursor(2, 15);
  oled.print(label);
  oled.setTextColor(OLED_WHITE);

  // ── detector internals (small) ──
  oled.setTextSize(1);
  oled.setCursor(0, 36);
  oled.print(F("FI "));
  oled.print(haveDecision ? last_fi : 0.0f, 2);
  oled.print(F(" / "));
  oled.print(FI_THRESHOLD, 2);

  oled.setCursor(0, 46);                  // the movement-energy gate
  oled.print(F("E "));
  oled.print(last_energy, 0);
  oled.print(F(" / "));
  oled.print(still_floor, 0);

  oled.setCursor(0, 56);
  oled.print(F("freezes: "));
  oled.print(freezeEpisodes);

  oled.display();
}

// ── OLED self-heal ──────────────────────────────────────────────────────────
// oledOK latches at begin(); without this, a panel that wasn't seated at boot —
// or one whose wire backs out mid-demo — stays black until a RESET. Probe the
// external bus on a slow cadence: if the panel vanished (two misses, debounced
// against a spurious NAK), stop drawing; if it (re)appears, re-begin() and
// resume. Cheap — a 1-byte ACK probe per call; the rare begin() (~25 ms) runs
// only on recovery and never touches the 64 Hz sampling path (the accelerometer
// is on Wire1, a SEPARATE SERCOM from this external Wire bus).
void serviceOled() {
  static uint32_t nextProbe = 0;
  static uint8_t  miss      = 0;
  if ((int32_t)(millis() - nextProbe) < 0) return;
  nextProbe = millis() + OLED_PROBE_MS;

  uint8_t addr = oledAddrFound ? oledAddrFound : OLED_ADDR;
  Wire.beginTransmission(addr);
  bool present = (Wire.endTransmission() == 0);
  if (!present) {                                   // some modules sit at 0x3D
    Wire.beginTransmission((uint8_t)0x3D);
    if (Wire.endTransmission() == 0) { present = true; addr = 0x3D; }
  }

  if (present) {
    miss = 0;
    if (!oledOK) {                                  // (re)appeared — re-init + resume
      oledAddrFound = addr;
#if OLED_SH1106
      oledOK = oled.begin(addr, true);
#else
      oledOK = oled.begin(SSD1306_SWITCHCAPVCC, addr, true, false);
#endif
      if (oledOK) drawOled();                       // show the live verdict, not blank
    }
  } else if (oledOK && ++miss >= 2) {               // two misses → panel really gone
    oledOK = false;  miss = 0;
  }
}

void setup() {
  Serial.begin(115200);                            // USB monitor (debug only)
  Serial1.begin(LINK_BAUD);                         // optional link to the hub
  CircuitPlayground.begin();
  CircuitPlayground.setAccelRange(LIS3DH_RANGE_8_G);  // ±8 g: ankle heel-strike exceeds ±2 g; match logger/streamer so the FI sees the same un-clipped signal
  CircuitPlayground.setBrightness(40);
  pinMode(MOTOR_PIN, OUTPUT);
  setMotor(false);

  // ── auto-calibrate the still-floor at boot (no button press needed) ──
  // Assume the board rests for the first WINDOW samples (~4 s) and derive the
  // movement-gate floor from the measured resting band-power. This is the SAME
  // code path button A runs, just triggered automatically. If the board was
  // moving at boot, hold it still and tap button A to recalibrate.
  calibrating = true;  calib_left = WINDOW;
  Serial.println("[boot] auto-calibrating still-floor: hold still ~4 s ...");

  showState();

  // ── on-body OLED on the external I2C bus (SDA=A5, SCL=A4) ──
  Wire.begin();
  Wire.setClock(400000);                            // fast-mode → ~25 ms/frame
  // I2C scan on the EXTERNAL bus (A5/A4): reports every device that ACKs, so a
  // blank panel is diagnosable over USB serial (wiring vs wrong address). We
  // then auto-pick whichever SSD1306 address is actually present (0x3C or 0x3D)
  // rather than trusting a fixed #define — begin() does NOT verify the ACK.
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      if (i2cCount < 8) i2cAddrs[i2cCount++] = a;
      if (a == 0x3C || a == 0x3D) oledAddrFound = a;   // an SSD1306-like address
    }
  }
  if (oledAddrFound) {                                // only begin() if present
#if OLED_SH1106
    oledOK = oled.begin(oledAddrFound, true);         // SH110X: begin(addr, reset)
#else
    oledOK = oled.begin(SSD1306_SWITCHCAPVCC, oledAddrFound, true, false);
#endif
  }
  if (oledOK) {                                     // splash until first decision
    // The two Wi-Fi lines MIRROR the ESP32 hub's AP_SSID / SoftAP IP — keep them
    // in sync if you rename the hotspot in esp32_hub.ino (USE_SOFTAP 1).
    oled.clearDisplay();
    oled.setTextColor(OLED_WHITE);
    oled.setTextSize(2);  oled.setCursor(2, 0);   oled.print(F("CADENCE"));
    oled.setTextSize(1);
    oled.setCursor(2, 20);  oled.print(F("WiFi: Cadence-Hub"));
    oled.setCursor(2, 32);  oled.print(F("open 192.168.4.1"));
    oled.setCursor(2, 50);  oled.print(F("auto-cal: hold still"));
    oled.display();
  }

  next_sample_us = micros();
}

void loop() {
  uint32_t now = micros();
  bool enabled = CircuitPlayground.slideSwitch();   // master cue enable

  // ── one-time boot diagnostic, printed from the loop (not setup) ──
  // Native-USB boards drop setup()'s prints during CDC enumeration, so the
  // I2C scan / OLED status is emitted here once the host is reliably attached.
  static bool diagPrinted = false;
  if (!diagPrinted && millis() > 1500) {
    diagPrinted = true;
    Serial.print("[diag] I2C ext(A5/A4):");
    if (i2cCount == 0) Serial.print(" none");
    else for (uint8_t i = 0; i < i2cCount; i++) { Serial.print(" 0x"); Serial.print(i2cAddrs[i], HEX); }
    if (oledAddrFound) {
      Serial.print("  OLED 0x"); Serial.print(oledAddrFound, HEX);
      Serial.println(oledOK ? " begin=OK" : " begin=FAIL");
    } else {
      Serial.println("  OLED not found (check SDA=A5, SCL=A4, 3.3V, GND)");
    }
  }

  // self-heal the OLED: re-probe the panel on a slow cadence so a loose wire or
  // power glitch recovers on its own (oledOK otherwise latches once at boot).
  serviceOled();

  // ── button A: (re)calibrate the still-floor — stand still, then press ──
  bool btnA = CircuitPlayground.leftButton();
  if (btnA && !btnA_prev && !calibrating) {
    calibrating = true;  calib_left = WINDOW;        // collect one fresh window
    cueing = false; setMotor(false); on_count = off_count = 0;
    Serial.println("calibrating still-floor: stand still ~4 s ...");
    showState();
    if (oledOK) drawOled();                          // shows "CALIB"
  }
  btnA_prev = btnA;

  // ── 1. freeze-alert pulse train (square wave at CUE_HZ) ──
  // Only actuates the motor when CUE_ENABLED; in the monitor build this stays
  // dormant — the freeze "alert" is the steady red NeoPixels + on-body OLED.
  if (CUE_ENABLED && cueing && enabled && !calibrating) {
    if ((int32_t)(now - next_cue_us) >= 0) {
      setMotor(!motor_on);
      next_cue_us += CUE_HALF_US;
      showState();
    }
  } else if (motor_on) {
    setMotor(false);                                // keep motor off otherwise
    showState();
  }

  // ── 2. sample accel at a steady FS, no drift ──
  if ((int32_t)(now - next_sample_us) >= 0) {
    next_sample_us += SAMPLE_US;

    float ax = CircuitPlayground.motionX();         // m/s²
    float ay = CircuitPlayground.motionY();
    float az = CircuitPlayground.motionZ();
    float mag = sqrtf(ax * ax + ay * ay + az * az);

    // ── optional: narrate the raw stream to the display hub (waveform panel) ──
    // Same wire format the streamer build uses (int16 milli-g, ax,ay,az\n), so
    // the hub renders it unchanged. Detection below still uses the float m/s²
    // magnitude — this is display only, and harmless with no hub attached.
    int16_t mgx = (int16_t)(ax * MS2_TO_MG);
    int16_t mgy = (int16_t)(ay * MS2_TO_MG);
    int16_t mgz = (int16_t)(az * MS2_TO_MG);
    Serial1.print(mgx); Serial1.print(','); Serial1.print(mgy); Serial1.print(','); Serial1.println(mgz);
    // Also echo to USB so the laptop dashboard works over a direct tether
    // (dashboard_server.py --transport serial). parse_line ignores the "FI=..."
    // debug lines below (>3 fields), so the two interleave harmlessly. Remove
    // this line for the on-body ESP32 deployment, where USB is power-only.
    Serial.print(mgx); Serial.print(','); Serial.print(mgy); Serial.print(','); Serial.println(mgz);

    ring[head] = mag;                               // append → ring buffer
    head = (head + 1) % WINDOW;
    if (filled < WINDOW) filled++;
    since_hop++;

    // ── 2b. calibration: measure resting energy over one still window ──
    if (calibrating) {
      if (calib_left > 0) calib_left--;
      if (calib_left == 0 && filled >= WINDOW) {
        computeFI();                                // fills g_pf, g_pl
        float rest = g_pf + g_pl;
        still_floor = STILL_MARGIN * rest;
        if (still_floor < MIN_FLOOR) still_floor = MIN_FLOOR;
        if (still_floor > MAX_FLOOR) still_floor = MAX_FLOOR;
        calibrating = false;  since_hop = 0;
        Serial.print("calibrated: rest_energy=");  Serial.print(rest, 3);
        Serial.print("  -> still_floor=");         Serial.println(still_floor, 3);
        showState();
        if (oledOK) drawOled();                     // floor updated, back to live
      }
      return;                                       // no detection while calibrating
    }

    // ── 3. every HOP samples (window full), run the gated detector ──
    if (filled >= WINDOW && since_hop >= HOP) {
      since_hop = 0;
      last_fi = computeFI();                         // fills g_pf, g_pl
      last_energy = g_pf + g_pl;                     // movement energy (both bands)
      haveDecision = true;                           // OLED can leave the splash

      bool moving = last_energy > still_floor;       // gate: is there real motion?
      bool freeze = moving && (last_fi > FI_THRESHOLD);
      g_state = !moving ? STILL : (freeze ? FREEZE : WALKING);

      if (freeze) { on_count++;  off_count = 0; }    // STILL & WALKING both clear
      else        { off_count++; on_count  = 0; }

      if (!cueing && on_count >= CONFIRM_ON) {
        cueing = true;  next_cue_us = micros();     // begin cueing now
        freezeEpisodes++;                            // confirmed freeze onset (OLED)
      } else if (cueing && off_count >= CONFIRM_OFF) {
        cueing = false; setMotor(false);
      }
      showState();

      // ── optional: telemetry to the display hub, one line per decision ──
      // "#STATE,FI\n" — the hub's freeze panel shows this. The state is the
      // on-board detector's OWN verdict, so the panel reflects live what the
      // board itself decided (classical Freeze Index — no CNN, no laptop).
      Serial1.print('#');
      Serial1.print(g_state == STILL ? "STILL" : g_state == FREEZE ? "FREEZE" : "WALKING");
      Serial1.print(',');
      Serial1.println(last_fi, 2);

      Serial.print("FI=");      Serial.print(last_fi, 2);
      Serial.print(" E=");      Serial.print(last_energy, 1);
      Serial.print("/");        Serial.print(still_floor, 1);
      Serial.print(g_state == STILL ? "  STILL"
                 : g_state == FREEZE ? "  FREEZE" : "  WALKING");
      if (cueing)               Serial.print(CUE_ENABLED ? " -> cue" : " -> FREEZE alert");
      if (!enabled && CUE_ENABLED) Serial.print("  (cue disabled by slide switch)");
      Serial.println();

      // ── on-body OLED: redraw ONCE per decision (every HOP = 2 s) ──
      if (oledOK) drawOled();
    }
  }
}
