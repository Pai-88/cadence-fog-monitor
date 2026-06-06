/*
  ═══════════════════════════════════════════════════════════════════════════
  !! NOT THE SUBMITTED BUILD — closed-loop *cueing* prototype (drives the motor).
  !! The submitted Cadence device is the MONITOR sketch, cue disabled:
  !!   firmware/variants/cpx_fog_standalone/cpx_fog_standalone.ino
  cpx_fog_streamer.ino  —  Circuit Playground Express
  Parkinson's closed-loop gait garment  ·  ENGF0031 Scenario 2
  ═══════════════════════════════════════════════════════════════════════════

  ONE onboard accelerometer in, ONE vibration motor out — a 3-in-1 device:
      • streams accel for TREMOR monitoring + FREEZE-of-gait detection (laptop CNN)
      • on a detected freeze, the laptop sends 'C' and this board pulses the
        motor as a rhythmic gait CUE (the "treatment" arm of the loop)

  ── ARCHITECTURE  (the wearable is now UNTETHERED) ──────────────────────────
        CPX  ──UART──▶  ESP32  ──Wi-Fi/TCP──▶  laptop (runs the CNN)
             ◀──UART──         ◀──Wi-Fi/TCP──
    The CPX no longer talks to the laptop over USB. Instead it talks to a small
    ESP32 over a short 3.3 V UART link; the ESP32 relays the stream to the laptop
    over Wi-Fi and relays the laptop's cue commands back. The payloads are byte-for-
    byte identical to the old USB protocol — only the wire changed — so nothing
    downstream on the laptop (the CNN, the dashboard) has to change.

  ── HARDWARE ──────────────────────────────────────────────────────────────
    Board : Circuit Playground Express (onboard LIS3DH 3-axis accelerometer)
    Link  : hardware UART (Serial1) to the ESP32 — 3.3 V both sides, so the
            lines connect DIRECTLY, no level shifter. TX and RX cross over and
            the two boards MUST share a ground:

                CPX A7 (Serial1 TX) ──▶ ESP32 GPIO5  (RX2)
                CPX A6 (Serial1 RX) ◀── ESP32 GPIO6  (TX2)
                CPX GND ───────────────  ESP32 GND        (common ground)

            (Serial1 is the SAMD21 hardware UART; on the CPX it is pre-mapped to
            pads A6 = RX and A7 = TX, so we just call Serial1 — no pin setup.)
    Motor : coin vibration motor on pad A1, switched by an NPN transistor —
            a CPX pad can't source the ~90 mA a motor needs, so:

                A1 ──[1 kΩ]── base (2N2222 / BC547)
                motor(+) ── VOUT (3.3 V)
                motor(−) ── collector
                emitter  ── GND
                1N4148 flyback diode across the motor (cathode → VOUT)

            (A0 is deliberately avoided — it is the onboard speaker/DAC pin.)
    The 10 onboard NeoPixels mirror the cue visually (red pulse) for the demo.

  ── LINK PROTOCOL  (UART on Serial1, 115200 baud, 8N1) ──────────────────────
    OUT  CPX → ESP32 → laptop, one line per sample at 64 Hz:
            ax,ay,az\n          three int16 in milli-g
    IN   laptop → ESP32 → CPX, single-char commands:
            'C'  start cueing (pulse motor at CUE_HZ)
            'S'  stop cueing

  64 Hz is chosen to match the Daphnet training dataset exactly, so the laptop's
  CNN sees the same sample rate it was trained on.

  ── DEBUG ───────────────────────────────────────────────────────────────────
    Set DEBUG_USB to 1 to also echo the stream to the USB serial monitor while
    you bring the link up. Leave it 0 in the field (micro-USB is power only).
  ═══════════════════════════════════════════════════════════════════════════
*/

#include <Adafruit_CircuitPlayground.h>

// ── Config ─────────────────────────────────────────────────
#define DEBUG_USB 0                               // 1 = also echo to USB Serial
const uint32_t LINK_BAUD      = 115200;           // UART to the ESP32 (8N1)
const uint32_t SAMPLE_HZ      = 64;
const uint32_t SAMPLE_US      = 1000000UL / SAMPLE_HZ;   // 15625 µs
const uint8_t  MOTOR_PIN      = A1;                       // via NPN transistor
const float    CUE_HZ         = 2.0;                      // gait-cue pulse rate
const uint32_t CUE_HALF_US    = (uint32_t)(1000000.0 / CUE_HZ / 2.0);
const float    MS2_TO_MG      = 1000.0 / 9.80665;         // m/s² → milli-g

// ── State ──────────────────────────────────────────────────
uint32_t next_sample_us = 0;
bool     cueing         = false;
bool     motor_on       = false;
uint32_t next_cue_us    = 0;

void setMotor(bool on) {
  motor_on = on;
  digitalWrite(MOTOR_PIN, on ? HIGH : LOW);
  // Mirror on the NeoPixel ring so the cue is visible in a demo / video.
  for (uint8_t i = 0; i < 10; i++)
    CircuitPlayground.setPixelColor(i, on ? 60 : 0, 0, 0);
}

void setup() {
  Serial1.begin(LINK_BAUD);                 // UART link to the ESP32 (A6/A7)
#if DEBUG_USB
  Serial.begin(115200);                     // optional USB debug echo
#endif
  CircuitPlayground.begin();
  CircuitPlayground.setBrightness(40);
  // ±8 g range (library default is ±2 g). At the ANKLE, walking heel-strikes
  // exceed 2 g, so ±2 g clips the locomotion band and inflates the Freeze Index
  // — the CNN must see the SAME un-clipped signal here as in the logger/Daphnet.
  CircuitPlayground.setAccelRange(LIS3DH_RANGE_8_G);
  pinMode(MOTOR_PIN, OUTPUT);
  setMotor(false);
  next_sample_us = micros();
}

void loop() {
  // ── 1. Handle incoming cue commands (from the laptop, via the ESP32) ──
  while (Serial1.available()) {
    char c = Serial1.read();
    if (c == 'C') {
      cueing = true;
      next_cue_us = micros();           // start a pulse immediately
    } else if (c == 'S') {
      cueing = false;
      setMotor(false);
    }
  }

  // ── 2. Non-blocking cue pulse train at CUE_HZ ──
  if (cueing) {
    uint32_t now = micros();
    if ((int32_t)(now - next_cue_us) >= 0) {
      setMotor(!motor_on);              // toggle → square-wave buzz
      next_cue_us += CUE_HALF_US;
    }
  }

  // ── 3. Sample + stream accel at a steady 64 Hz (up to the ESP32) ──
  uint32_t now = micros();
  if ((int32_t)(now - next_sample_us) >= 0) {
    next_sample_us += SAMPLE_US;        // schedule next tick (no drift)

    int16_t ax = (int16_t)(CircuitPlayground.motionX() * MS2_TO_MG);
    int16_t ay = (int16_t)(CircuitPlayground.motionY() * MS2_TO_MG);
    int16_t az = (int16_t)(CircuitPlayground.motionZ() * MS2_TO_MG);

    Serial1.print(ax); Serial1.print(',');
    Serial1.print(ay); Serial1.print(',');
    Serial1.println(az);

#if DEBUG_USB
    Serial.print(ax); Serial.print(',');
    Serial.print(ay); Serial.print(',');
    Serial.println(az);
#endif
  }
}
