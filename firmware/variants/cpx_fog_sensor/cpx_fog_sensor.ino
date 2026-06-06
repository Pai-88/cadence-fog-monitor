/*
  ═══════════════════════════════════════════════════════════════════════════
  !! NOT THE SUBMITTED BUILD — closed-loop *cueing* prototype (drives the motor).
  !! The submitted Cadence device is the MONITOR sketch, cue disabled:
  !!   firmware/variants/cpx_fog_standalone/cpx_fog_standalone.ino
  cpx_fog_sensor.ino  —  Circuit Playground Express  (the sensor + cue end)
  Parkinson's closed-loop gait garment  ·  ENGF0031 Scenario 2
  ═══════════════════════════════════════════════════════════════════════════

  The CPX is the body-worn SENSOR + ACTUATOR. The thinking is done OFF the CPX,
  on an ESP32-S3 sitting beside it that runs the freeze-of-gait CNN on-body
  (see firmware/esp32_s3_fog). This sketch does two jobs, both non-blocking:

      • samples the onboard accelerometer at 64 Hz and streams it to the S3
      • pulses the vibration motor as a rhythmic gait CUE whenever the S3
        decides a freeze is happening and sends 'C'  (stops again on 'S')

  No laptop, no Wi-Fi in the loop — the whole detect→cue path lives on
  the garment. The wire protocol is byte-for-byte identical to
  cpx_fog_streamer.ino, so this same CPX firmware also works behind the ESP32
  Wi-Fi bridge (the laptop/dashboard path) — only the board on the other end changes.

  ── ARCHITECTURE ────────────────────────────────────────────────────────────
        CPX  ──UART──▶  ESP32-S3      S3 runs the CNN, decides WALK/FREEZE/STILL
             ◀──UART──                S3 sends 'C' / 'S' cue commands back

  ── HARDWARE / WIRING ───────────────────────────────────────────────────────
    Board : Circuit Playground Express (onboard LIS3DH 3-axis accelerometer)
    Link  : hardware UART (Serial1), 3.3 V on both sides → wire directly, NO
            level shifter. TX/RX cross over and the boards MUST share a ground:

                CPX A7 (Serial1 TX) ──▶ S3 GPIO18   (PIN_RX in esp32_s3_fog.ino)
                CPX A6 (Serial1 RX) ◀── S3 GPIO17   (PIN_TX in esp32_s3_fog.ino)
                CPX GND ───────────────  S3 GND      (common ground — required)

            (On the CPX, Serial1 is pre-mapped to pads A6 = RX and A7 = TX, so we
             just call Serial1 with no pin setup.)
    Motor : coin vibration motor on pad A1 via an NPN transistor (a CPX pad
            cannot source the motor's ~90 mA on its own):

                A1 ──[1 kΩ]── base (2N2222 / BC547)
                motor(+) ── VOUT (3.3 V)
                motor(−) ── collector
                emitter  ── GND
                1N4148 flyback diode across the motor (cathode → VOUT)

            (Avoid pad A0 — it is the onboard speaker/DAC pin.)
    The 10 onboard NeoPixels mirror the cue (red pulse) so it reads on camera.

  ── LINK PROTOCOL  (Serial1, 115200 baud, 8N1) ──────────────────────────────
    OUT  CPX → S3, one line per sample at 64 Hz:
            ax,ay,az\n          three int16 in milli-g
    IN   S3 → CPX, single-char commands:
            'C'  start cueing (pulse the motor at CUE_HZ)
            'S'  stop cueing
    64 Hz matches the Daphnet training rate, and milli-g matches the z-score
    stats baked into the S3 sketch — keep BOTH exactly as they are here.

  ── DEBUG ───────────────────────────────────────────────────────────────────
    Set DEBUG_USB to 1 to also echo the stream to the USB serial monitor while
    bringing the link up. Leave it 0 in the field (micro-USB is power only then).
  ═══════════════════════════════════════════════════════════════════════════
*/

#include <Adafruit_CircuitPlayground.h>

// ── Config ─────────────────────────────────────────────────
#define DEBUG_USB 1                                   // 1 = also echo to USB Serial (laptop dashboard via --transport serial); revert to 0 for on-body ESP32-S3 deployment
const uint32_t LINK_BAUD   = 115200;                  // must match the S3's LINK_BAUD
const uint32_t SAMPLE_HZ   = 64;                      // must match the CNN's sample rate
const uint32_t SAMPLE_US   = 1000000UL / SAMPLE_HZ;   // 15625 µs between samples
const uint8_t  MOTOR_PIN   = A1;                      // vibration motor (via NPN)
const float    CUE_HZ      = 2.0;                     // gait-cue pulse rate (Hz)
const uint32_t CUE_HALF_US = (uint32_t)(1000000.0 / CUE_HZ / 2.0);  // half-period
const float    MS2_TO_MG   = 1000.0 / 9.80665;        // m/s² → milli-g

// ── State ──────────────────────────────────────────────────
uint32_t next_sample_us = 0;
bool     cueing         = false;     // are we inside a cue (motor pulse-train)?
bool     motor_on       = false;     // current motor level (toggled each half-period)
uint32_t next_cue_us    = 0;

// Drive the motor and mirror it on the NeoPixel ring for the demo.
void setMotor(bool on) {
  motor_on = on;
  digitalWrite(MOTOR_PIN, on ? HIGH : LOW);
  for (uint8_t i = 0; i < 10; i++)
    CircuitPlayground.setPixelColor(i, on ? 60 : 0, 0, 0);
}

void setup() {
  Serial1.begin(LINK_BAUD);                 // UART link to the S3 (pads A6/A7)
#if DEBUG_USB
  Serial.begin(115200);                     // optional USB debug echo
#endif
  CircuitPlayground.begin();
  CircuitPlayground.setBrightness(40);
  // CircuitPlayground.lis.setRange(LIS3DH_RANGE_4_G);  // widen if gait clips at ±2 g
  pinMode(MOTOR_PIN, OUTPUT);
  setMotor(false);
  next_sample_us = micros();
}

void loop() {
  // ── 1. cue commands coming back from the S3 ──
  while (Serial1.available()) {
    char c = Serial1.read();
    if (c == 'C') {                  // S3 detected a freeze → start the cue
      cueing = true;
      next_cue_us = micros();        // fire the first pulse immediately
    } else if (c == 'S') {           // freeze cleared → stop the cue
      cueing = false;
      setMotor(false);
    }
  }

  // ── 2. non-blocking motor pulse-train at CUE_HZ while cueing ──
  if (cueing) {
    uint32_t now = micros();
    if ((int32_t)(now - next_cue_us) >= 0) {
      setMotor(!motor_on);           // toggle → ~2 Hz square-wave buzz
      next_cue_us += CUE_HALF_US;
    }
  }

  // ── 3. sample + stream accel at a steady 64 Hz up to the S3 ──
  uint32_t now = micros();
  if ((int32_t)(now - next_sample_us) >= 0) {
    next_sample_us += SAMPLE_US;     // fixed schedule → no sample-rate drift

    int16_t ax = (int16_t)(CircuitPlayground.motionX() * MS2_TO_MG);
    int16_t ay = (int16_t)(CircuitPlayground.motionY() * MS2_TO_MG);
    int16_t az = (int16_t)(CircuitPlayground.motionZ() * MS2_TO_MG);

    Serial1.print(ax); Serial1.print(',');
    Serial1.print(ay); Serial1.print(',');
    Serial1.println(az);             // "ax,ay,az\n" — exactly what the S3 parses

#if DEBUG_USB
    Serial.print(ax); Serial.print(',');
    Serial.print(ay); Serial.print(',');
    Serial.println(az);
#endif
  }
}
