/*
  ═══════════════════════════════════════════════════════════════════════════
  esp32_hub.ino  —  ESP32 (WROOM DevKitC)
  Parkinson's gait-freeze MONITOR garment  ·  ENGF0031 Scenario 2
  ═══════════════════════════════════════════════════════════════════════════

  The "no-host-PC" build. The ESP32 is a self-contained Wi-Fi HUB: it relays the
  CPX's freeze telemetry and adds heart-rate / SpO2, serving a live dashboard a
  laptop/phone opens over Wi-Fi — over the ESP32's own hotspot by default (no
  router needed), or over a network it joins (USE_SOFTAP 0):

        CPX  ──UART──▶  ESP32  ──Wi-Fi──▶  laptop / phone browser (dashboard)
                          ▲
        MAX30102 ──I2C────┘
        (HR + SpO2)

  The on-body OLED is NOT on this hub — it is driven by the CPX itself (see
  cpx_fog_standalone.ino, external I2C on pads A4/A5), so the wearer's screen
  keeps working even if this hub is off. This hub is the Wi-Fi + vitals layer.

  WHAT THIS SKETCH DOES (monitor build — telemetry + vitals, no actuator):
      • reads the 64 Hz accel stream the CPX sends over the UART (Serial2);
      • reads the CPX detector's per-decision freeze verdict (#STATE,FI line);
      • reads heart-rate + SpO2 from a MAX30102 pulse-oximeter over I2C;
      • hosts a tiny web server (port 80) + WebSocket (port 81) that push accel
        + HR + SpO2 + freeze state to a browser dashboard.

  Cadence is a freeze-of-gait MONITOR: it senses, detects (on the CPX, classical
  Freeze Index) and DISPLAYS — it deliberately drives no vibration cue. The CPX
  runs cpx_fog_standalone.ino with its coin-motor cue disabled (CUE_ENABLED 0)
  and shows the freeze category on its OWN OLED; this hub adds the Wi-Fi
  dashboard + HR/SpO2 readout for a carer/clinician.

  ── WIRING ───────────────────────────────────────────────────────────────────
    CPX  → ESP32  (UART, 3.3 V both sides — connect DIRECTLY, no level shifter;
                   TX/RX cross over and the boards MUST share a ground):
        CPX A7 (Serial1 TX) ──▶ ESP32 GPIO16 (RX2)
        CPX A6 (Serial1 RX) ◀── ESP32 GPIO17 (TX2)
        CPX GND ───────────────  ESP32 GND          (common ground, mandatory)

    MAX30102 → ESP32  (I2C; the breakout has its own 3.3 V regulator):
        VIN ── 3V3        SDA ── GPIO21
        GND ── GND        SCL ── GPIO22
    (The on-body OLED is on the CPX, not here — this hub's I2C carries only the
     MAX30102.)

    Power each board from its own 5 V (USB power bank) — do NOT also link the
    two 3.3 V rails.

    NOTE (module variant): GPIO16/17 are the default Serial2 pins on a WROOM
    DevKitC; on a WROVER the PSRAM takes them, so pick two free GPIOs (e.g.
    25/26) and rewire. On an ESP32-S3 the default I2C pins differ — set SDA_PIN/
    SCL_PIN below to match your board's silkscreen.

  ── LIBRARIES (install via Arduino Library Manager) ──────────────────────────
    • "WebSockets" by Markus Sattler  (a.k.a. arduinoWebSockets / Links2004)
    • "SparkFun MAX3010x Pulse and Proximity Sensor Library"
    (The Adafruit SSD1306 + GFX libs live with the CPX sketch, not here.)
    WebServer, WiFi, LittleFS and Wire ship with the ESP32 Arduino core.

  ── ONE-TIME: upload the dashboard page to the ESP32's flash filesystem ───────
    This sketch serves dashboard.html from LittleFS. Copy the project's
    dashboard.html into a "data/" folder next to this .ino, then upload it once
    with the "ESP32 Sketch Data Upload" tool (Arduino IDE) or `pio run -t
    uploadfs` (PlatformIO). If the file is missing, "/" returns a short hint
    instead of the dashboard.

  ── SET BEFORE FLASHING — how the laptop reaches the dashboard over Wi-Fi ─────
    USE_SOFTAP 1  (DEFAULT, recommended for demos): the ESP32 makes its OWN
        hotspot. On the laptop/phone, join Wi-Fi "Cadence-Hub" (pass: cadence123)
        and open  http://192.168.4.1/  — no router, no IP hunting, works in any
        room. While joined, that device has no internet (fine for a demo).
    USE_SOFTAP 0: the ESP32 instead JOINS your network — set WIFI_SSID / WIFI_PASS
        (home/lab Wi-Fi or a phone hotspot the laptop is also on), then open the
        IP it prints on the USB serial monitor:  http://<that-ip>/   (NOTE: many
        campus/guest networks block device-to-device, so the page won't load —
        prefer SoftAP for the assessment.)
  ═══════════════════════════════════════════════════════════════════════════
*/

#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <LittleFS.h>
#include <Wire.h>
#include "MAX30105.h"          // SparkFun lib drives the MAX30102 too
#include "spo2_algorithm.h"    // Maxim's HR + SpO2 estimator

// ── SET THESE ──────────────────────────────────────────────
// How the laptop/phone reaches the dashboard over Wi-Fi:
//   1 = ESP32 is its OWN hotspot (recommended for demos — self-contained)
//   0 = ESP32 joins an existing Wi-Fi network (set WIFI_SSID / WIFI_PASS)
#define USE_SOFTAP 1

// SoftAP (USE_SOFTAP 1): the hotspot the ESP32 creates. The laptop joins THIS,
// then opens http://192.168.4.1/ . AP_PASS is the hotspot's OWN password (WPA2
// needs >= 8 chars) — change it to taste; it is not a credential to a real net.
const char* AP_SSID = "Cadence-Hub";
const char* AP_PASS = "cadence123";

// STA (USE_SOFTAP 0): an existing network to JOIN. Placeholders — set your own.
const char* WIFI_SSID = "YOUR_WIFI_SSID";       // ← your network / phone hotspot
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";   // ← its password

// ── UART link to the CPX (Serial2) ─────────────────────────
const uint32_t LINK_BAUD = 115200;              // 8N1, matches the CPX
const int RXD2 = 16;   // ESP32 RX2  ← CPX A7 (TX)   (WROVER: use a free pin e.g. 25)
const int TXD2 = 17;   // ESP32 TX2  → CPX A6 (RX)   (WROVER: use a free pin e.g. 26)

// ── MAX30102 I2C pins (defaults for a WROOM DevKitC) ───────
const int SDA_PIN = 21;
const int SCL_PIN = 22;

// ── Servers + status LED ───────────────────────────────────
const uint16_t HTTP_PORT = 80;     // serves the dashboard page
const uint16_t WS_PORT   = 81;     // streams JSON to the browser
const int STATUS_LED = 2;          // solid = a browser is watching, blink = connecting

WebServer        http(HTTP_PORT);
WebSocketsServer webSocket(WS_PORT);
MAX30105         pox;              // the MAX30102 sensor

// ── How often we push each kind of update to the browser ───
const uint32_t WAVEFORM_MS = 50;   // 20 Hz — the live accel trace
const uint32_t VITALS_MS   = 1000; //  1 Hz — HR / SpO2 / freeze state
uint32_t lastWaveform = 0, lastVitals = 0;

// ── Accel samples parsed from the UART, awaiting the next waveform push ──
const int ACC_CAP = 64;            // plenty for one 50 ms batch at 64 Hz
int16_t accX[ACC_CAP], accY[ACC_CAP], accZ[ACC_CAP];
int     accN = 0;

// ── Latest freeze telemetry (only set once the CPX/CNN sends it; see below) ──
char    gaitState[12] = "";        // "" = unknown, else WALKING / FREEZE / STILL
float   freezeIndex   = 0.0f;
bool    haveState     = false;

// ── MAX30102 rolling buffers for the 100-sample SpO2 window ──
uint32_t irBuf[100], redBuf[100];
int      poxFilled   = 0;          // how many of the 100 slots are populated
int      sinceCompute = 0;         // new samples since the last HR/SpO2 estimate
int32_t  heartRate = 0, spo2 = 0;  // latest estimates
int8_t   hrValid   = 0, spo2Valid = 0;

// ── Network: bring Wi-Fi up and keep it up (called in setup + each loop) ──────
// SoftAP: start the hotspot once, then early-return (cheap to call every loop).
// STA:    (re)join the network whenever the link drops, blinking the LED.
void netManage() {
#if USE_SOFTAP
  static bool apUp = false;
  if (apUp) return;
  WiFi.mode(WIFI_AP);
  apUp = WiFi.softAP(AP_SSID, AP_PASS);
  if (apUp) {
    Serial.print("SoftAP '");                Serial.print(AP_SSID);
    Serial.print("' up (pass: ");            Serial.print(AP_PASS);
    Serial.print(") — join it, then open  http://");
    Serial.print(WiFi.softAPIP());           Serial.println("/");
  } else {
    Serial.println("SoftAP failed to start");
  }
#else
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 3000) {
    digitalWrite(STATUS_LED, (millis() / 150) & 1);   // blink while joining
    delay(10);
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Wi-Fi up — open  http://");
    Serial.print(WiFi.localIP());
    Serial.println("/   on a device on the same network");
  }
#endif
}

// ── Parse one line from the CPX ──────────────────────────────────────────────
// Two line kinds share the UART:
//   "ax,ay,az"            three int16 milli-g per sample   → the waveform
//   "#WALKING,1.23"       optional freeze telemetry (state,FI) the CPX may send
//                          once it computes a decision. Harmless if never sent.
void parseLine(char* s) {
  if (s[0] == '#') {                                   // freeze telemetry line
    char* comma = strchr(s + 1, ',');
    if (!comma) return;
    *comma = '\0';
    strncpy(gaitState, s + 1, sizeof(gaitState) - 1);
    gaitState[sizeof(gaitState) - 1] = '\0';
    freezeIndex = atof(comma + 1);
    haveState = true;
    return;
  }
  // accel: split on the two commas
  char* c1 = strchr(s, ',');
  if (!c1) return;
  char* c2 = strchr(c1 + 1, ',');
  if (!c2) return;                                     // malformed → skip (e.g. boot banner)
  *c1 = *c2 = '\0';
  if (accN < ACC_CAP) {
    accX[accN] = (int16_t)atoi(s);
    accY[accN] = (int16_t)atoi(c1 + 1);
    accZ[accN] = (int16_t)atoi(c2 + 1);
    accN++;
  }
}

// ── Drain the UART, splitting on newlines (partial lines carry across calls) ──
void pumpSerial() {
  static char line[64];
  static int  len = 0;
  while (Serial2.available()) {
    char ch = (char)Serial2.read();
    if (ch == '\n' || ch == '\r') {
      if (len > 0) { line[len] = '\0'; parseLine(line); len = 0; }
    } else if (len < (int)sizeof(line) - 1) {
      line[len++] = ch;
    } else {
      len = 0;                                         // overlong garbage → drop
    }
  }
}

// ── Poll the MAX30102 without blocking the rest of the loop ──────────────────
// We keep a 100-sample window of IR + RED; each ~25 fresh samples we re-run
// Maxim's estimator (≈ once a second). HR/SpO2 are only valid with a clean
// signal and a still finger on the sensor.
void pumpPulseOx() {
  pox.check();                                         // pull anything the FIFO has
  while (pox.available()) {
    uint32_t red = pox.getRed();
    uint32_t ir  = pox.getIR();
    pox.nextSample();
    if (poxFilled < 100) {                             // initial fill
      redBuf[poxFilled] = red; irBuf[poxFilled] = ir; poxFilled++;
    } else {                                           // slide the window by one
      memmove(redBuf, redBuf + 1, 99 * sizeof(uint32_t));
      memmove(irBuf,  irBuf  + 1, 99 * sizeof(uint32_t));
      redBuf[99] = red; irBuf[99] = ir;
    }
    sinceCompute++;
  }
  if (poxFilled == 100 && sinceCompute >= 25) {
    sinceCompute = 0;
    maxim_heart_rate_and_oxygen_saturation(
        irBuf, 100, redBuf, &spo2, &spo2Valid, &heartRate, &hrValid);
  }
}

// ── Broadcasts ───────────────────────────────────────────────────────────────
void sendWaveform() {
  if (accN == 0) return;
  // {"type":"waveform","accel":[[x,y,z],...]}
  static char buf[1400];
  int p = snprintf(buf, sizeof(buf), "{\"type\":\"waveform\",\"accel\":[");
  for (int i = 0; i < accN; i++) {
    p += snprintf(buf + p, sizeof(buf) - p, "%s[%d,%d,%d]",
                  i ? "," : "", accX[i], accY[i], accZ[i]);
    if (p > (int)sizeof(buf) - 32) break;              // safety: never overrun
  }
  snprintf(buf + p, sizeof(buf) - p, "]}");
  webSocket.broadcastTXT(buf);
  accN = 0;                                            // batch sent → reset
}

void sendVitals() {
  // One message carries the vitals and (if known) the freeze state.
  static char buf[256];
  snprintf(buf, sizeof(buf),
    "{\"type\":\"vitals\",\"hr\":%ld,\"hr_valid\":%d,"
    "\"spo2\":%ld,\"spo2_valid\":%d,"
    "\"state\":%s%s%s,\"freeze_index\":%.2f,\"has_state\":%s}",
    (long)heartRate, hrValid, (long)spo2, spo2Valid,
    haveState ? "\"" : "", haveState ? gaitState : "null", haveState ? "\"" : "",
    freezeIndex, haveState ? "true" : "false");
  webSocket.broadcastTXT(buf);
}

// ── HTTP: serve the dashboard from LittleFS ──────────────────────────────────
void handleRoot() {
  if (LittleFS.exists("/dashboard.html")) {
    File f = LittleFS.open("/dashboard.html", "r");
    http.streamFile(f, "text/html");
    f.close();
  } else {
    http.send(200, "text/html",
      "<h2>ESP32 hub is running</h2><p>Upload <code>dashboard.html</code> to "
      "LittleFS (put it in <code>data/</code> and run the Sketch Data Upload "
      "tool), then refresh.</p>");
  }
}

void setup() {
  Serial.begin(115200);                                // USB debug (ESP32's own port)
  Serial2.begin(LINK_BAUD, SERIAL_8N1, RXD2, TXD2);    // UART link from the CPX
  pinMode(STATUS_LED, OUTPUT);
  Serial.println("\nESP32 hub starting...");

  // MAX30102 over I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  if (pox.begin(Wire, I2C_SPEED_FAST)) {
    // ledBrightness, sampleAverage, ledMode(2=Red+IR), sampleRate, pulseWidth, adcRange
    pox.setup(60, 4, 2, 100, 411, 4096);
    Serial.println("MAX30102 ready (place a fingertip on it for HR/SpO2)");
  } else {
    Serial.println("MAX30102 NOT found — check I2C wiring (HR/SpO2 will read 0)");
  }

  // Filesystem for the dashboard page
  if (!LittleFS.begin(true)) Serial.println("LittleFS mount failed");

  // Wi-Fi (SoftAP hotspot or join a network — see USE_SOFTAP) + servers
  netManage();
  http.on("/", handleRoot);
  http.begin();
  webSocket.begin();
}

void loop() {
  netManage();
  http.handleClient();
  webSocket.loop();

  pumpSerial();      // CPX → accel + (optional) freeze telemetry
  pumpPulseOx();     // MAX30102 → HR / SpO2

  // solid LED once at least one browser is connected, else blink (works in
  // SoftAP and STA — in AP mode WiFi.status() is never WL_CONNECTED).
  bool watched = webSocket.connectedClients() > 0;
  digitalWrite(STATUS_LED, watched ? HIGH : ((millis() / 150) & 1));

  uint32_t now = millis();
  if (now - lastWaveform >= WAVEFORM_MS) { lastWaveform = now; sendWaveform(); }
  if (now - lastVitals   >= VITALS_MS)   { lastVitals   = now; sendVitals();   }
}
