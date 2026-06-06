/*
  ═══════════════════════════════════════════════════════════════════════════
  esp32_wifi_bridge.ino  —  ESP32 (WROOM DevKitC)
  Parkinson's closed-loop gait garment  ·  ENGF0031 Scenario 2
  ═══════════════════════════════════════════════════════════════════════════

  A transparent UART ⇄ Wi-Fi BRIDGE. It carries the same bytes the old USB cable
  used to, so the wearable can be untethered:

        CPX  ──UART──▶  ESP32  ──Wi-Fi/TCP──▶  laptop (runs the CNN)
             ◀──UART──         ◀──Wi-Fi/TCP──

  The ESP32 does NO signal processing. It just pumps bytes both ways:
      • Serial2 (from the CPX)  →  TCP socket (up to the laptop)     [accel stream]
      • TCP socket (from the laptop) →  Serial2 (down to the CPX)    ['C' / 'S' cue]

  ── ROLES ───────────────────────────────────────────────────────────────────
    The laptop is the TCP SERVER; this ESP32 is the CLIENT and dials in to it. That
    keeps the fixed address on the laptop (the base station), which is easy to pin
    with a static / reserved-DHCP lease, while the ESP32 just needs to know it.

  ── WIRING  (3.3 V both sides — connect DIRECTLY, no level shifter) ──────────
    TX/RX cross over and the two boards MUST share a ground:

        CPX A7 (Serial1 TX) ──▶ ESP32 GPIO5  (RX2)
        CPX A6 (Serial1 RX) ◀── ESP32 GPIO6  (TX2)
        CPX GND ───────────────  ESP32 GND        (common ground, mandatory)

    Power each board from its own 5 V (USB power bank / battery) — do NOT also
    link the two 3.3 V rails.

    NOTE (module variant): the pins below are set for an ESP32-S3 mini (GPIO5/6,
    routed through the S3 UART matrix — Serial2 still works). On a WROOM DevKitC
    use 16/17; on a WROVER (PSRAM steals 16/17) pick two free GPIOs and rewire.

  ── LINK PROTOCOL  (115200 baud, 8N1, identical to the old USB protocol) ─────
    up    ax,ay,az\n   three int16 in milli-g, 64 Hz   (CPX → laptop)
    down  'C' / 'S'    one char on a freeze state-change (laptop → CPX)

  ── SET BEFORE FLASHING ──────────────────────────────────────────────────────
    WIFI_SSID / WIFI_PASS — the network the host is on (e.g. your phone's hotspot).
    HOST_IP / HOST_PORT     — the host's IP (your laptop) and the port it listens on
                            (fog.config.BRIDGE_PORT, default 8765).
  ═══════════════════════════════════════════════════════════════════════════
*/

#include <WiFi.h>

// ── SET THESE ──────────────────────────────────────────────
// Wi-Fi creds are injected at flash time via -DCAD_SSID / -DCAD_PASS so real
// credentials never live in this repo. The placeholders are only a fallback when
// you compile WITHOUT those build flags.
#ifndef CAD_SSID
#define CAD_SSID YOUR_WIFI_SSID
#endif
#ifndef CAD_PASS
#define CAD_PASS YOUR_WIFI_PASSWORD
#endif
#define CAD_STR2(s) #s
#define CAD_STR(s)  CAD_STR2(s)
const char* WIFI_SSID = CAD_STR(CAD_SSID);      // set with -DCAD_SSID=<name> at flash time
const char* WIFI_PASS = CAD_STR(CAD_PASS);      // set with -DCAD_PASS=<pass> at flash time
const char* HOST_IP   = "192.168.1.50";         // ← set to YOUR laptop's IP (run: ipconfig getifaddr en0)
const uint16_t HOST_PORT = 8765;                  // ← must match BRIDGE_PORT on the host

// ── UART link to the CPX (Serial2) ─────────────────────────
const uint32_t LINK_BAUD = 115200;              // 8N1, matches the CPX
const int RXD2 = 44;   // ESP32 RX2  ← CPX A7 (Serial1 TX)   (S3 mini "RX" silk = GPIO44)
const int TXD2 = 43;   // ESP32 TX2  → CPX A6 (Serial1 RX)   (S3 mini "TX" silk = GPIO43)

// ── Status LED (onboard on most WROOM DevKitC boards) ──────
const int STATUS_LED = 2;   // solid = linked to the laptop, blinking = connecting

// ── Reconnect throttle ─────────────────────────────────────
const uint32_t RECONNECT_MS    = 1000;   // don't hammer connect() every loop
const uint32_t CONNECT_TMO_MS  = 3000;   // give up a connect attempt after this

WiFiClient client;
uint32_t lastConnectAttempt = 0;
uint32_t lastBlink = 0;
bool ledState = false;

// Join Wi-Fi if we have dropped off. Non-fatal: we simply retry next loop.
void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  // Wait briefly for the join so we don't spam begin(); blink the LED meanwhile.
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < CONNECT_TMO_MS) {
    digitalWrite(STATUS_LED, (millis() / 150) & 1);
    delay(10);
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Wi-Fi up, IP ");
    Serial.println(WiFi.localIP());
  }
}

// (Re)open the TCP connection to the laptop. Throttled so a missing server doesn't
// stall the byte pump. Returns with client either connected or not — callers
// guard on client.connected() before writing.
void ensureServer() {
  if (client.connected()) return;
  if (millis() - lastConnectAttempt < RECONNECT_MS) return;
  lastConnectAttempt = millis();
  client.stop();                                   // clear any half-open socket
  if (client.connect(HOST_IP, HOST_PORT, CONNECT_TMO_MS)) {
    client.setNoDelay(true);                       // low latency for tiny lines
    Serial.println("TCP link to laptop up");
  }
}

void setup() {
  Serial.begin(115200);                            // USB debug (ESP32's own port)
  Serial2.begin(LINK_BAUD, SERIAL_8N1, RXD2, TXD2);// UART link to the CPX
  pinMode(STATUS_LED, OUTPUT);
  WiFi.mode(WIFI_STA);
  Serial.println("\nESP32 Wi-Fi bridge starting...");
}

void loop() {
  ensureWifi();
  ensureServer();

  bool linked = client.connected();
  digitalWrite(STATUS_LED, linked ? HIGH : ((millis() / 150) & 1));

  // ── CPX → laptop : batch whatever arrived on the UART into one socket write ──
  // 64 Hz × ~12 B/line is tiny, but batching avoids one TCP packet per byte.
  static uint8_t up[256];
  int n = 0;
  while (Serial2.available() && n < (int)sizeof(up)) up[n++] = (uint8_t)Serial2.read();
  if (n > 0 && linked) client.write(up, n);

  // ── laptop → CPX : forward the cue bytes ('C' / 'S') straight down the UART ──
  while (linked && client.available()) Serial2.write((uint8_t)client.read());
}
