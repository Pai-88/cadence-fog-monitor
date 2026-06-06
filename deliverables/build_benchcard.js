/* Cadence — one-page A4 "Demo Run Card" for the ENGF0031 live assessment.
 * Build:  NODE_PATH=$(npm root -g) node build_benchcard.js
 * Output: cadence_benchcard.pptx  (single A4-portrait slide) → convert to PDF.
 *
 * A printable bench reference so the run sequence, the on-screen legend and the
 * Wi-Fi join details don't have to be recited from memory on the day. Every
 * value here mirrors the firmware as built:
 *   - CPX  cpx_fog_standalone.ino : OLED on external I2C (A5/A4), button-A calib,
 *                                   NeoPixel colours, FI / energy-gate fields.
 *   - ESP32 esp32_hub.ino         : SoftAP "Cadence-Hub" → http://192.168.4.1/.
 * Honest framing: research prototype, NOT a medical device.
 */
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const {
  FaListOl, FaMicrochip, FaWifi, FaExclamationTriangle, FaPowerOff,
} = require("react-icons/fa");

// ── palette (shared with the Cadence leaflet; NO "#" prefixes) ──
const DEEP = "0B3B4A";   // hero / footer band
const TEAL = "0E7C86";   // monitor accent
const BLUE = "1C7293";   // detect accent
const AMBER = "F2A541";  // display accent (the on-body screen)
const MINT = "2BB7A3";   // highlights / signal motif
const BG   = "F5F8F8";   // page background
const CARD = "FFFFFF";
const INK  = "12343B";   // body text
const MUTE = "5E7378";   // muted text
const ICE  = "CFE8EA";   // light text on dark

function renderIconSvg(Icon, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(Icon, { color, size: String(size) }));
}
async function icon(Icon, color, size = 256) {
  const svg = renderIconSvg(Icon, color, size);
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}
const shadow = () => ({ type: "outer", color: "0B3B4A", blur: 7, offset: 3, angle: 90, opacity: 0.12 });

(async () => {
  const I = {
    run:     await icon(FaListOl, "#FFFFFF"),
    body:    await icon(FaMicrochip, "#FFFFFF"),
    wifi:    await icon(FaWifi, "#FFFFFF"),
    warn:    await icon(FaExclamationTriangle, "#FFFFFF"),
    power:   await icon(FaPowerOff, "#FFFFFF"),
  };

  const pres = new pptxgen();
  pres.defineLayout({ name: "A4P", width: 8.27, height: 11.69 });
  pres.layout = "A4P";
  pres.author = "UCL ENGF0031 Scenario 2";
  pres.title = "Cadence — Demo Run Card";
  const s = pres.addSlide();
  s.background = { color: BG };

  const PW = 8.27, PH = 11.69, M = 0.55, CW = PW - 2 * M;

  // helper: a rounded white card with a soft shadow
  const card = (x, y, w, h) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.09, fill: { color: CARD },
    line: { color: "E2ECEC", width: 1 }, shadow: shadow(),
  });
  // helper: a coloured icon-circle + section title
  const header = (x, y, color, iconData, text) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: 0.4, h: 0.4, fill: { color } });
    s.addImage({ data: iconData, x: x + 0.1, y: y + 0.1, w: 0.2, h: 0.2 });
    s.addText(text, {
      x: x + 0.52, y: y - 0.03, w: 6.2, h: 0.46, margin: 0, valign: "middle",
      color: INK, fontFace: "Arial", fontSize: 16, bold: true, charSpacing: 1,
    });
  };

  // ════════ HERO BAND ════════
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: PW, h: 1.55, fill: { color: DEEP } });
  // accelerometer-signal motif along the hero's lower edge (on-brand)
  const baseY = 1.55, bw = 0.06, pitch = 0.135;
  for (let i = 0, x = 0.04; x < PW; i++, x += pitch) {
    const h = 0.08 + 0.26 * Math.abs(Math.sin(i * 0.7) * 0.6 + Math.sin(i * 0.27) * 0.5);
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: baseY - h, w: bw, h, fill: { color: MINT, transparency: 74 },
    });
  }
  s.addText("UCL ENGF0031  ·  SCENARIO 2  ·  LIVE DEMO", {
    x: M, y: 0.26, w: CW, h: 0.3, margin: 0, color: MINT, fontFace: "Arial",
    fontSize: 11.5, bold: true, charSpacing: 3, align: "left",
  });
  s.addText("Cadence — Demo Run Card", {
    x: M, y: 0.5, w: CW, h: 0.66, margin: 0, color: "FFFFFF", fontFace: "Arial",
    fontSize: 32, bold: true, align: "left",
  });
  s.addText(
    "Freeze-of-gait monitor  ·  one ankle accelerometer  ·  on-body OLED + wireless dashboard",
    { x: M, y: 1.14, w: CW, h: 0.34, margin: 0, color: ICE, fontFace: "Arial",
      fontSize: 12, align: "left" });

  // ════════ SECTION 1 — RUN IT (numbered) ════════
  card(M, 1.78, CW, 3.18);
  header(M + 0.25, 1.96, AMBER, I.run, "RUN IT  —  in this order");
  const steps = [
    "Power both boards: CPX on its LiPo, ESP32 on a USB power bank. The CPX OLED shows the CADENCE splash.",
    "On the laptop, join Wi-Fi  Cadence-Hub  (password  cadence123), then open  http://192.168.4.1/",
    "Stand still and press button A on the CPX.  \"CALIB\" shows for ~4 s, then the movement floor is set.",
    "Walk normally  →  green LED ring + \"WALKING\" on the OLED + a live accel trace on the dashboard.",
    "Provoke a freeze: buzz the foot fast in place — don't step  →  red ring + inverted \"FREEZE\", and  freezes: N  ticks up.",
  ];
  let yy = 2.52;
  const stepH = 0.475;
  steps.forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x: M + 0.27, y: yy, w: 0.3, h: 0.3, fill: { color: TEAL } });
    s.addText(String(i + 1), {
      x: M + 0.27, y: yy, w: 0.3, h: 0.3, margin: 0, align: "center", valign: "middle",
      color: "FFFFFF", fontFace: "Arial", fontSize: 13, bold: true,
    });
    s.addText(t, {
      x: M + 0.72, y: yy - 0.04, w: CW - 1.05, h: stepH, margin: 0, valign: "middle",
      color: INK, fontFace: "Arial", fontSize: 11.5, align: "left", lineSpacingMultiple: 1.0,
    });
    yy += stepH;
  });

  // ════════ SECTION 2 — two columns: on-body + wireless ════════
  const colY = 5.16, colH = 2.62, colW = (CW - 0.26) / 2;
  const colXL = M, colXR = M + colW + 0.26;

  // helper: bullet list inside a card
  const bullets = (x, y, w, items) => s.addText(
    items.map((it, i) => ({
      text: it.t,
      options: {
        bullet: { code: "2022", indent: 12 }, color: it.c || INK,
        fontFace: "Arial", fontSize: 10.6, bold: !!it.b, breakLine: true,
        paraSpaceAfter: 5,
      },
    })),
    { x, y, w, h: colH - 0.7, margin: 0, valign: "top", align: "left" });

  card(colXL, colY, colW, colH);
  header(colXL + 0.2, colY + 0.18, TEAL, I.body, "ON THE BODY (CPX)");
  bullets(colXL + 0.26, colY + 0.74, colW - 0.5, [
    { t: "OLED, big:  WALKING / STILL / FREEZE", b: true },
    { t: "FI  x.xx / 1.82   (ratio vs threshold)" },
    { t: "E  energy / floor   (the movement gate)" },
    { t: "freezes: N   (confirmed-onset count)" },
    { t: "LED ring:  green = walk,  teal = still" },
    { t: "blue = calibrating,   red = FREEZE", c: MUTE },
  ]);

  card(colXR, colY, colW, colH);
  header(colXR + 0.2, colY + 0.18, BLUE, I.wifi, "WIRELESS (ESP32)");
  bullets(colXR + 0.26, colY + 0.74, colW - 0.5, [
    { t: "Hotspot:  Cadence-Hub", b: true },
    { t: "Password:  cadence123" },
    { t: "Open:  http://192.168.4.1/", b: true, c: BLUE },
    { t: "Dashboard: trace, freeze state, HR/SpO2" },
    { t: "ESP32 LED: solid = browser on, blink = wait" },
    { t: "Finger on the MAX30102 for HR/SpO2", c: MUTE },
  ]);

  // ════════ SECTION 3 — troubleshooting ════════
  const tY = 8.0, tH = 2.55;
  card(M, tY, CW, tH);
  header(M + 0.25, tY + 0.18, "C0603A", I.warn, "IF IT MISBEHAVES");
  // troubleshooting rows (label in colour, fix in ink) — one paragraph each
  const trb = [
    ["OLED blank", "check SDA=A5 / SCL=A4 and a shared GND; some panels are at 0x3D, not 0x3C."],
    ["Dashboard won't load", "be joined to Cadence-Hub (not your usual Wi-Fi); it has no internet, which is normal."],
    ["Stuck on STILL / never FREEZE", "a clean stop reads STILL — buzz the foot fast in place; if still nothing, re-calibrate (button A while truly still)."],
    ["FREEZE while standing still", "re-calibrate; the movement floor is set too low for this wearer/surface."],
    ["HR / SpO2 blank", "rest a fingertip on the MAX30102 and hold still ~10 s for a valid reading."],
  ];
  s.addText(
    trb.map(([k, v], i) => ([
      { text: k + "  —  ", options: { color: "C0603A", bold: true, fontFace: "Arial", fontSize: 10.8 } },
      { text: v, options: { color: INK, fontFace: "Arial", fontSize: 10.8, breakLine: true, paraSpaceAfter: 7 } },
    ])).flat(),
    { x: M + 0.28, y: tY + 0.74, w: CW - 0.56, h: tH - 0.9, margin: 0, valign: "top", align: "left" });

  // ════════ FOOTER ════════
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: PH - 0.62, w: PW, h: 0.62, fill: { color: DEEP } });
  s.addText(
    "Research prototype — not a medical device.   Detector: Freeze Index (Moore 2008) + movement gate (Bachlin 2010); the CNN is an offline benchmark.",
    { x: M, y: PH - 0.55, w: CW, h: 0.46, margin: 0, valign: "middle", color: ICE,
      fontFace: "Arial", fontSize: 9.5, align: "left" });

  await pres.writeFile({ fileName: "cadence_benchcard.pptx" });
  console.log("wrote cadence_benchcard.pptx");
})();
