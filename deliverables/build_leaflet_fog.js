/* Cadence — one-page A4 advert/leaflet for the ENGF0031 Parkinson's gait garment.
 * Build:  NODE_PATH=$(npm root -g) node build_leaflet_fog.js
 * Output: cadence_leaflet_fog.pptx  (single A4-portrait slide)
 *
 * Honest framing: this is a student research prototype, NOT a medical device.
 * The copy is grounded in the project's verified facts: a freeze-of-gait MONITOR
 * (sense + detect + on-body OLED display), 64 Hz single ankle accelerometer,
 * Daphnet LOSO validation. No vibration cue — this build reports freezes on a
 * screen for a carer/clinic, it does not deliver a cueing therapy.
 */
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const {
  FaWaveSquare, FaShoePrints, FaDesktop, FaBrain, FaMicrochip,
  FaArrowRight, FaCheckCircle,
} = require("react-icons/fa");

// ── palette (clinical teal, print-friendly; NO "#" prefixes) ──
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
    monitor: await icon(FaWaveSquare, "#FFFFFF"),
    detect:  await icon(FaShoePrints, "#FFFFFF"),
    cue:     await icon(FaDesktop, "#FFFFFF"),
    sense:   await icon(FaMicrochip, "#0E7C86"),
    brain:   await icon(FaBrain, "#1C7293"),
    bell:    await icon(FaDesktop, "#F2A541"),
    arrow:   await icon(FaArrowRight, "#0E7C86"),
    check:   await icon(FaCheckCircle, "#2BB7A3"),
  };

  const pres = new pptxgen();
  pres.defineLayout({ name: "A4P", width: 8.27, height: 11.69 });
  pres.layout = "A4P";
  pres.author = "UCL ENGF0031 Scenario 2";
  pres.title = "Cadence — Parkinson's gait garment";
  const s = pres.addSlide();
  s.background = { color: BG };

  const PW = 8.27, M = 0.55, CW = PW - 2 * M;   // page width, margin, content width

  // ════════ HERO BAND ════════
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: PW, h: 2.55, fill: { color: DEEP } });
  // accelerometer-signal motif along the hero's lower edge (subtle, on-brand)
  const baseY = 2.55, bw = 0.06, pitch = 0.135;
  for (let i = 0, x = 0.04; x < PW; i++, x += pitch) {
    const h = 0.10 + 0.30 * Math.abs(Math.sin(i * 0.7) * 0.6 + Math.sin(i * 0.27) * 0.5);
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: baseY - h, w: bw, h, fill: { color: MINT, transparency: 74 },
    });
  }
  s.addText("UCL ENGF0031  ·  SMART CLOTHING PROTOTYPE", {
    x: M, y: 0.34, w: CW, h: 0.3, margin: 0, color: MINT, fontFace: "Arial",
    fontSize: 12, bold: true, charSpacing: 3, align: "left",
  });
  s.addText("Cadence", {
    x: M, y: 0.6, w: CW, h: 1.0, margin: 0, color: "FFFFFF", fontFace: "Arial",
    fontSize: 60, bold: true, align: "left",
  });
  s.addText(
    "A wearable that catches a Parkinson's gait freeze and shows it the instant it happens — on the body, on screen.",
    { x: M, y: 1.62, w: 6.7, h: 0.62, margin: 0, color: ICE, fontFace: "Arial",
      fontSize: 15.5, align: "left", lineSpacingMultiple: 1.05 });

  // ════════ 3-IN-1 CARDS ════════
  s.addText("ONE SENSOR.  THREE JOBS.", {
    x: M, y: 2.78, w: CW, h: 0.32, margin: 0, color: INK, fontFace: "Arial",
    fontSize: 15, bold: true, charSpacing: 2,
  });
  const cardY = 3.2, cardH = 2.28, gap = 0.22;
  const cardW = (CW - 2 * gap) / 3;
  const cards = [
    { ac: TEAL, ic: I.monitor, t: "MONITOR",
      d: "Tracks walking activity and movement energy continuously from one ankle accelerometer." },
    { ac: BLUE, ic: I.detect, t: "DETECT",
      d: "Spots freezing of gait with a model trained and validated on real patient recordings." },
    { ac: AMBER, ic: I.cue, t: "DISPLAY",
      d: "Shows the live freeze state on an on-body OLED and logs every episode for the clinic." },
  ];
  cards.forEach((c, i) => {
    const x = M + i * (cardW + gap);
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: cardY, w: cardW, h: cardH, fill: { color: CARD }, shadow: shadow(),
    });
    s.addShape(pres.shapes.RECTANGLE, { x, y: cardY, w: cardW, h: 0.09, fill: { color: c.ac } });
    // icon in a colored circle, top-left
    const cx = x + 0.22, cy = cardY + 0.28, d = 0.66;
    s.addShape(pres.shapes.OVAL, { x: cx, y: cy, w: d, h: d, fill: { color: c.ac } });
    s.addImage({ data: c.ic, x: cx + 0.17, y: cy + 0.17, w: 0.32, h: 0.32 });
    s.addText(c.t, {
      x: x + 0.22, y: cardY + 1.06, w: cardW - 0.44, h: 0.34, margin: 0,
      color: INK, fontFace: "Arial", fontSize: 16.5, bold: true, charSpacing: 1, align: "left",
    });
    s.addText(c.d, {
      x: x + 0.22, y: cardY + 1.42, w: cardW - 0.44, h: 0.75, margin: 0,
      color: MUTE, fontFace: "Arial", fontSize: 11.5, align: "left", valign: "top",
      lineSpacingMultiple: 1.05,
    });
  });

  // ════════ HOW IT WORKS — closed loop ════════
  const loopY = 5.92;
  s.addText("HOW IT WORKS — FROM ANKLE TO SCREEN", {
    x: M, y: loopY, w: CW, h: 0.32, margin: 0, color: INK, fontFace: "Arial",
    fontSize: 15, bold: true, charSpacing: 2,
  });
  const stripY = loopY + 0.44, stripH = 1.18;
  s.addShape(pres.shapes.RECTANGLE, {
    x: M, y: stripY, w: CW, h: stripH, fill: { color: "EAF2F2" },
  });
  // three nodes with arrows between
  const nodes = [
    { ic: I.sense, t: "SENSE", d: "64 Hz · ankle" },
    { ic: I.brain, t: "DETECT", d: "freeze? (model)" },
    { ic: I.bell, t: "DISPLAY", d: "OLED + log" },
  ];
  const nodeW = 1.55, nodeH = 0.86;
  const slots = 3, totalNodes = slots * nodeW;
  const arrowW = (CW - 0.5 - totalNodes) / (slots - 1);  // 0.5 padding inside strip
  let nx = M + 0.25;
  const nodeY = stripY + (stripH - nodeH) / 2;
  nodes.forEach((n, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: nx, y: nodeY, w: nodeW, h: nodeH, rectRadius: 0.08,
      fill: { color: CARD }, line: { color: "CDE0E0", width: 1 },
    });
    s.addImage({ data: n.ic, x: nx + 0.16, y: nodeY + 0.25, w: 0.36, h: 0.36 });
    s.addText([
      { text: n.t + "\n", options: { bold: true, color: INK, fontSize: 12.5, breakLine: true } },
      { text: n.d, options: { color: MUTE, fontSize: 9.5 } },
    ], { x: nx + 0.6, y: nodeY + 0.06, w: nodeW - 0.66, h: nodeH - 0.12, margin: 0,
        fontFace: "Arial", align: "left", valign: "middle" });
    if (i < nodes.length - 1) {
      s.addImage({ data: I.arrow, x: nx + nodeW + arrowW / 2 - 0.13, y: nodeY + nodeH / 2 - 0.13, w: 0.26, h: 0.26 });
    }
    nx += nodeW + arrowW;
  });
  s.addText(
    "Each freeze shows on-screen the instant it is detected, and is logged for review — one sensor in, one clear answer out.",
    { x: M, y: stripY + stripH + 0.08, w: CW, h: 0.3, margin: 0, color: MUTE,
      fontFace: "Arial", fontSize: 10.5, italic: true, align: "left" });

  // ════════ STAT CALLOUTS ════════
  const statY = 8.18;
  const stats = [
    { n: "≈50%", l: "of people with Parkinson's\nexperience freezing of gait*", c: TEAL },
    { n: "64 Hz", l: "single ankle\naccelerometer", c: BLUE },
    { n: "~2 s", l: "freeze re-checked,\nshown on the body", c: AMBER },
  ];
  const stW = (CW - 2 * gap) / 3;
  stats.forEach((st, i) => {
    const x = M + i * (stW + gap);
    s.addShape(pres.shapes.RECTANGLE, { x, y: statY, w: stW, h: 1.0, fill: { color: CARD }, shadow: shadow() });
    s.addText(st.n, { x, y: statY + 0.1, w: stW, h: 0.5, margin: 0, color: st.c,
      fontFace: "Arial", fontSize: 30, bold: true, align: "center" });
    s.addText(st.l, { x: x + 0.1, y: statY + 0.58, w: stW - 0.2, h: 0.38, margin: 0, color: MUTE,
      fontFace: "Arial", fontSize: 10, align: "center", lineSpacingMultiple: 0.95 });
  });
  s.addText("*Estimates vary with disease stage and assessment; commonly cited around 50%.", {
    x: M, y: statY + 1.04, w: CW, h: 0.22, margin: 0, color: MUTE, fontFace: "Arial",
    fontSize: 8.5, italic: true, align: "left",
  });

  // ════════ POSITIONING + VALIDATION ════════
  const posY = 9.62;
  s.addText([
    { text: "In good company.  ", options: { bold: true, color: INK, fontSize: 11 } },
    { text: "Builds on ideas demonstrated by Apple Watch's Movement Disorder API, PDMonitor and STAT-ON — clinical-grade Parkinson's monitors — in one low-cost wearable.",
      options: { color: MUTE, fontSize: 11 } },
  ], { x: M, y: posY, w: CW, h: 0.5, margin: 0, fontFace: "Arial", align: "left", lineSpacingMultiple: 1.05 });

  const badgeY = 10.26;
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: badgeY, w: CW, h: 0.44, fill: { color: "E7F4F0" } });
  s.addImage({ data: I.check, x: M + 0.16, y: badgeY + 0.1, w: 0.24, h: 0.24 });
  s.addText("Validated on real patient recordings (Daphnet) using leave-one-subject-out testing.", {
    x: M + 0.5, y: badgeY, w: CW - 0.6, h: 0.44, margin: 0, color: "1B5E54", fontFace: "Arial",
    fontSize: 11, bold: true, align: "left", valign: "middle",
  });

  // ════════ FOOTER BAND ════════
  const footY = 10.92;
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: footY, w: PW, h: PW * 0 + (11.69 - footY), fill: { color: DEEP } });
  s.addText("Cadence — wearable freeze-of-gait monitor", {
    x: M, y: footY, w: 3.4, h: 11.69 - footY, margin: 0, color: ICE, fontFace: "Arial",
    fontSize: 10, bold: true, align: "left", valign: "middle",
  });
  s.addText([
    { text: "Research prototype — not a medical device", options: { breakLine: true } },
    { text: "UCL ENGF0031 Scenario 2  ·  2026" },
  ], { x: 4.05, y: footY, w: PW - M - 4.05, h: 11.69 - footY, margin: 0, color: ICE, fontFace: "Arial",
    fontSize: 9.5, align: "right", valign: "middle", lineSpacingMultiple: 1.1 });

  await pres.writeFile({ fileName: "cadence_leaflet_fog.pptx" });
  console.log("wrote cadence_leaflet_fog.pptx");
})();
