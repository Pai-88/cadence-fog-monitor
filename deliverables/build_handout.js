/* Cadence — 2-page A4 results handout to print for teammates.
 *   page 1: "does it work?" — headline metrics, confusion matrix, baselines
 *   page 2: model interpretability — the 4 fog_plots PNGs + the takeaway
 * Build:  NODE_PATH=$(npm root -g) node build_handout.js
 * Matches the leaflet's palette so the printed set looks cohesive.
 */
const pptxgen = require("pptxgenjs");

const DEEP = "0B3B4A", TEAL = "0E7C86", BLUE = "1C7293", AMBER = "F2A541",
      MINT = "2BB7A3", BG = "F5F8F8", CARD = "FFFFFF", INK = "12343B",
      MUTE = "5E7378", ICE = "CFE8EA", RED = "E15554";
const PLOTS = "/Users/paing/Documents/scenario2_pd/colab/fog_plots/";
const shadow = () => ({ type: "outer", color: "0B3B4A", blur: 7, offset: 3, angle: 90, opacity: 0.12 });

const pres = new pptxgen();
pres.defineLayout({ name: "A4P", width: 8.27, height: 11.69 });
pres.layout = "A4P";
pres.author = "UCL ENGF0031 Scenario 2";
pres.title = "Cadence — FoG detector results";
const PW = 8.27, PH = 11.69, M = 0.55, CW = PW - 2 * M;

function hero(s, kicker, title, sub, titleSize = 30) {
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: PW, h: 1.35, fill: { color: DEEP } });
  s.addText(kicker, { x: M, y: 0.26, w: CW, h: 0.28, margin: 0, color: MINT, fontFace: "Arial",
    fontSize: 11, bold: true, charSpacing: 3 });
  s.addText(title, { x: M, y: 0.5, w: CW, h: 0.55, margin: 0, color: "FFFFFF", fontFace: "Arial",
    fontSize: titleSize, bold: true });
  s.addText(sub, { x: M, y: 1.04, w: CW, h: 0.28, margin: 0, color: ICE, fontFace: "Arial",
    fontSize: 11.5 });
}
function footer(s, pageTxt) {
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: PH - 0.62, w: PW, h: 0.62, fill: { color: DEEP } });
  s.addText("Cadence — closed-loop gait garment", { x: M, y: PH - 0.62, w: 4.0, h: 0.62, margin: 0,
    color: ICE, fontFace: "Arial", fontSize: 10, bold: true, valign: "middle" });
  s.addText(pageTxt, { x: PW - M - 4.0, y: PH - 0.62, w: 4.0, h: 0.62, margin: 0, color: MINT,
    fontFace: "Arial", fontSize: 9.5, align: "right", valign: "middle" });
}
function sectionLabel(s, txt, y) {
  s.addText(txt, { x: M, y, w: CW, h: 0.3, margin: 0, color: INK, fontFace: "Arial",
    fontSize: 13.5, bold: true, charSpacing: 1.5 });
}

// ════════════════ PAGE 1 — RESULTS ════════════════
const s1 = pres.addSlide();
s1.background = { color: BG };
hero(s1, "REAL PATIENT DATA  ·  LEAVE-ONE-SUBJECT-OUT", "Does the freeze detector work?",
  "FoGNet 1-D CNN  ·  Daphnet (10 Parkinson's patients)  ·  ankle accelerometer", 28);

// headline stat callouts
const stats = [
  { n: "0.71", l: "of real freezes caught\n(sensitivity)", c: TEAL },
  { n: "0.84", l: "of walking correctly cleared\n(specificity)", c: BLUE },
  { n: "0.78", l: "balanced accuracy\n(sens + spec) / 2", c: AMBER },
];
const gap = 0.22, sw = (CW - 2 * gap) / 3, sy = 1.62;
stats.forEach((st, i) => {
  const x = M + i * (sw + gap);
  s1.addShape(pres.shapes.RECTANGLE, { x, y: sy, w: sw, h: 1.12, fill: { color: CARD }, shadow: shadow() });
  s1.addShape(pres.shapes.RECTANGLE, { x, y: sy, w: sw, h: 0.08, fill: { color: st.c } });
  s1.addText(st.n, { x, y: sy + 0.14, w: sw, h: 0.55, margin: 0, color: st.c, fontFace: "Arial",
    fontSize: 36, bold: true, align: "center" });
  s1.addText(st.l, { x: x + 0.1, y: sy + 0.68, w: sw - 0.2, h: 0.4, margin: 0, color: MUTE,
    fontFace: "Arial", fontSize: 10, align: "center", lineSpacingMultiple: 0.95 });
});

// confusion matrix (left)
sectionLabel(s1, "CONFUSION MATRIX  ·  n=8982", 3.05);
const x0 = M, y0 = 3.5, lw = 0.95, cwid = 1.22, hh = 0.46, vh = 0.92;
s1.addText([{ text: "true ", options: { color: MUTE } }, { text: "↓", options: { color: INK, bold: true } },
  { text: "   pred ", options: { color: MUTE } }, { text: "→", options: { color: INK, bold: true } }],
  { x: x0, y: y0, w: lw, h: hh, margin: 2, fontFace: "Arial", fontSize: 7.5, align: "center", valign: "middle" });
[["no-freeze", x0 + lw], ["freeze", x0 + lw + cwid]].forEach(([t, x]) => {
  s1.addShape(pres.shapes.RECTANGLE, { x, y: y0, w: cwid, h: hh, fill: { color: DEEP } });
  s1.addText(t, { x, y: y0, w: cwid, h: hh, margin: 0, color: "FFFFFF", fontFace: "Arial", fontSize: 10.5, bold: true, align: "center", valign: "middle" });
});
[["no-freeze", y0 + hh], ["freeze", y0 + hh + vh]].forEach(([t, y]) => {
  s1.addShape(pres.shapes.RECTANGLE, { x: x0, y, w: lw, h: vh, fill: { color: BLUE } });
  s1.addText(t, { x: x0, y, w: lw, h: vh, margin: 0, color: "FFFFFF", fontFace: "Arial", fontSize: 10.5, bold: true, align: "center", valign: "middle" });
});
const cells = [
  { x: x0 + lw, y: y0 + hh, fill: "E7EEEE", n: "6842", c: INK, sub: "correct walk" },
  { x: x0 + lw + cwid, y: y0 + hh, fill: "FCEFD6", n: "1285", c: "A9742A", sub: "false alarm" },
  { x: x0 + lw, y: y0 + hh + vh, fill: "FBE3E1", n: "248", c: RED, sub: "MISSED freeze" },
  { x: x0 + lw + cwid, y: y0 + hh + vh, fill: "DCF1E8", n: "607", c: TEAL, sub: "caught freeze" },
];
cells.forEach((cl) => {
  s1.addShape(pres.shapes.RECTANGLE, { x: cl.x, y: cl.y, w: cwid, h: vh, fill: { color: cl.fill }, line: { color: "FFFFFF", width: 2 } });
  s1.addText([{ text: cl.n + "\n", options: { fontSize: 20, bold: true, color: cl.c, breakLine: true } },
    { text: cl.sub, options: { fontSize: 8.5, color: MUTE } }],
    { x: cl.x, y: cl.y, w: cwid, h: vh, margin: 0, fontFace: "Arial", align: "center", valign: "middle" });
});

// baselines (right)
s1.addText("VERSUS BASELINES", { x: 4.5, y: 3.05, w: 3.2, h: 0.3, margin: 0, color: INK,
  fontFace: "Arial", fontSize: 13.5, bold: true, charSpacing: 1.5 });
s1.addTable([
  [{ text: "Approach", options: { bold: true, color: "FFFFFF", fill: { color: DEEP } } },
   { text: "Sens", options: { bold: true, color: "FFFFFF", fill: { color: DEEP }, align: "center" } },
   { text: "Spec", options: { bold: true, color: "FFFFFF", fill: { color: DEEP }, align: "center" } }],
  ["FoGNet (CNN)", { text: "0.71", options: { align: "center", bold: true, color: TEAL } }, { text: "0.84", options: { align: "center", bold: true, color: TEAL } }],
  ["Freeze-Index", { text: "0.002", options: { align: "center", color: RED } }, { text: "0.994", options: { align: "center" } }],
  ["Always 'no-freeze'", { text: "0.00", options: { align: "center", color: RED } }, { text: "1.00", options: { align: "center" } }],
], { x: 4.5, y: 3.5, w: 3.2, colW: [1.7, 0.75, 0.75], rowH: [0.34, 0.34, 0.34, 0.34], fontFace: "Arial",
  fontSize: 10.5, color: INK, border: { type: "solid", color: "D7E2E2", pt: 1 }, valign: "middle" });
s1.addText("Each held-out patient is scored once, then pooled — no patient is in both train and test.",
  { x: 4.5, y: 5.0, w: 3.2, h: 0.6, margin: 0, color: MUTE, fontFace: "Arial", fontSize: 10, italic: true, lineSpacingMultiple: 1.05 });

// why-not-accuracy callout
s1.addShape(pres.shapes.RECTANGLE, { x: M, y: 6.05, w: CW, h: 0.92, fill: { color: "FBF1DD" } });
s1.addShape(pres.shapes.RECTANGLE, { x: M, y: 6.05, w: 0.1, h: 0.92, fill: { color: AMBER } });
s1.addText([
  { text: "Why not “accuracy”?  ", options: { bold: true, color: "9A6A1E" } },
  { text: "A model that NEVER calls a freeze scores 90.5% accuracy — yet catches zero freezes and prevents zero falls. A missed freeze can mean a fall, so sensitivity & specificity are the honest metrics here, not accuracy.",
    options: { color: INK } },
], { x: M + 0.28, y: 6.05, w: CW - 0.5, h: 0.92, margin: 0, fontFace: "Arial", fontSize: 11.5, valign: "middle", lineSpacingMultiple: 1.05 });

// dataset facts card
sectionLabel(s1, "DATASET & METHOD", 7.2);
s1.addShape(pres.shapes.RECTANGLE, { x: M, y: 7.55, w: CW, h: 1.35, fill: { color: CARD }, shadow: shadow() });
s1.addText([
  { text: "8982 four-second windows (2 s hop), one ankle accelerometer @ 64 Hz", options: { bullet: true, breakLine: true } },
  { text: "Freeze prevalence 9.5%  (855 freeze / 8127 walk) — a rare, safety-critical event", options: { bullet: true, breakLine: true } },
  { text: "Leave-One-Subject-Out across all 10 patients; metrics pooled over held-out windows", options: { bullet: true, breakLine: true } },
  { text: "S04 & S10 have no freeze episodes → used for training only (sensitivity undefined as a test fold)", options: { bullet: true } },
], { x: M + 0.25, y: 7.65, w: CW - 0.5, h: 1.15, margin: 0, color: INK, fontFace: "Arial", fontSize: 11, paraSpaceAfter: 5 });

// per-subject spread
s1.addText([
  { text: "Per-patient sensitivity spans 0.45–0.98 ", options: { bold: true, color: INK } },
  { text: "(wide between-subject variability is expected for real Parkinson's gait; the pooled 0.71 is the headline).", options: { color: MUTE } },
], { x: M, y: 9.1, w: CW, h: 0.5, margin: 0, fontFace: "Arial", fontSize: 10.5, lineSpacingMultiple: 1.05 });

footer(s1, "Results  ·  page 1 of 2");

// ════════════════ PAGE 2 — INTERPRETABILITY ════════════════
const s2 = pres.addSlide();
s2.background = { color: BG };
hero(s2, "MODEL INTERPRETABILITY", "Which signals flag a freeze?",
  "RandomForest over 9 engineered features  ·  held-out patient S05", 28);

// takeaway callout
s2.addShape(pres.shapes.RECTANGLE, { x: M, y: 1.5, w: CW, h: 0.78, fill: { color: "E7F4F0" } });
s2.addShape(pres.shapes.RECTANGLE, { x: M, y: 1.5, w: 0.1, h: 0.78, fill: { color: TEAL } });
s2.addText([
  { text: "Takeaway:  ", options: { bold: true, color: "1B5E54" } },
  { text: "both SHAP and permutation importance rank the classical Freeze-Index ", options: { color: INK } },
  { text: "last", options: { bold: true, color: RED } },
  { text: " of 9 features — so the dead FI baseline is explained, not just observed. Movement dynamics (jerk, dominant frequency, magnitude) carry the signal.", options: { color: INK } },
], { x: M + 0.28, y: 1.5, w: CW - 0.5, h: 0.78, margin: 0, fontFace: "Arial", fontSize: 11.5, valign: "middle", lineSpacingMultiple: 1.05 });

function plotBlock(x, y, w, img, cap) {
  const h = w / 1.5;
  s2.addShape(pres.shapes.RECTANGLE, { x: x - 0.06, y: y - 0.06, w: w + 0.12, h: h + 0.12, fill: { color: CARD }, shadow: shadow() });
  s2.addImage({ path: img, x, y, w, h });
  s2.addText(cap, { x, y: y + h + 0.02, w, h: 0.22, margin: 0, color: MUTE, fontFace: "Arial", fontSize: 9.5, align: "center", italic: true });
  return h;
}
// row of two ranking plots (symmetric across the content width)
const rw = 3.38, ry = 2.55;
plotBlock(M, ry, rw, PLOTS + "interp_shap_bar.png", "SHAP — mean impact per feature");
plotBlock(M + rw + 0.41, ry, rw, PLOTS + "interp_perm_importance.png", "Permutation importance — accuracy drop");
// beeswarm centered
const bw = 3.8, by = ry + rw / 1.5 + 0.55;
plotBlock((PW - bw) / 2, by, bw, PLOTS + "interp_shap_beeswarm.png", "SHAP beeswarm — red = high feature value");
// partial dependence (wide aspect 3.6:1), centered with clear space above the footer
const pw2 = 7.0, ph2 = pw2 / 3.6, px2 = (PW - pw2) / 2, py = by + bw / 1.5 + 0.55;
s2.addShape(pres.shapes.RECTANGLE, { x: px2 - 0.06, y: py - 0.06, w: pw2 + 0.12, h: ph2 + 0.12, fill: { color: CARD }, shadow: shadow() });
s2.addImage({ path: PLOTS + "interp_partial_dependence.png", x: px2, y: py, w: pw2, h: ph2 });
s2.addText("Partial dependence — P(freeze) vs. the three most influential features",
  { x: px2, y: py + ph2 + 0.06, w: pw2, h: 0.22, margin: 0, color: MUTE, fontFace: "Arial", fontSize: 9.5, align: "center", italic: true });

footer(s2, "Interpretability  ·  page 2 of 2");

pres.writeFile({ fileName: "results_handout.pptx" }).then((f) => console.log("wrote", f));
