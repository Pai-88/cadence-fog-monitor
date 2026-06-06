export const meta = {
  name: 'cadence-report',
  description: 'Draft a ~50pp Cadence technical report: 16 sections, each self-compiled + visually self-checked',
  phases: [
    { title: 'Draft', detail: 'one agent per section: write LaTeX, compile vs preamble, render + eyeball diagrams, fix' },
  ],
}

// ---------------------------------------------------------------------------
// Shared ground-truth FACTS (every number REAL — agents must not invent any).
// String.raw keeps LaTeX backslashes literal.
// ---------------------------------------------------------------------------
const FACTS = String.raw`
=== CADENCE — GROUND-TRUTH FACTS (use ONLY these numbers; invent nothing) ===

PROJECT: "Cadence" is a low-cost, ankle-worn wearable that MONITORS freeze-of-gait
(FoG) in Parkinson's disease. Module: UCL ENGF0031 Scenario 2 "Smart Clothing". It
DETECTS and DISPLAYS freeze episodes — it is a MONITOR, NOT a therapy. A coin
vibration motor is fitted but DISABLED in firmware: the device reports freezes on an
OLED + NeoPixel LEDs + a live web dashboard; it does NOT buzz/cue. Never describe it
as delivering a cue or therapy.

CLINICAL: Freeze of gait = a sudden, brief inability to step ("feet glued to the
floor"), common in mid/late Parkinson's and a major fall risk. The tell-tale
signature is rapid 3-8 Hz trembling/shuffling of the feet IN PLACE (not standing
still). Episodes last seconds. Cadence uses a single ankle accelerometer.

SENSOR: Adafruit Circuit Playground Express (CPX), SAMD21 MCU, onboard LIS3DH 3-axis
MEMS accelerometer. Sampled at 64 Hz, range +/-8 g (heel-strike exceeds +/-2 g, so
+/-2 g would clip). Units: milli-g (mg) on the host, m/s^2 on-device. Nyquist 32 Hz
comfortably covers every band of interest.

MAGNITUDE: the 3 axes are combined into one orientation-invariant scalar
a = sqrt(ax^2 + ay^2 + az^2); the mean is subtracted (removes gravity / DC) before
spectral analysis.

SPECTRAL: Welch power spectral density (Hann-tapered) over a 256-sample (4.0 s)
window; bin spacing df = fs/N = 64/256 = 0.25 Hz. Bands: locomotor 0.5-3 Hz, freeze
3-8 Hz, rest-tremor 4-6 Hz. Band power = sum of the PSD over that band's bins.

FREEZE INDEX (Moore et al. 2008): FI = power(3-8 Hz) / power(0.5-3 Hz) on the
mean-removed magnitude. It is a RATIO, hence scale-invariant (mg vs g cancels). A
tiny 1e-9 is added to the denominator to avoid divide-by-zero. REAL per-phase FI on
the worksheet capture (capture1.csv, 64 Hz, 41.4 s): still ~1.1 (max 1.71); walk
~1.2 (max 2.08); freeze mean 6.89, median 4.45, max 21.35.

WINDOWING: 256-sample (4 s) windows, 128-sample (2 s) hop -> 50% overlap -> one
decision every 2 s. The long window buys spectral resolution for the 3-8 Hz band;
FoG lasts seconds so this is appropriate. A freeze shorter than the window, or split
across a window boundary, can be diluted/missed.

SHIPPED DETECTOR (the glass box): a window is FREEZE iff
(movement_energy > still_floor) AND (FI > FI_THRESHOLD), then debounced. Deployed
FI_THRESHOLD = 1.815 (biased to sensitivity). The clean sensitivity+specificity
(Youden) optimum on capture1 is 2.10. REAL threshold sweep on capture1 (19 windows:
6 freeze, 13 non-freeze): at FI>1.815 -> TP=5, FP=1, TN=12, FN=1 -> sensitivity
0.83, specificity 0.92; at FI>2.10 -> TP=5, FP=0, TN=13, FN=1 -> sensitivity 0.83,
specificity 1.00. The board ships 1.815 because a MISSED freeze can mean a fall.
(A library default constant elsewhere is 2.0; the DEPLOYED value is 1.815.)

MOVEMENT ENERGY + GATE: movement_energy = band_power(0.5-3) + band_power(3-8) of the
magnitude — NOT scale-invariant; it measures how much the leg is actually moving. A
still leg has near-zero energy but noise/noise can spike FI, so a STILL-floor gate
forces STILL when energy < still_floor (auto-calibrated at boot from resting energy).
State machine: STILL if not moving; else FREEZE if FI>threshold; else WALKING.

DEBOUNCE: a new state must persist 2 consecutive control windows before it is
committed (assert-after-2, release-after-2), so a one-off bump never flips the verdict.

CNN (the "given more time" upgrade — NOT the shipped detector): FoGNet, a 1-D CNN,
~18k parameters, input a raw 3x256 window (3 axes x 4 s at 64 Hz). Architecture:
Conv1d(3->16, kernel 15) -> BatchNorm -> ReLU -> maxpool x2; Conv1d(16->32, kernel 9)
-> BN -> ReLU -> maxpool x2; Conv1d(32->64, kernel 5) -> BN -> ReLU; Global Average
Pooling (collapses each of the 64 channels to one number); Linear(64->32) -> ReLU ->
Dropout(0.3) -> Linear(32->2); the two logits -> softmax -> P(freeze). Runs <1 ms on
a laptop CPU. Trained on the public Daphnet FoG dataset (Bachlin et al. 2010) with
Leave-One-Subject-Out (LOSO) cross-validation; loss = binary cross-entropy, optimised
by gradient descent / backprop. Pooled LOSO confusion (n=8982 windows): TN=6842,
FP=1285, FN=248, TP=607 -> sensitivity 0.71, specificity 0.84, accuracy 0.83.

GLASS-BOX ALTERNATIVE: a 9-feature RandomForest (interpretable). Nested-LOSO
operating point t* ~ 0.10 -> sensitivity 0.75, specificity 0.76. Threshold-free
ranking: ROC-AUC 0.82, PR-AUC 0.33.

HEADLINE METRIC: sensitivity (freezes caught) & specificity (calm gait correctly left
alone), NOT raw accuracy. Freezes are rare, so accuracy misleads: a classifier that
NEVER flags a freeze can score ~0.90 accuracy yet 0.00 sensitivity. Definitions:
sensitivity = TP/(TP+FN); specificity = TN/(TN+FP); precision = TP/(TP+FP);
accuracy = (TP+TN)/total.

HARDWARE: CPX senses + runs the on-board FI detector + drives an external SSD1306
OLED (second I2C bus, SDA=A5/SCL=A4) and the onboard NeoPixel ring (state colours) —
untethered on the ankle. Optional path: CPX --UART--> ESP32 --Wi-Fi/TCP--> laptop,
where a dashboard server renders the live display (a 3-board topology). Two firmware
roles: (1) RECORDER = cpx_fog_logger.ino — flip the SLIDE SWITCH to log accel +
auto-labelled phases to the CPX's 2 MB SPI flash untethered, then pull it over USB
with cpx_dump.py; (2) MONITOR = cpx_fog_standalone.ino — live OLED + LED freeze
display. Coin motor fitted but DISABLED. Worn in an ankle sleeve.

SOFTWARE: a torch-free Python package "fog" (config; dsp = magnitude / freeze_index /
movement_energy / band_power / AccelFilter; streaming = serial+replay receiver, ring
buffer, real-time pacing; metrics = sensitivity/specificity/confusion; normalize) +
the CNN (model.py, torch) + the firmware (Arduino/C++) + a dashboard server (Python
asyncio + websocket -> dashboard.html). DESIGN LESSON: the live FI detector reads RAW
accel — per-axis band-pass filtering BEFORE taking magnitude flattens the FI ratio
(walking and freezing look alike), so the FI path is deliberately unfiltered. 73 unit
tests pass.

LIVE DASHBOARD PANELS:
- Gait Status: the headline STILL / WALKING / FREEZE verdict (gate -> FI -> debounce).
- Freeze Margin: a meter = live FI as a % of the 1.815 trip line (100% = at/over the
  trip -> freeze); shows "—/at rest" when the gate has it STILL. A glass-box
  "how close to a freeze" confidence.
- Freeze Index: the raw ratio (walk ~0.5-1; freeze >2, up to 20+); trip line 1.815.
- Tremor Power: 4-6 Hz band power (rest-tremor band) + a present/clear badge.
- Movement Energy: 0.5-8 Hz band power (how hard the leg is moving) + the MOVING/STILL
  gate badge (above/below still_floor).
- Locomotion Power: 0.5-3 Hz band power (the walking-rhythm energy).
- Accelerometer waveform: the live raw 3-axis trace (64 Hz).
Relationships: FI = freeze-band / loco-band; movement energy = freeze-band + loco-band;
the gate uses energy; the state uses gate + FI + debounce. In glass-box (FI) mode the
CNN is bypassed (that is why an old "probability" box used to read blank; it has been
replaced by the Freeze Margin meter).

REAL FIGURES you may \includegraphics (NO path, NO extension; graphicspath is preset):
  fi_per_phase, operating_point, capture1_trace, smoketest_trace, ankle_run_trace,
  confusion_matrix, roc_curve, pr_curve, accuracy_caveat, score_hist, calibration,
  feature_dist, feature_corr, decision_contour, metrics_panel, param_grid,
  sigmoid, gradient_descent, bce_loss, train_curve, per_subject, partial_dependence,
  perm_importance, shap_bar, shap_beeswarm, results_sheet,
  wiring_pinout, sleeve_layout, coding_flowchart.

NUMBERS RULE: use ONLY the quantitative values above. Do NOT invent new accuracy
numbers, dataset sizes, parameter counts, or results. If you need a number not listed,
describe it qualitatively instead.
`

// ---------------------------------------------------------------------------
const STYLE = String.raw`
=== HOUSE STYLE (the shared preamble is already written — \input it; do NOT add
\usepackage or redefine colours) ===

Colours available: navy (primary/models), steel, teal (sensing), amber (evaluation),
purple (interpretability), freeze (red, decision), grn (walking/OK), ink, mute, and
pale backgrounds paleteal/palenavy/paleamber/palefreeze, plus mist, codebg, rule.

Callout environments (use 2-4 per section to separate intuition from rigour):
  \begin{intuition} ... \end{intuition}   (teal)
  \begin{keyidea} ... \end{keyidea}       (navy)
  \begin{mathsbox} ... \end{mathsbox}     (amber — for unpacking an equation)
  \begin{takeaway} ... \end{takeaway}     (red — the one-line "so what")

Code: \begin{lstlisting}[style=py] ... \end{lstlisting}  or  [style=cpp].
Macros: \code{...}, \FI, \norm{...}, and \cnnvol{cx}{halfw}{halfh}{frontstyle}{backstyle}
(a 3-D CNN tensor block, draw inside a tikzpicture).
TikZ libs loaded: positioning, arrows.meta, calc, shapes.geometric, shapes.misc,
backgrounds, fit, decorations.pathreplacing, decorations.markings, patterns,
shadows.blur, 3d, angles, quotes, intersections. pgfplots compat=1.18 + groupplots +
fillbetween.

VISUAL RULES (this must look WORLD-CLASS; the user specifically hates overlapping or
jittery diagrams):
- Every section MUST contain at least one figure; strong sections have 2-3. Mix REAL
  data figures (\includegraphics from the list) with ORIGINAL conceptual TikZ/pgfplots.
- Diagrams must NEVER overlap or run off the page. Give TikZ nodes explicit spacing,
  wrap wide pictures in \resizebox{0.95\textwidth}{!}{...} or use scale=, and CENTER
  each inside a figure with a \caption.
- Top level is \section{...}; you may use \subsection. Do NOT use \chapter, and do NOT
  emit any preamble, \documentclass, \begin{document}, or \input — only the body.
- Lead intuitive, then formalise. Explain to a sharp reader who is NOT a DSP/ML expert.
- Reference figures by EXACT filename, no extension: \includegraphics[width=0.8\linewidth]{roc_curve}.
- Tables: booktabs. Equations: numbered where they matter.
`

// ---------------------------------------------------------------------------
const SELFCHECK = String.raw`
=== MANDATORY SELF-VALIDATION (do this before returning) ===
Let DIR = /tmp/cad_<your section number, e.g. 05>. Then:
1. mkdir -p DIR
2. Write DIR/t.tex with EXACTLY this structure:
     \input{/Users/paing/Documents/scenario2_pd/report/preamble.tex}
     \begin{document}
     <YOUR \section{...} fragment here>
     \end{document}
3. cd DIR && tectonic t.tex   — FIX every error until it compiles to t.pdf cleanly.
4. RENDER AND LOOK:  pdftoppm -jpeg -r 110 t.pdf DIR/page
   then READ each DIR/page-*.jpg image and critically inspect YOUR OWN diagrams for:
   overlapping nodes/text, anything running off the page edge, cramped or illegible
   TikZ, a figure wider than the text block, awkward spacing. FIX and recompile until
   the pages look clean and professional. This visual check is REQUIRED, not optional.
5. Aim for roughly the target page count of dense, well-illustrated content.

Return JSON: filename (e.g. "05_fourier.tex"), latex (ONLY the \section{...} body, no
wrapper), compiled (true), pages (integer you measured), figures_used (array), notes.
`

// ---------------------------------------------------------------------------
const SECTIONS = [
  { id:'01', file:'01_aim.tex', title:'Aim & Clinical Motivation', pages:3,
    figures:['(conceptual TikZ: still->walk->FREEZE->walk gait strip)'],
    brief: String.raw`Open the report. Explain Parkinson's disease briefly, then freeze of gait (what it
    feels like, why it is dangerous — falls, lost independence), and why continuous
    monitoring from a cheap wearable matters. State Cadence's aim precisely: faithfully
    DETECT and intuitively DISPLAY freeze episodes from a single ankle accelerometer.
    Make the MONITOR-not-therapy positioning crisp (motor fitted but disabled). Contrast
    "monitor" vs "cueing therapy". Draw a conceptual gait-timeline strip
    (still -> walk -> FREEZE -> walk) as a TikZ figure. Use intuition + keyidea boxes.
    End with a short "what this report covers" roadmap.`},

  { id:'02', file:'02_overview.tex', title:'System Overview: The End-to-End Pipeline', pages:3,
    figures:['(hero TikZ: full pipeline block diagram)'],
    brief: String.raw`Give the bird's-eye view: accelerometer (64 Hz) -> magnitude -> sliding windows ->
    {FFT -> band power -> Freeze Index} and {CNN} -> movement-energy gate -> 2-window
    debounce -> state (STILL/WALKING/FREEZE) -> on-body OLED+LEDs and Wi-Fi -> laptop
    dashboard. Make clear the glass-box FI is the SHIPPED detector and the CNN is the
    upgrade. Build ONE large, polished horizontal pipeline block diagram (the report's
    hero figure) in TikZ — colour-code stages (teal sensing, navy compute, red decision)
    and keep generous spacing so nothing overlaps; wrap in \resizebox. Briefly name the
    three "boards" (CPX, ESP32, laptop). This section orients the reader for everything
    that follows.`},

  { id:'03', file:'03_sensing.tex', title:'Sensing: The Accelerometer & Data Acquisition', pages:3,
    figures:['(TikZ: ankle axes + a raw 3-axis trace)'],
    brief: String.raw`Explain the LIS3DH MEMS accelerometer: what a MEMS accelerometer physically is
    (tiny spring-mass whose deflection changes a capacitance), the 3 axes, the +/-8 g
    range and WHY (heel strike exceeds 2 g and would clip), 64 Hz sampling and Nyquist
    (32 Hz > 8 Hz so all bands are safe), units (mg / m per s^2), the gravity offset.
    Describe the fixed-rate, non-blocking sampling loop on the SAMD21 (drift-free
    scheduling). Draw the ankle with its 3 axes and a small raw 3-axis trace (TikZ or
    pgfplots). intuition + mathsbox (Nyquist) boxes.`},

  { id:'04', file:'04_magnitude.tex', title:'Acceleration Magnitude', pages:3,
    figures:['(TikZ: 3 axes collapsing into one magnitude signal; a 3-D vector + its norm)'],
    brief: String.raw`Explain WHY we collapse 3 axes into one magnitude: orientation invariance — the
    sleeve can rotate on the ankle and we must not let the answer depend on how the board
    sits. Define the Euclidean norm a = sqrt(ax^2+ay^2+az^2). Explain removing the mean
    (kills gravity and DC) so the spectrum reflects motion, not posture. Give a tiny
    worked numeric example. Draw a 3-D acceleration vector with its norm, and a figure of
    the three axes merging into one scalar trace. intuition + mathsbox + keyidea.`},

  { id:'05', file:'05_fourier.tex', title:'Fourier Analysis: From Time to Frequency', pages:4,
    figures:['(TikZ/pgfplots: a time signal -> its spectrum; a Hann window; frequency bins)'],
    brief: String.raw`Teach the Fourier idea from scratch and intuitively: any signal is a sum of
    sinusoids; the DFT/FFT decomposes a window into frequency bins; bin spacing
    df = fs/N = 0.25 Hz for N=256 at 64 Hz; the power spectrum is |X(f)|^2; Welch's
    method tapers the window (Hann) to cut spectral leakage and estimates the PSD;
    Parseval -> total power is conserved. Intuition: walking energy sits low (~1-2 Hz),
    freeze trembling sits in 3-8 Hz. Draw a conceptual time-domain wiggle with an arrow
    to its frequency spectrum (two peaks), plus a Hann-window shape, using pgfplots.
    Keep equations clean in mathsbox. This is a core teaching section — make it shine.`},

  { id:'06', file:'06_freeze_index.tex', title:'Band Power & the Freeze Index', pages:4,
    figures:['fi_per_phase','(TikZ/pgfplots: walk vs freeze spectra with the two bands shaded)'],
    brief: String.raw`Define band power as the PSD integrated over a band; introduce the locomotor band
    (0.5-3 Hz) and freeze band (3-8 Hz). Define the Freeze Index FI = P(3-8)/P(0.5-3)
    (Moore 2008). Explain WHY a ratio: scale-invariance (mg vs g cancels), it normalises
    for amplitude so only the spectral SHAPE matters. Use the REAL per-phase numbers
    (still ~1.1, walk max 2.08, freeze mean 6.89 / max 21.35) and \includegraphics the
    REAL figure fi_per_phase. Draw two conceptual spectra (walking: low band tall;
    freezing: freeze band tall) with the two bands shaded, and show the ratio flipping.
    Mention the 1e-9 guard. mathsbox + intuition + takeaway.`},

  { id:'07', file:'07_windowing.tex', title:'Real-Time Windowing', pages:3,
    figures:['(TikZ: overlapping 4 s windows with a 2 s hop along a timeline)'],
    brief: String.raw`Explain the streaming geometry: 256-sample (4 s) windows, 128-sample (2 s) hop,
    50% overlap -> a fresh decision every 2 s. Why 4 s (spectral resolution for the 3-8 Hz
    band; FoG lasts seconds). The resolution-vs-latency trade-off. The ring buffer.
    Why a freeze shorter than the window, or split across a boundary, gets diluted/missed.
    Draw a clean sliding-window timeline (several overlapping 4 s windows stepping by 2 s)
    in TikZ with non-overlapping labels. intuition + keyidea.`},

  { id:'08', file:'08_detector.tex', title:'The Shipped Detector: Threshold, Gate, Debounce', pages:5,
    figures:['operating_point','capture1_trace','(TikZ: the STILL/WALKING/FREEZE decision tree)'],
    brief: String.raw`The heart of the device. The decision rule: FREEZE iff moving (energy>still_floor)
    AND FI>threshold, then debounced. Deployed threshold 1.815 (sensitivity-biased) vs the
    clean optimum 2.10; present the REAL capture1 sweep as a booktabs table (1.815 ->
    5/1/12/1 -> 0.83/0.92; 2.10 -> 5/0/13/1 -> 0.83/1.00) and explain the bias toward
    sensitivity (a missed freeze can mean a fall). Explain the movement-energy still-gate
    (standing still: noise/noise can spike FI; the gate forces STILL; auto-calibrated at
    boot). Explain the 2-window debounce (assert-after-2 / release-after-2). Draw the
    STILL/WALKING/FREEZE decision tree (moving? -> FI>1.815? -> 2-in-a-row? -> state) in
    TikZ. \includegraphics operating_point and capture1_trace. takeaway + keyidea boxes.`},

  { id:'09', file:'09_cnn.tex', title:'The CNN: FoGNet Architecture', pages:4,
    figures:['(TikZ: the FoGNet architecture using \\cnnvol tensor blocks)'],
    brief: String.raw`Explain why a CNN (it LEARNS temporal filters rather than using one fixed ratio).
    Walk FoGNet end to end (input 3x256; three conv blocks 16/32/64 channels, kernels
    15/9/5, each Conv->BN->ReLU, maxpool x2 after the first two; global average pooling;
    Linear 64->32 -> ReLU -> Dropout 0.3 -> Linear 32->2; softmax -> P(freeze); ~18k
    params; <1 ms/CPU). Explain convolution intuitively (a small learned filter slides
    over time and lights up on a pattern), widening receptive fields, max-pool
    (downsample -> longer spans), GAP (translation invariance: a freeze anywhere in the
    window still counts). Draw a publication-quality architecture diagram with the
    \cnnvol macro (tensor volumes shrinking in time, growing in channels), wrapped in
    \resizebox so it fits. intuition + keyidea + mathsbox (convolution).`},

  { id:'10', file:'10_training.tex', title:'Training the CNN: Loss, Gradient Descent & Generalisation', pages:4,
    figures:['bce_loss','gradient_descent','sigmoid','train_curve','calibration'],
    brief: String.raw`Supervised learning on Daphnet (Bachlin 2010): labelled windows. The sigmoid/logit
    turning a score into a probability; binary cross-entropy loss (give the formula and
    its intuition: punish confident wrong calls); gradient descent / backprop (the update
    rule, "roll downhill"); over-fitting and the regularisers (dropout, batch-norm). The
    CRITICAL evaluation protocol: Leave-One-Subject-Out (LOSO) — train on N-1 patients,
    test on the held-out one — because a random window split leaks "the person, not the
    freeze"; nested LOSO for the threshold; calibration. \includegraphics the REAL
    figures sigmoid, bce_loss, gradient_descent, train_curve, calibration (lay them out
    tidily, e.g. a 2x2/subfigure grid, no overflow). mathsbox (BCE) + keyidea (LOSO).`},

  { id:'11', file:'11_metrics.tex', title:'Measuring Accuracy: Confusion, Sensitivity & Specificity', pages:4,
    figures:['confusion_matrix','roc_curve','pr_curve','accuracy_caveat','score_hist'],
    brief: String.raw`Define the confusion matrix (TP/FP/TN/FN) and the metrics (sensitivity, specificity,
    precision, accuracy) with a booktabs 2x2. Explain WHY accuracy is the wrong headline
    for rare events — the never-freeze classifier scores ~0.90 accuracy yet 0.00
    sensitivity. Give the REAL CNN pooled-LOSO numbers (TN6842/FP1285/FN248/TP607,
    n=8982 -> sens 0.71, spec 0.84, acc 0.83). Explain ROC + AUC (0.82), PR + AUC (0.33),
    and Youden's J for choosing the operating point. \includegraphics confusion_matrix,
    roc_curve, pr_curve, accuracy_caveat, score_hist (tidy grid). mathsbox (definitions)
    + takeaway (headline = sensitivity & specificity).`},

  { id:'12', file:'12_interpretability.tex', title:'Interpretability: Opening the Box', pages:3,
    figures:['perm_importance','shap_bar','partial_dependence','decision_contour'],
    brief: String.raw`Cover the interpretable 9-feature RandomForest alternative (nested-LOSO sens 0.75 /
    spec 0.76; ROC-AUC 0.82). Explain permutation importance (shuffle a feature, watch
    accuracy drop), SHAP (which features push a given window toward "freeze"), and partial
    dependence. Argue WHY interpretability matters clinically (trust, debuggability) and
    why Cadence SHIPS the fully-transparent Freeze Index rather than a black box.
    \includegraphics perm_importance, shap_bar, partial_dependence, decision_contour
    (tidy). keyidea (glass box vs black box) + intuition (SHAP).`},

  { id:'13', file:'13_hardware.tex', title:'Hardware', pages:4,
    figures:['wiring_pinout','sleeve_layout','coding_flowchart','(TikZ: 3-board topology)'],
    brief: String.raw`The physical build. The Circuit Playground Express (SAMD21, onboard LIS3DH,
    NeoPixels, buttons, slide switch); the external SSD1306 OLED on the second I2C bus
    (SDA=A5/SCL=A4); the onboard NeoPixel ring showing state colours; the optional ESP32
    Wi-Fi bridge; the 3-board topology (CPX --UART--> ESP32 --Wi-Fi/TCP--> laptop). The
    coin motor fitted but DISABLED (monitor). The ankle sleeve / garment integration and
    power. The two firmware roles: RECORDER (cpx_fog_logger, slide-switch, SPI-flash,
    untethered capture, pulled by cpx_dump.py) vs MONITOR (cpx_fog_standalone, live
    OLED+LED display). Note the on-device DSP fits the 2 s per-hop budget on the SAMD21.
    \includegraphics wiring_pinout, sleeve_layout, coding_flowchart, and draw the 3-board
    topology in TikZ. keyidea + intuition.`},

  { id:'14', file:'14_software.tex', title:'Software', pages:4,
    figures:['(TikZ: module/dataflow diagram)','(code listing)'],
    brief: String.raw`The software architecture: the torch-free "fog" package (config, dsp, streaming,
    metrics, normalize) + the CNN (model.py) + the firmware + the dashboard server
    (asyncio + websocket -> dashboard.html). Describe dsp (magnitude/freeze_index/
    movement_energy/band_power), the streaming receiver (serial+replay, ring buffer,
    real-time pacing), the dashboard server's control loop (state machine + debounce),
    and the data pipeline (capture -> daphnet -> analyze -> figures). State the DESIGN
    LESSON: the live FI detector reads RAW accel because per-axis band-pass filtering
    before magnitude flattens the FI ratio. Mention 73 passing unit tests. Draw a clean
    module/dataflow diagram (TikZ) and include ONE short code listing (e.g. a faithful
    freeze_index in [style=py], or the gated state decision). keyidea + takeaway.`},

  { id:'15', file:'15_dashboard.tex', title:'The Live Dashboard: What Every Metric Means', pages:5,
    figures:['(TikZ: an annotated dashboard mock with callouts)','smoketest_trace'],
    brief: String.raw`THE most important section for the reader — explain, very intuitively AND
    conceptually, what each live-display metric indicates and how to "read the leg" off
    the screen. Cover every panel: Gait Status (STILL/WALKING/FREEZE — the verdict and how
    it is derived: gate -> FI -> debounce); Freeze Margin (the new meter: live FI as a %
    of the 1.815 trip, 100% = freeze, "—/at rest" when gated STILL — the glass-box
    confidence); Freeze Index (the ratio; walk ~0.5-1, freeze >2 up to 20+; the trip
    line); Tremor Power (4-6 Hz, rest-tremor band, present/clear badge); Movement Energy
    (0.5-8 Hz, how hard the leg moves, the MOVING/STILL gate); Locomotion Power (0.5-3 Hz,
    the walking rhythm); the raw Accelerometer waveform. Make the RELATIONSHIPS explicit:
    FI = freeze-band / loco-band; energy = freeze-band + loco-band; the gate uses energy;
    the state uses gate + FI + debounce. Note the CNN is bypassed in glass-box mode (why
    the old probability box was blank, now the Freeze Margin). Draw an annotated dashboard
    MOCK in TikZ (boxes laid out like the real panels with callout arrows explaining each
    — keep it tidy, no overlap, \resizebox to fit). Use a per-metric table (panel | what
    it shows | intuition | typical still/walk/freeze values) with booktabs. Lots of
    intuition boxes. This section should make a non-expert able to glance at the screen
    and understand the wearer's gait.`},

  { id:'16', file:'16_results_future.tex', title:'Results, Limitations, Future Work & Conclusion', pages:5,
    figures:['metrics_panel','per_subject','(appendix: parameter table)'],
    brief: String.raw`Summarise what works with the REAL numbers (FI on capture1: 0.83/0.92 at 1.815;
    CNN LOSO 0.71/0.84; RF 0.75/0.76, ROC-AUC 0.82). Be HONEST about limitations: a single
    ankle sensor; standing-still "freezes" do not register (must be in-place 3-8 Hz
    trembling); small personal dataset; LOSO generalisation gap; FI confounds (voluntary
    shaking, vehicle vibration, handling the device). Future work: per-patient threshold
    calibration, sensor fusion (gyro), the CNN on-device via TFLM on an ESP32-S3, more
    body sites, and a regulated closed-loop cueing version as a later step. Conclude by
    tying back to the aim. Then an APPENDIX: a booktabs table of the signal-processing
    parameters (fs 64 Hz, window 256/4 s, hop 128/2 s, df 0.25 Hz, bands, FI threshold
    1.815, debounce 2) and a short glossary (FoG, FI, LOSO, sensitivity, specificity, PSD,
    GAP). \includegraphics metrics_panel and per_subject. takeaway box to close.`},
]

// ---------------------------------------------------------------------------
function buildPrompt(s) {
  return (
    'You are writing ONE section of a polished ~50-page LaTeX technical report on the\n' +
    '"Cadence" Parkinson freeze-of-gait monitor. Produce world-class, well-illustrated,\n' +
    'rigorous-but-intuitive LaTeX, then self-validate it by compiling and visually\n' +
    'inspecting it. Work entirely from the facts below — invent NO numbers.\n\n' +
    FACTS + '\n\n' + STYLE + '\n\n' +
    '================ YOUR SECTION ================\n' +
    'Title: ' + s.title + '\n' +
    'Filename to report: ' + s.file + '\n' +
    'Target length: ~' + s.pages + ' pages.\n' +
    'Suggested figures: ' + s.figures.join('; ') + '\n\n' +
    'Brief:\n' + s.brief + '\n\n' +
    SELFCHECK
  )
}

const SECTION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    filename:    { type: 'string' },
    latex:       { type: 'string', description: 'ONLY the \\section{...} body, no preamble/wrapper' },
    compiled:    { type: 'boolean' },
    pages:       { type: 'number' },
    figures_used:{ type: 'array', items: { type: 'string' } },
    notes:       { type: 'string' },
  },
  required: ['filename', 'latex', 'compiled', 'pages'],
}

// ---------------------------------------------------------------------------
phase('Draft')
log('Drafting ' + SECTIONS.length + ' sections in parallel — each self-compiles and eyeballs its own diagrams.')

const results = await parallel(
  SECTIONS.map((s) => () =>
    agent(buildPrompt(s), {
      label: 'sec ' + s.id + ' — ' + s.title,
      phase: 'Draft',
      schema: SECTION_SCHEMA,
    }).then((r) => (r ? { ...r, _order: s.id, _file: s.file } : null))
  )
)

const ok = results.filter(Boolean)
log('Drafted ' + ok.length + '/' + SECTIONS.length + ' sections. Compiled OK: ' +
    ok.filter((r) => r.compiled).length + '. Total measured pages: ' +
    ok.reduce((a, r) => a + (r.pages || 0), 0) + '.')

return ok.sort((a, b) => String(a._file).localeCompare(String(b._file)))
