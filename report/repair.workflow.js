export const meta = {
  name: 'cadence-report-repair',
  description: 'Fix flagged visual defects in 9 report section figures, each self-validated by re-render',
  phases: [{ title: 'Repair', detail: 'one agent per section file: fix, recompile, re-render, confirm clean' }],
}
const PREAMBLE = '/Users/paing/Documents/scenario2_pd/report/preamble.tex'
const SECDIR = '/Users/paing/Documents/scenario2_pd/report/sections'
const OLD = '/tmp/report_pages'

const FILES = [
  { file: '01_aim.tex', pages: [7], issues: [
    `HIGH: in the gait-strip diagram the red "FREEZE" label at the top is overlapped / struck through by the tall red zig-zag waveform spikes. Raise the FREEZE label clear above the waveform's maximum, or shrink the spike amplitude under it, so the label is fully legible.`,
    `HIGH: in the Freeze-Index-per-phase bar chart, the value labels "1.13" and "1.20" sit on the dashed "deployed threshold FI = 1.815" line and read as struck through. Move the small-bar value labels clearly ABOVE the dashed line (offset above each bar).`,
    `MEDIUM: the "per-phase max" marker renders as a detached grey horizontal dash floating high in the plot, disconnected from any bar. Make it an attached cap/whisker on each bar, or drop it and keep the value labels.`,
  ]},
  { file: '02_overview.tex', pages: [9, 11], issues: [
    `MEDIUM: in the pipeline block diagram the annotation "shipped (glass box)" collides with / is cut by the dashed border of the ANALYSE container box. Move it inward/upward to clear the dashed boundary, or put a white fill behind the label.`,
    `LOW: the small secondary text inside the pipeline boxes ("256 samp (4 s) / 50% overlap", "energy > floor?", "1-D conv ~18k par to P(freeze)") is too small/soft — bump the sub-label font one step and/or box padding for print legibility.`,
  ]},
  { file: '03_sensing.tex', pages: [13, 15], issues: [
    `HIGH: in the band / sampling-ceiling diagram, the two band-range captions collide in the middle and render as an unreadable mash like "0.5 Hz to 3Hz to 8 Hz". Give each band its OWN non-overlapping label: "locomotor 0.5-3 Hz" centred over the blue band and "freeze 3-8 Hz" centred over the red band (separate baselines, each anchored to its band centre).`,
    `MEDIUM: in the MEMS axis diagram the horizontal "z (vertical, gravity)" label overlaps the rotated vertical "lower leg" label, tangling the glyphs. Move the rotated "lower leg" label away from the z-axis/leg or shift the z label so they do not intersect.`,
  ]},
  { file: '05_fourier.tex', pages: [22, 23], issues: [
    `MEDIUM: in the DFT-bin spectrum, the labels "locomotor band 0.5-3 Hz" and "freeze band 3-8 Hz" sit at the very top and collide with the dashed band-boundary gridlines and the tallest bars. Add y-axis headroom (raise ymax) or lower the labels into clear space so they touch neither gridlines nor bars.`,
    `LOW: in the "Raw vs windowed slice" panel, the "raw (seam jump)" label is too close to the dashed curve / right frame and "Hann-tapered" overlaps the navy curve. Nudge both labels clear of the curves.`,
  ]},
  { file: '06_freeze_index.tex', pages: [24, 26], issues: [
    `HIGH: in the Freeze-Index-per-phase bar chart, the dashed "deployed threshold FI = 1.815" line passes through the "1.13" and "1.20" value labels (struck through) and the threshold label text sits on the freeze bar. Offset the small-bar value labels above the dashed line and move the threshold label into clear whitespace.`,
    `MEDIUM: BROKEN CROSS-REFERENCE — the Freeze Index equation prints as number (10) but later text cites it as "Eq. (25)". Fix the label/ref pairing so the in-text reference shows the same number the equation actually prints. (Make sure every equation reference resolves correctly.)`,
    `MEDIUM: the "per-phase max" grey tick for the freeze phase floats as an isolated dash at the top, disconnected from any bar — attach it as a cap on the bar or remove it.`,
  ]},
  { file: '09_cnn.tex', pages: [41, 42], issues: [
    `MEDIUM: in the FoGNet architecture diagram (the hero figure), under the Dense/Logits stage the annotations "softmax to P(freeze)" and "ReLU, Drop 0.3" are interleaved and crowded on overlapping baselines. Separate them vertically so each sits cleanly under its stage with no collision. Make this diagram crisp.`,
    `NOTE: the per-patient bar-chart issues on p42 (out-of-order subject S04, faint grey "no freeze window" labels) are inside a PRE-RENDERED included PDF figure (per_subject) — you cannot edit that PDF. SKIP those and list them under "skipped". Only fix things drawn in TikZ/pgfplots in this file.`,
  ]},
  { file: '10_training.tex', pages: [44, 45, 46], issues: [
    `MEDIUM: in your TikZ/pgfplots panels, in-plot annotation text ("Decisive linear / slope approx 1/4", "decision point z=0", "early stop", "best checkpoint", etc.) is very small and low-contrast. Increase the annotation font a step and use a dark colour (ink or navy) so it is legible; space stacked annotations apart so they do not crowd the curves.`,
    `NOTE: if any of these panels are PRE-RENDERED included PDF figures (sigmoid, bce_loss, gradient_descent, train_curve, calibration) rather than TikZ/pgfplots you drew, you CANNOT edit them — SKIP and list under "skipped". Only fix diagrams you drew yourself.`,
  ]},
  { file: '14_software.tex', pages: [64], issues: [
    `HIGH: in the module / dataflow diagram, the arrow from "Daphnet dataset" to "model.py" is routed straight THROUGH the body of the "normalize / scaling helpers" node box. Re-route this connector so no edge passes through any node box (bend the arrow around it, or reposition a node), keeping the diagram tidy and non-overlapping.`,
    `LOW: once re-routed, confirm the intentional strikethrough on the "normalize" node still reads as a deliberate "unused" convention (add a one-line legend note if it could be misread).`,
  ]},
  { file: '15_dashboard.tex', pages: [68, 71], issues: [
    `MEDIUM: in the annotated dashboard-mock diagram, the FREEZE MARGIN panel headline "118 %" shows a DOUBLED / overlapping percent sign (a second smaller percent glyph drawn over the first). Make a single clean "118%" render (remove the duplicate node or stray extra percent).`,
    `LOW: the right-hand italic callout annotations ("The verdict.", "The gate.", "Ground truth.", and sub-labels) are very small light-grey italic — darken them (use ink instead of a pale grey) and/or enlarge one step for legibility.`,
    `LOW: in the per-metric table the "STILL / WALKING / FREEZE" cell wraps after every slash, inconsistently with other cells — widen that column or control the line break so it reads cleanly.`,
  ]},
]

const FIX_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    file: { type: 'string' },
    fixed: { type: 'boolean' },
    compiled: { type: 'boolean' },
    summary: { type: 'string' },
    skipped: { type: 'array', items: { type: 'string' } },
  },
  required: ['file', 'fixed', 'compiled'],
}

phase('Repair')
log('Repairing ' + FILES.length + ' section files in parallel (each recompiles + re-renders to verify).')

const pad = (n) => String(n).padStart(2, '0')
const results = await parallel(FILES.map((F) => () => {
  const id = F.file.replace('.tex', '')
  const imgs = F.pages.map((p) => OLD + '/pg-' + pad(p) + '.jpg')
  const prompt =
    'You are fixing VISUAL DEFECTS in ONE LaTeX section of a polished technical report, WITHOUT changing its ' +
    'meaning or any numbers. The file to edit IN PLACE is:\n  ' + SECDIR + '/' + F.file + '\n\n' +
    'A visual-QA reviewer flagged these defects (read the OLD page images afterwards to SEE each one):\n' +
    F.issues.map((s, i) => '  ' + (i + 1) + '. ' + s).join('\n') + '\n\n' +
    'OLD rendered page images showing the defects — READ each to see the problem:\n  ' + imgs.join('\n  ') + '\n\n' +
    'STEPS:\n' +
    '1. Read the section file and the old image(s). Locate each flagged diagram/label.\n' +
    '2. Edit the file in place to fix every FIXABLE defect. Preserve all prose, numbers, equations and figures — ' +
    'adjust only layout, positioning, sizing, label placement, and arrow routing. Do NOT add packages or redefine ' +
    'colours (the shared preamble already has everything). If a defect is inside a pre-rendered included PDF figure ' +
    '(not TikZ/pgfplots you can edit), SKIP it and record it under "skipped".\n' +
    '3. VALIDATE: make a temp dir /tmp/' + id + '_fix and write t.tex there as a standalone wrapper that: inputs the ' +
    'shared preamble at ' + PREAMBLE + ', opens the document, inputs your edited section file at ' + SECDIR + '/' + F.file +
    ', and closes the document. Then run "tectonic t.tex" in that dir and fix until it compiles with no errors.\n' +
    '4. RE-RENDER AND LOOK: run "pdftoppm -jpeg -r 110 t.pdf /tmp/' + id + '_fix/p", then READ the new p-*.jpg images ' +
    'and CONFIRM every defect is gone and you introduced no new overlap/overflow/off-page content. Iterate until clean.\n' +
    '5. Leave the corrected file saved at ' + SECDIR + '/' + F.file + ' (overwrite it).\n' +
    'Return: file, fixed, compiled, a short summary of the changes, and any skipped (unfixable, in-figure) items.'
  return agent(prompt, { label: 'fix ' + F.file, phase: 'Repair', schema: FIX_SCHEMA })
}))

const ok = results.filter(Boolean)
log('Repaired ' + ok.filter((r) => r.fixed).length + '/' + FILES.length + ' files; all compiled: ' +
    ok.every((r) => r.compiled))
return ok
