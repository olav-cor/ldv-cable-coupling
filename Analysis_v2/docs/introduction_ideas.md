# Introduction — structure & content ideas

Working notes for the thesis introduction, building on your draft paragraphs.
Citation suggestions point to PDFs you already have in
`Research_Module/Literature/` (or that are cited in the Simone draft).

## Suggested paragraph flow

1. **DAS principle & strengths** (your first paragraph, keep)
2. **Coupling: why cables are normally buried**
3. **The unburied niche: three use cases** (your (1)–(3), keep)
4. **Lunar motivation** (your second paragraph, keep)
5. **The problem: degraded signal quality when unburied** (evidence)
6. **Mechanistic picture: two-step strain transfer + bending stress relief**
7. **The gap this thesis fills + objective**
8. **Approach in one paragraph** (LDV lab experiment at CIWE)
9. **Thesis outline** (one sentence per chapter)

## Filling your `(cite)` placeholders

| Your placeholder | Candidate references (in your Literature folder / Simone draft) |
|---|---|
| "spatial sampling rate down to meter scale (cite)" | Lindsey & Martin 2021 (Fiber-Optic Seismology review); Hartog 2017 |
| "wind … decreased SNR (cite)" | Hudson et al. 2021 (Antarctica); Probst et al. 2026; Viens et al. 2025 |
| "DAS … proposed for the Moon (cite)" | Wu et al. 2024 (you cite it in the research-module report); Zhai et al. 2024 |
| "(HIGH SCATTERING ENV)" | lunar regolith scattering: Zandanel et al. 2026 (lunar-simulant preprint in your folder) + classic Apollo seismology scattering refs (to find, e.g. Latham et al. 1970) |
| "withstand extreme radiation or temperatures (sources)" | needs a space-qualification source — not in your folder yet; look for radiation-hardened fiber reviews (e.g. Girard et al., "Radiation effects on silica-based optical fibers") |
| unburied coupling evidence | An et al. 2023 (coupled vs uncoupled traffic signals); Harmon et al. 2022 (coupling strategies, hammer range); Zandanel et al. 2026 (buried vs unburied in simulant); Wilczynski (untrenched near-surface, in your folder) |
| rapid-response deployment | Mjehovich et al. 2023 (rapid surface deployment); Viens et al. 2025 (explosion series) |
| two-step transfer / cable-to-fiber step | Reinsch et al. 2017; Hubbard et al. 2022; Celli et al. 2023 (ground-cable coupling simulation) |
| bending stress relief + Θ model | Probst et al. 2026 (empirical, proposes mechanism); Simone draft (analytical model — cite as in-prep/companion paper, ask Simone how) |

## Content bullets per paragraph

**1. DAS principle & strengths**
- One-sentence measurement principle (Rayleigh backscatter phase → axial
  strain (rate) averaged over the gauge length); inherently 1-D, axial.
- Strengths: km aperture, m-scale channel spacing, robust passive cable, one
  interrogator = thousands of channels; reuse of existing telecom fibers.
- Optionally one sentence on where DAS is established (boreholes/VSP,
  trenched surface arrays) to set up "buried is the default".

**2. Why burial**
- Burial provides (a) continuous mechanical contact/coupling by overburden
  pressure and (b) wind-noise shielding — cite the wind/noise papers here.
- Cost: trenching effort, time, infrastructure; impossible/undesirable in
  the scenarios that follow.

**3. Unburied niche — your three cases (keep, maybe add citations)**
- (1) rapid response: aftershocks, landslides → Mjehovich 2023, Viens 2025.
- (2) restricted/hazardous access: volcanoes.
- (3) ground disturbance undesirable/impractical: hard rock, lava, glaciers
  (glacier DAS work exists — Hudson et al. 2021 is Antarctica).

**4. Lunar motivation**
- DAS proposed for the Moon (Wu 2024, Zhai 2024): rover/astronaut/lander
  deployment; dense sampling attractive in the strongly scattering regolith
  (ties to your HIGH SCATTERING ENV note — dense arrays help where classic
  event identification by single stations struggles).
- No atmosphere → no wind noise → the *coupling* question becomes THE
  signal-quality question (nice logical pivot to your topic).
- Fiber survivability (radiation/temperature) — needs the external source.
- Lower gravity changes the coupling physics itself (Θ ∝ g², slip ∝ normal
  force) — strong motivation for a *mechanistic* (not just empirical) model,
  because you cannot empirically test 1/6 g on Earth.

**5. Evidence of poor unburied coupling**
- Amplitude/SNR reduction in field comparisons (An 2023, Harmon 2022,
  Zandanel 2026), persisting without wind (Probst 2026 lab) → coupling, not
  just noise.

**6. Mechanistic picture**
- Two-step framework: ground→cable surface (poorly understood when unburied)
  and cable→fiber core (shear-lag, well studied — Reinsch 2017).
- Unburied: contact only at discrete points, segments bridge in between;
  suspended segments can *bend* instead of stretching — bending stress
  relief (Probst 2026, proposed qualitatively).
- The analytical model (Simone draft): segment = doubly clamped beam;
  efficiency η = 1/(1+Θ) with Θ = ½(w₀/r)² — sag vs radius. One short
  paragraph, full derivation deferred to the theory chapter.

**7. Gap + objective**
- The model exists but is untested against controlled measurements across
  cable types, stiffnesses, gap lengths and sag states.
- Objective: quantify ground-to-cable strain transfer of suspended cable
  segments in the laboratory, across 7 cables × 3 gap lengths × sag states,
  and test the Θ model — statically (quasi-static η) and dynamically (FRFs,
  resonance behaviour).
- Explicit research questions work well here, e.g.:
  1. Does the measured quasi-static η follow 1/(1+Θ)?
  2. How does the transfer behave through the first resonance, and does it
     match the predicted dip/amplification structure?
  3. Are the model's assumptions (linearity in amplitude, symmetric
     extension/compression response) valid in the tested regime?
  4. What limits detectability of these effects experimentally (excitation
     directionality, coherence)?

**8. Approach**
- Shaker-driven cable segments, scanning 3-D LDV at CIWE (ETH), log sweeps
  1–500 Hz; LDV measures the full 3-D displacement field → cable elongation
  and boundary elongation → transfer functions (point to processing
  chapter).
- One honest scoping sentence: the LDV measures *cable* motion — the
  experiment isolates the ground-to-cable step; the cable-to-fiber step and
  interrogator effects (gauge length etc.) are out of scope. (Your
  research-module report's DAS review covers those — a condensed version can
  live in the background chapter.)

**9. Thesis outline**
- One sentence each: background/theory → experimental setup → processing
  (your processing chapter) → results → discussion → conclusions/outlook
  (outlook: more materials, granular surfaces, real DAS comparison — reuse
  the outlook framing from your research-module report).

## Other things worth considering

- **Figure early in the intro**: a concept sketch of bending stress relief
  (suspended segment, axial vs flexural accommodation — analogous to Simone
  Fig. 1) plus a photo of the CIWE setup. Readers get the mechanism in one
  glance.
- **Contributions list** (3–5 bullets at the end of the intro): controlled
  multi-cable dataset; validation of the Θ model; dynamic FRF
  characterisation incl. resonance; methodology (three strain estimators,
  estimator comparison); asymmetry/linearity bounds.
- Define η and Θ symbols at first use; keep notation identical to the
  processing chapter.
- Decide early how to cite the Simone draft (companion paper in prep?) —
  affects how much theory you must re-derive in your own theory chapter.
- The research-module report's Sections 2.1–2.2 (gauge length, pulse width,
  directionality, instrument response) are background material — condense
  into the thesis background chapter rather than the introduction.
