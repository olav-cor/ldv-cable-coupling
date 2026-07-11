# Chapter draft (bullets): "Quantitative Analysis: DAS"

Working material for the chapter after the Introduction, with subchapters
**Technology → Physical background → Cable coupling** (Probst/Simone theory as
the focus). Everything is in bullet form so you write the prose yourself.

**Anti-plagiarism rules used here and to keep in mind while writing:**

- Every bullet ends with its source in brackets. Where a bullet condenses a
  *verbatim quote* from your `Literature_Review_Summary.docx`, it is marked
  **[Q]** — those sentences in particular must be rephrased in your own words,
  since the summary doc contains copied text.
- Numbers (dB values, ranges, thresholds) are safe to reuse as facts, but the
  sentence around them must be yours.
- The Simone draft is unpublished — ask Simone how to cite it (companion
  paper in prep / personal communication / thesis supervisor material). Your
  Research_Module folder contains **v2** (`Cable_coupling_v2_Simone.pdf`),
  which has a written Discussion; the `Papers/` folder in Experiment2 has v1.
  Work from v2.

**Source map** (citation tag → file in `Research_Module/Literature/`):

| Tag | File |
|---|---|
| He & Liu 2021 | `DAS introduction/Optical Fiber Distributed Acoustic Sensors A Review.pdf` |
| Lindsey et al. 2020 | `DAS introduction/On_the_broadband_instrument_response_of_fibre_optic_DAS_arrays_2020.pdf` |
| Paitz et al. 2021 | `DAS introduction/Paitz et al. - 2021 - Empirical Investigations...pdf` |
| Dean et al. 2017 | `DAS introduction/Dean et al. - 2017 - The effect of gauge length...pdf` |
| Dean et al. 2016 (verify ref) | `DAS introduction/The effects of pulse width on fibre-optic distributed vibration sensing data.pdf` (scanned; no text layer) |
| Hubbard 2022 | `DAS introduction/Hubbard - 2022 - Direction-dependent strain transfer function of DA.pdf` |
| Hartog et al. (verify ref) | `DAS introduction/Fibre optic based vibration sensing Nature of the measurement.pdf` (scanned; no text layer) |
| Reinsch et al. 2017 | `Cable coupling/Reinsch_2017_Meas._Sci._Technol._28_127003.pdf` |
| Celli et al. 2023 | `Cable coupling/Celli et al. - 2023 - Full-waveform simulation of DAS records...pdf` |
| Harmon et al. 2022 | `Cable coupling/Harmon et al. - 2022 - Surface deployment of DAS systems...pdf` |
| Mjehovich et al. 2023 | `Cable coupling/Rapid_surface_deployment_of_a_DAS_system_Mjehovich_2023.pdf` |
| Hudson et al. 2025 (preprint) | `Cable coupling/Hudson et al. - 2024 - Unlocking DAS amplitude information through cohere.pdf` |
| Probst et al. 2026 | `Cable coupling/Earth and Space Science - 2026 - Probst - Controlled Source DAS Coupling Tests...pdf` (same study as the Zandanel-listed preprint `DAS_coupling_lunar_simulant_Preprint.pdf`) |
| Simone draft v2 | `Cable coupling/Cable_coupling_v2_Simone.pdf` |
| Wilczynski et al. 2026 | `Cable coupling/Wilczynski_untrenched_DAS_near_surface.pdf` |
| Probst strain-methods note | `Methods_to_derive_inline_strain_Probst.pdf` |
| Koh et al. 1999 | `koh-et-al-1999-low-tension-cable-dynamics-numerical-and-experimental-studies.pdf` |
| Suslov et al. 2025 (verify authors) | `Cables_Bending_stiffness_1-s2.0-S2665917424006482-main.pdf` (Measurement: Sensors 38, 101672) |

Wind/lunar refs cited *through* the Simone draft's reference list (An et al.
2023, Castongia et al. 2017, Forbriger et al. 2024, Hudson et al. 2021, Viens
et al. 2025, Wu et al. 2024, Zhai et al. 2024, Zandanel et al. 2026, Kuvshinov
2016, Hartog 2017, Lindsey & Martin 2021) — get the originals before citing
directly, or cite via the draft.

---

## 2.1 Technology: how DAS measures

### Measurement principle

- DAS interrogates Rayleigh backscattering (RBS) of coherent laser pulses in a
  standard fibre; interference of the backscatter within one spatial-resolution
  cell gives high sensitivity, and the *phase* of the backscatter tracks axial
  strain of the fibre [He & Liu 2021] **[Q]**.
- Seismological framing: DAS measures the phase of backscattered pulses and
  relates it to the axial strain (or strain rate) induced by a passing
  wavefield — a distributed, inherently one-dimensional strain measurement
  along the cable [Lindsey et al. 2020] **[Q]**; scattering centres are unknown
  individually but assumed dense, homogeneous, time-invariant [Lindsey et al.
  2020] **[Q]**.
- Two functions of any DAS system: (1) *localize* the scatter along the fibre
  — most systems via optical time-domain reflectometry (OTDR, pulse
  time-of-flight); (2) *retrieve the vibration signal* from the RBS, whose
  interference state changes under strain (phase shift + intensity
  fluctuation) [He & Liu 2021] **[Q]**.
- Full vibration information (amplitude, frequency, phase) at all positions
  along fibres up to ~100 km [He & Liu 2021] **[Q]**.

### Interrogation variants (keep short; focus on OTDR as in your report plan)

- OTDR family: intensity-based OTDR → coherent/phase-sensitive variants
  (COTDR, φ-OTDR) using narrow-linewidth lasers; phase demodulation restores
  amplitude/frequency/phase and defines "DAS" proper [He & Liu 2021].
- Spatial resolution of OTDR: Z_SR = c·T_p / (2n) — set by pulse width T_p and
  refractive index n [He & Liu 2021].
- OFDR: frequency-swept continuous probe; high SNR + high resolution but
  laser-phase-noise/crosstalk limited → short range, low response bandwidth —
  mention only in passing [He & Liu 2021] **[Q]**.
- TGD-OFDR: time-gated hybrid; resolution set by sweep range instead of pulse
  width → ~100 km range with metre resolution [He & Liu 2021] **[Q]**.
- Demodulation routes: direct phase calculation + differentiation vs.
  frequency scanning; both linear in applied strain; most commercial systems
  are phase-demodulation type [He & Liu 2021] **[Q]**.
- Dual-pulse method: two pulses (different frequency, fixed delay) — the
  extracted phase is already the spatial differential; a commercial example is
  the Terra15 interrogator, which effectively measures fibre velocity
  [He & Liu 2021; Terra15 web page — replace by citable source] **[Q]**.
- Instrument diversity caveat: DAS instruments differ in one- vs two-pulse
  input, phase vs amplitude analysis, digital vs photonic phase measurement,
  time vs frequency domain — response statements are instrument-specific
  (Lindsey et al. studied the Silixa iDAS v2) [Lindsey et al. 2020] **[Q]**.

### Key acquisition parameters

- **Gauge length**: the differencing baseline over which strain is averaged
  (metres to tens of metres); acts as a spatial low-pass filter on the strain
  field [Lindsey et al. 2020; Dean et al. 2017] **[Q]**.
  - Too small → poor SNR; too large → reduced resolution and wavelet
    distortion; optimum depends on the velocity and frequency content of the
    measured waves; multiple gauge lengths can pay off when velocities vary
    [Dean et al. 2017] **[Q]**.
  - Below ~8 m gauge length the phase–strain relation becomes nonlinear as the
    gauge length approaches the pulse length [Dean et al. 2017] **[Q]**.
  - Hardware- or software-defined depending on architecture; selectable
    post-acquisition in some systems [Lindsey et al. 2020, citing Hartog
    2017] **[Q]**.
  - Thesis link: your LDV "virtual channels" have no gauge length — one
    honest sentence that gauge-length averaging over many suspended segments
    is exactly what a real DAS would add on top of the single-segment physics
    you measure (Simone draft makes this segment-averaging point in its
    model section) [Simone draft v2].
- **Pulse width & shape**: SNR ∝ pulse energy; higher resolution needs
  narrower pulses → SNR trade-off is *the* OTDR design tension [He & Liu 2021]
  **[Q]**; pulse power capped by onset of nonlinear fibre effects [He & Liu
  2021] **[Q]**; interference fading makes some channels randomly weak
  [He & Liu 2021] **[Q]**.
  - Pulse shape is manufacturer-fixed, approximately Gaussian; at high spatial
    resolution it matters as much as gauge length for the magnitude response
    [Hubbard 2022; Dean et al. 2016] **[Q]**.
- **Directionality**: DAS senses only axial strain → strong angular
  dependence compared to geophones; broadside-arriving P energy can appear
  ~10× weaker than on collocated inertial sensors; helically wound cables
  (~30° wrap) approach isotropy but lose direction information [He & Liu 2021;
  Lindsey et al. 2020, citing Kuvshinov 2016] **[Q]**.
  - Thesis link: directionality is why your experiment separates axial
    (chord-projected) from transverse cable motion — the fibre only "sees"
    the axial part.

## 2.2 Physical background: response, performance, limits

### Instrument response

- Idealized DAS transfer function w.r.t. apparent wavenumber factorizes into
  gauge-length and pulse-shape terms, H(k) = G(k)·P(k), assuming *perfect
  coupling* [Hubbard 2022] **[Q]** — note for the thesis: every published
  response study assumes the coupling your experiment quantifies.
- Empirically flat response: 1:1 agreement with ground motion for periods of
  10–120 s; at 1–10 s amplitude runs 3–11 dB hot (possibly conduit/coupling
  effects); phase response flat [Lindsey et al. 2020] **[Q]**.
- Across 17 octaves (1/3600 Hz – 60 Hz, four sites): DAS ≈ ground deformation
  within ~4 dB amplitude and ~0.167π phase scatter; deviations partly from
  phase-velocity assumptions in the conversion — supports waveform-based uses
  (FWI) [Paitz et al. 2021] **[Q]**.
- Long-period capability demonstrated down to T = 200 s (earthquakes) and even
  tidal periods in the lab [Lindsey et al. 2020, citing Yu 2019, Becker &
  Coleman 2019] **[Q]**.
- Unlike conventional seismometers there is no standard poles-and-zeros
  instrument response yet; response = optics + fibre + cable + coupling as one
  system [Lindsey et al. 2020] **[Q]**.

### Noise & limitations (condense to one paragraph)

- Noise types: common-mode (infinite-velocity, interrogator vibration), laser
  drift spikes, time-invariant low-amplitude channel patterns (laser
  frequency/gauge/pulse-dependent — or poor coupling) [Lindsey et al. 2020]
  **[Q]**.
- Range limit: fibre attenuation vs. minimum acceptable SNR; sensitivity limit:
  photons per pulse & backscatter profile; max strain rate: phase unwrapping /
  fibre elasticity [He & Liu 2021; Lindsey et al. 2020] **[Q]**.
- Amplitude fidelity in the field: untrenched surface DAS shows amplitude
  deviations above ~50 Hz and reduced coherence at larger offsets, while the
  surface-wave band (<35 Hz) stays usable [Wilczynski et al. 2026 — *not yet
  in your review*].

## 2.3 Cable coupling (core subchapter)

### Two-step strain-transfer framework

- Ground strain reaches the fibre core in two steps: (1) **ground-to-cable**
  transfer to the cable jacket; (2) **cable-to-fibre** transfer through the
  concentric cable layers [Simone draft v2; Reinsch et al. 2017].
- Step 2 (cable-to-fibre) is comparatively well understood: shear-lag
  mechanics through the layers; efficiency depends on geometry and materials
  and improves with wavelength — near-ideal at seismic wavelengths; empirics:
  tight-buffered > loose-tube (gel-filled) [Reinsch et al. 2017; via Simone
  draft: Castongia et al. 2017, Forbriger et al. 2024] **[Q]**.
- Step 1 for **buried** cables: soil/backfill mechanics; spring-model +
  full-waveform simulations show the stiffness of the material immediately
  around the cable dominates — up to 2× amplification, phase delays,
  selective surface-wave amplification, interface waves; coupling and site
  effects vary along the cable [Celli et al. 2023] **[Q]**.
- Step 1 for **unburied** cables — the open problem this thesis addresses:
  only the cable's own weight provides contact; contact is intermittent at
  discrete points; friction-limited tangential force [Simone draft v2].

### Empirical evidence on unburied deployments

- Cemented vs. uncoupled same-road comparison: uncoupled cable → weaker
  amplitudes, lower SNR [An et al. 2023, cited via Simone draft].
- Coupling strategies on grass: uncoupled cable captured hammer energy only
  to ~10 m; weighting more than doubled the range (though improvements are
  not universal — Forbriger et al. 2024) [Harmon et al. 2022].
- Rapid surface deployment is attractive exactly where trenching is
  impossible/too slow; pressing cable into snow beat draping [Mjehovich et
  al. 2023].
- Untrenched vs. geophones/accelerometers in hardrock: usable surface-wave
  band (reliable Vs profiles, especially in common-receiver gathers), but
  reduced sensitivity/coherence at high frequency and offset, a systematic
  time delay in the surface-wave band, and failed ambient-noise
  interferometry due to variable channel coupling [Wilczynski et al. 2026 —
  *new to your review*].
- Controlled shaker tests in lunar regolith simulant (the empirical companion
  to your thesis): burial improves amplitude/phase reliability; thick, stiff
  unburied cables (OD5.5, OD3) ≈ buried performance with stable amplitude
  ratios and phases; thin OD0.9 cable weak and deployment-sensitive
  (settling into dust improved it, redeployment degraded it); free-hanging
  segments degrade coupling for *all* cables; wind is a major terrestrial
  noise source that the Moon lacks [Probst et al. 2026 / Zandanel-preprint].

### Quantifying coupling in the field — and the parallel to your processing

- Problem: absolute DAS amplitudes (magnitudes, moment tensors, attenuation)
  require knowing the coupling; no in-field quantification method existed
  [Hudson et al. 2025 preprint] **[Q]**.
- Their method: model fibre + couplings as springs (Kelvin–Voigt damping for
  attenuation, optional atmosphere springs); energy balance between adjacent
  channels ⇒ waveform **coherency of neighbouring channels** (zero-lag
  normalized cross-correlation, moveout-corrected) measures *relative*
  coupling; calibration against a reference (well-coupled channel or
  colocated sensor) gives approximate absolute coefficients [Hudson et al.
  2025].
- Subtlety they highlight: perfectly *uncoupled* neighbours are also mutually
  coherent — coherency is relative, and fully decoupled fibre records
  nothing, which is how the ambiguity is broken in practice [Hudson et al.
  2025].
- Their pragmatic conclusion: often a binary classification suffices —
  channels are either ≈ perfectly coupled or too poorly coupled for any
  amplitude work [Hudson et al. 2025] **[Q]**.
- **Parallels to your processing (worth 1–2 paragraphs in the thesis):**
  1. Same philosophy, different estimator: Hudson gates amplitude
     *usability* on waveform coherency between adjacent channels; you gate
     the strain-transfer FRF on the **magnitude-squared coherence γ²(f)**
     between boundary input and cable elongation before band-averaging η.
     Both say: amplitude information is only meaningful where coherence is
     high.
  2. What they must calibrate, you measure: their coherency yields only
     *relative* coupling because the true ground motion is unknown; your
     laboratory reference (shaker/endpoint elongation) makes η an *absolute*
     transfer coefficient — the lab experiment provides exactly the ground
     truth the field method lacks. Your η vs Θ curve is, in their language, a
     physics-based calibration of the coupling coefficient.
  3. Their binary good/bad-coupling finding maps onto your coherence-gated
     band: bins below γ² = 0.7 are discarded rather than corrected.
  4. Method transfer idea (outlook material): their frequency-domain
     alternative (adaptive covariance filtering) and your Welch-coherence
     gating are directly comparable; and their spring-damper channel model
     could be parameterized with your measured η(Θ).
  5. Difference to flag: Hudson's coherency is *spatial* (channel-to-channel,
     time-domain), yours is *input–output* (frequency-resolved); citing them
     as "coherence-based coupling quantification exists; we use an
     input-referenced variant possible in the lab" is accurate and clean.

### The analytical bending-stress-relief model (Probst/Simone — the theory your thesis tests)

Recommended build-up (all [Simone draft v2] unless noted):

- **Model idea**: draped cable = sequence of suspended segments between
  discrete contact points; each segment = doubly clamped beam deformed by its
  endpoint motion; DAS response ≈ gauge-length average over many segments.
- **Assumptions** (list them — your experiment tests several): initially
  straight cable, level contact points, in-line strain field with wavelength
  ≫ segment, no slip at contacts (⇒ no segment interaction), deformation
  restricted to axial mode x_l + first flexural mode x_f with
  φ(x) = ½[1 − cos(2πx/L)].
- **Static derivation chain** (your processing chapter already contains the
  full math; here only the structure): arc-length constraint
  L = L₀ + x_l − ½C₁x_f² → gravity equilibrium sag
  x_f0 = ρAgL⁴/(4π⁴EI) (Eq. 8) → energy minimization of the incremental
  deformation → strain-transfer efficiency
  **η = δx_l/δL = 1/(1+Θ)**, Θ = ρ²g²A³L⁸/(128π⁸E²I³) (Eq. 14), which
  collapses to **Θ = ½(w₀/r)²** with the measured sag (Eq. 16).
- **Physical reading**: sag-to-radius ratio governs everything; w₀ < r/4
  (Θ < 1/32) ⇒ rod-like, near-perfect transfer; larger sag ⇒ endpoint motion
  absorbed as bending change ("bending stress relief"). Pretension raises η
  at given Θ but does not move the cutoff [Simone draft v2, Discussion].
- **Dynamics**: fundamental clamped-beam resonance ω₁ = (22.4/L²)√(EI/ρA);
  quasi-static plateau below f₁ → destructive coupling dip just below f₁ →
  ~180° phase flip at resonance → constructive amplification >100 % above f₁
  (severity grows with Θ) → inertia suppresses bending at high f, η → 100 %.
  Resonance regime must be avoided for reliable strain recording.
- **Model ⇄ field evidence** (v2 Discussion, new vs v1): explains reduced
  unburied amplitudes and the thick/stiff-cable advantage reported
  empirically [An et al. 2023; Harmon et al. 2022; Zandanel et al. 2026;
  Probst et al. 2026 — via Simone draft v2]; direct quantitative comparison
  impossible in the field because segment length L is never measured — the
  quantitative gap your controlled experiment fills.
- **Stated limitations** (mirror them in your discussion): no slip, no
  inherent (spool) bend, steady-state response, bare ground, single-segment
  model [Simone draft v2].
- **Deriving η from LDV data** — Simone's methods note describes both
  estimators: the geometric arc-length route (including the linearized form
  δd_i ≈ ê_i·(u_{i+1} − u_i) to avoid catastrophic cancellation — exactly
  your Method 2) and a mode-shape-projection route (fit φ(x), read δx_f, use
  the constraint δx_l = δL + C₁x_f0·δx_f; valid only while the cable actually
  deforms in that mode shape) [Probst strain-methods note]. Your thesis
  implements the first plus two gradient-based variants; the mode-shape
  projection is honest outlook material (it fails when higher-order modes
  appear — which your modal analysis detects).

### Supporting mechanics literature (short, one paragraph)

- Low-tension cable dynamics: sag-to-span < 1:8 ⇒ parabolic equilibrium shape
  is a good approximation (justifies your constrained parabola sag fit);
  low-tension, large-displacement cables need dedicated numerics — context
  for why the segment problem is nontrivial [Koh et al. 1999 — *new to your
  review*].
- Measured bending stiffness of standard SMF-28 fibre is broadly consistent
  with Euler–Bernoulli beam theory (traceable force-displacement
  measurements) — supports modelling fibre/cables as EB beams and connects to
  your three-point-bending E measurements [Suslov et al. 2025 (verify
  authors), Measurement: Sensors 38 — *new to your review*].

## 2.4 DAS for the Moon (either own subchapter or folded into the introduction)

- Proposed lunar fibre seismology: robustness + dense sampling attractive for
  moonquakes/impacts; rover unrolling km of fibre [Wu et al. 2024 — cited in
  your report; Zhai et al. 2024 via Simone draft].
- High-scattering regolith argument for dense arrays; buried vs unburied in
  simulant: earthquakes detectable unburied but weaker [Zandanel et al. 2026
  via Simone draft; Probst et al. 2026].
- No atmosphere ⇒ no wind noise ⇒ coupling becomes *the* signal-quality
  question [Probst et al. 2026].
- Lower gravity: Θ ∝ g² shrinks (good) but normal force shrinks too ⇒ slip
  risk (bad) — the mechanistic model is what lets you extrapolate to 1/6 g at
  all [Simone draft v2].
- Radiation/temperature hardness of fibre: still needs an external source
  (not in your folder) — e.g. radiation-effects-on-fibre reviews.

---

## Gaps / to-do before writing

- [ ] Obtain originals for the pass-through citations (An 2023, Forbriger
      2024, Wu 2024, Zhai 2024, Kuvshinov 2016, Hartog 2017, Lindsey & Martin
      2021, Castongia 2017) rather than citing via the Simone draft.
- [ ] Verify exact references of the two scanned PDFs (pulse-width paper ≈
      Dean et al. 2016; "Nature of the measurement" ≈ Hartog et al.) — no
      text layer, I could not read authors/year from the files.
- [ ] Ask Simone: citation form for the draft + whether the strain-methods
      note can be cited or should be absorbed silently into your methods.
- [ ] Missing external source: fibre survivability in space (radiation,
      thermal cycling).
- [ ] Decide where gauge-length/pulse-width detail lives: full detail here
      (2.1) and only one scoping sentence in the experimental chapter.
