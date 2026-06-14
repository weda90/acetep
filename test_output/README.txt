Test Suite: Vocal Generation Comparison
=========================================
Prompt: "modern pop rock, powerful female vocal, emotional and energetic"
Lyrics: lirik3.txt
Duration: 30s

All files are ~5.5MB (mono/stereo 48kHz WAV).

01_baseline.wav      — steps=8, cfg=4.0 (default)
02_steps16.wav       — steps=16 (more refinement)
03_steps4.wav        — steps=4 (faster, lower quality)
04_cfg6.wav          — cfg=6.0 (stronger conditioning)
05_cfg8.wav          — cfg=8.0 (even stronger)
06_shift2.wav        — shift=2.0 (softer denoising)
07_shift4.wav        — shift=4.0 (sharper denoising)
08_nolm.wav          — LM disabled (instrumental baseline)
09_no_consistent.wav — per-chunk LM (no single blueprint)
10_best_combo.wav    — steps=16 + cfg=6.0 + shift=2.0

Compare which config produces the clearest vocals.
Focus on: 01 (baseline) vs 04 (cfg6) vs 10 (combo) vs 08 (no LM control).
