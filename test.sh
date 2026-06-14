#!/usr/bin/env bash
# Test berbagai konfigurasi untuk mencari yang menghasilkan vokal terbaik
# Semua test pakai prompt + lirik yang sama

set -e

ACEVENV="$(cd "$(dirname "$0")" && pwd)/venv"
ACEGEN="$ACEVENV/bin/acegen"

PROMPT="modern pop rock, powerful female vocal, emotional and energetic"
LYRICS="lirik3.txt"
DURATION=30  # cukup 30s per test biar cepat
DIR="test_output"
mkdir -p "$DIR"

echo "=== Test Suite: Vocal Generation ==="
echo "Prompt: $PROMPT"
echo "Lyrics: $LYRICS"
echo "Duration: ${DURATION}s"
echo "Output: $DIR/"
echo ""

# ── Baseline: default ──
echo "1. Baseline (default: steps=8, cfg=4.0)"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION -o "$DIR/01_baseline.wav" 2>&1 | tail -1
echo ""

# ── More diffusion steps ──
echo "2. More steps: steps=16"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION --steps 16 -o "$DIR/02_steps16.wav" 2>&1 | tail -1
echo ""

echo "3. Steps=4 (faster, less quality)"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION --steps 4 -o "$DIR/03_steps4.wav" 2>&1 | tail -1
echo ""

# ── Higher CFG ──
echo "4. Higher CFG: cfg=6.0"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION --cfg 6.0 -o "$DIR/04_cfg6.wav" 2>&1 | tail -1
echo ""

echo "5. Higher CFG: cfg=8.0"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION --cfg 8.0 -o "$DIR/05_cfg8.wav" 2>&1 | tail -1
echo ""

# ── Different shift ──
echo "6. Lower shift: shift=2.0"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION --shift 2.0 -o "$DIR/06_shift2.wav" 2>&1 | tail -1
echo ""

echo "7. Higher shift: shift=4.0"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION --shift 4.0 -o "$DIR/07_shift4.wav" 2>&1 | tail -1
echo ""

# ── LM variants ──
echo "8. No LM (instrumental comparison)"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION --no-lm -o "$DIR/08_nolm.wav" 2>&1 | tail -1
echo ""

echo "9. Per-chunk LM (--no-consistent)"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION --no-consistent -o "$DIR/09_no_consistent.wav" 2>&1 | tail -1
echo ""

# ── Best combo ──
echo "10. Best guess: steps=16 + cfg=6.0 + shift=2.0"
time $ACEGEN generate -t "$PROMPT" -l "$LYRICS" -d $DURATION --steps 16 --cfg 6.0 --shift 2.0 -o "$DIR/10_best_combo.wav" 2>&1 | tail -1
echo ""

echo "=== Done! ==="
echo ""
echo "Files:"
ls -lh "$DIR"/*.wav | awk '{print $NF " (" $5 ")" }'
echo ""
echo "Listen and compare: open $DIR/"
