# acegen

CLI untuk generate musik dengan ACE-Step1.5 MLX 4-bit di Apple Silicon.

## Install

```bash
pip install -e /path/to/acetep
```

## Usage

```bash
# Generate dari prompt
acegen generate -t "jazz piano, intimate" -d 30 -o jazz.wav

# Dengan lirik (inline atau dari file)
acegen generate -t "pop song" -l "lyrics.txt" -d 60 -L id

# Instrumental tanpa LM (lebih cepat, kualitas lebih rendah)
acegen generate -t "ambient" --no-lm -o ambient.wav --steps 4

# Lagu panjang (180s) — otomatis chunked + consistent LM
acegen generate -t "pop rock" -l lirik.txt -d 180 -o lagu.wav

# Custom chunk size untuk GPU terbatas
acegen generate -t "ambient" -d 180 -c 20 -o ambient.wav

# Per-chunk LM (fallback, kurang konsisten)
acegen generate -t "pop" -l lirik.txt -d 180 --no-consistent -o lagu.wav
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-t`, `--text` | required | Prompt deskripsi musik |
| `-l`, `--lyrics` | `""` | Lirik (teks langsung atau path file `.txt`) |
| `-d`, `--duration` | `30` | Durasi dalam detik |
| `-o`, `--output` | `output.wav` | Nama file output |
| `-L`, `--lang` | `"en"` | Kode bahasa vokal (`en`, `id`, `zh`, dll) |
| `--no-lm` | — | Matikan 5Hz LM |
| `--lm-model` | `"0.6B"` | Ukuran LM (`0.6B` atau `4B`) |
| `--steps` | `8` | Jumlah diffusion steps |
| `--shift` | `3.0` | Timestep shift |
| `--cfg` | `1.0` | Guidance scale |
| `--bpm` | auto | Override BPM |
| `--key` | auto | Override key/scale |
| `--seed` | random | Seed reproducibility |
| `-c`, `--chunk-duration` | `30` | Maks detik per chunk (0=disable) |
| `--no-consistent` | — | Matikan single-LM consistent chunking |
| `-v` | — | Output detail |

## Catatan

- `use_lm=True` wajib untuk kualitas bagus (default ON)
- Model otomatis di-cache setelah pertama load
- Untuk durasi > 120s (tergantung GPU), chunking aktif otomatis:
  LM blueprint dibuat sekali untuk full durasi, lalu diffusion
  dijalankan per-chunk dengan blueprint yang sama + crossfade 2s
  (= consistent mode). Gunakan `--no-consistent` untuk fallback
  (per-chunk LM, blueprint berbeda tiap chunk)
- Lihat `docs/catatan-perbaikan.md` untuk dokumentasi fix issue
