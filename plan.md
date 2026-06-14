# Plan: CLI App `acegen`

CLI wrapper untuk generate musik dari text/lyrics menggunakan ACE-Step1.5 MLX 4-bit via `mlx_audio`.

## Commands

```
acegen generate       # Generate musik dari prompt
acegen config         # Tampilkan default params
acegen info           # Informasi model dan status
```

## `acegen generate` Options

| Flag | Default | Description |
|------|---------|-------------|
| `--text`, `-t` | required | Prompt musik |
| `--lyrics`, `-l` | `""` | Lirik (file `.txt` atau inline) |
| `--duration`, `-d` | `30.0` | Durasi (detik) |
| `--output`, `-o` | `output.wav` | File output |
| `--lang`, `-L` | `"en"` | Bahasa vokal (`en`, `id`, dll) |
| `--lm` | `True` | Gunakan 5Hz LM |
| `--lm-model` | `"0.6B"` | Ukuran LM |
| `--steps` | `8` | Diffusion steps |
| `--shift` | `3.0` | Timestep shift |
| `--cfg` | `1.0` | Guidance scale |
| `--bpm` | auto | BPM override |
| `--key` | auto | Key override |
| `--seed` | random | Seed |
| `--model` | `"mlx-community/ACE-Step1.5-MLX-4bit"` | HF model ID |
| `--chunk-duration`, `-c` | `0` | Aktifkan chunked mode: maks detik per chunk (default: disabled) |
| `--no-consistent` | — | Matikan single-LM consistent chunking |
| `--verbose`, `-v` | `False` | Output detail |

## Chunked Mode (Long Audio)

Aktif dengan flag `-c <detik>`. Untuk lagu yang melebihi kapasitas GPU
(> ~120s tergantung GPU), durasi dipecah menjadi beberapa chunk,
lalu di-stitch dengan crossfade 2 detik.

### Consistent Mode (Default)

1. **5Hz LM jalan SEKALI** untuk full durasi → 1 blueprint koheren
2. Blueprint dipecah per chunk (seed berurutan, BPM/key/prompt tetap)
3. Diffusion jalan per-chunk dengan blueprint yang sama
4. Crossfade 2s di setiap boundary → musik tetap konsisten

### Fallback (`--no-consistent`)

Gunakan jika single LM gagal (GPU terbatas):
1. Tiap chunk generate LM sendiri (blueprint berbeda tiap chunk)
2. Seed, BPM, key, prompt tetap antar chunk
3. Crossfade 2s untuk smoothing transisi

## Examples

```bash
# Short generation (single pass) — default, no chunking
acegen generate -t "jazz piano" -d 30 -o jazz.wav

# Long generation (180s) — gunakan -c untuk aktifkan chunking
acegen generate -t "pop song" -l lyrics.txt -d 180 -L id -c 30 -o lagu.wav

# Custom chunk size 20s
acegen generate -t "ambient" -d 180 -c 20 -o ambient.wav

# Per-chunk LM (less consistent, fallback)
acegen generate -t "pop" -l lyrics.txt -d 180 -c 30 --no-consistent -o lagu.wav
```

## Struktur File

```
acetep/
├── acegen/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   └── generate.py
├── pyproject.toml
├── song.py
└── plan.md
```
