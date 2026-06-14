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
| `--chunk-duration`, `-c` | `30` | Maks detik per chunk (0=disable) |
| `--verbose`, `-v` | `False` | Output detail |

## Chunked Mode (Long Audio)

Untuk lagu lebih dari ~120s (tergantung GPU), durasi dipecah otomatis menjadi
beberapa chunk, di-generate terpisah dengan seed berurutan, lalu di-stitch
dengan crossfade 2 detik agar transisi mulus.

Alur:
1. `n_chunks = ceil(duration / chunk_duration)`
2. Lirik dibagi proporsional per chunk
3. Tiap chunk di-generate dengan seed `seed + i`, prompt/BPM/key tetap
4. Overlap chunk[i] dengan chunk[i+1] sebesar 2s
5. Linear crossfade (fade-out / fade-in) di setiap boundary
6. Output: 1 file WAV utuh

## Examples

```bash
# Short generation (single pass)
acegen generate -t "jazz piano" -d 30 -o jazz.wav

# Long generation with chunking (e.g. 180s)
acegen generate -t "pop song" -l lyrics.txt -d 180 -L id -o lagu.wav

# Custom chunk size (for tighter GPU)
acegen generate -t "ambient" -d 180 -c 20 -o ambient.wav

# Disable chunking
acegen generate -t "beat" -d 30 -c 0 -o beat.wav
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
