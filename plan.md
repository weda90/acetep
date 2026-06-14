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
| `--shift` | `1.0` | Timestep shift |
| `--cfg` | `1.0` | Guidance scale (auto 4.0 jika ada lirik) |
| `--bpm` | auto | BPM override |
| `--key` | auto | Key override |
| `--seed` | random | Seed |
| `--model` | `"mlx-community/ACE-Step1.5-MLX-4bit"` | HF model ID |
| `--chunk-duration`, `-c` | `0` | Aktifkan chunked mode: maks detik per chunk (default: disabled) |
| `--no-consistent` | — | Matikan single-LM consistent chunking |
| `--verbose`, `-v` | `False` | Output detail |

## Vocal Fixes

MLX port ACE-Step memiliki 2 bug yang menyebabkan vokal tidak muncul:

| Bug | Fix |
|-----|-----|
| LM prompt tidak cantumkan bahasa | Monkey-patch `_format_prompt()` tambah `- language: en` |
| Sinyal vokal lemah di diffusion | `guidance_scale` auto 4.0 + prompt auto-enrich `", with English vocals"` |

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

### Chunking Inconsistency

Current crossfade stitch (2s overlap) sering menyebabkan:
- Vokal prominence tidak konsisten antar chunk
- Beat/bpm patah/pitch shift di sambungan
- Artifak audible di overlap region

**Planned fix: Repaint-based stitching** — ganti crossfade dengan `task_type="repaint"`
untuk regenerate overlap region secara natural (lihat Audio Input section di bawah).

## Audio Input (source_audio)

ACE-Step1.5 menerima audio input via parameter `source_audio` (mx.array).
6 task type, tapi MLX turbo hanya support 3:

| Task | Fungsi | Tersedia di MLX? |
|------|--------|-----------------|
| `text2music` | Text → musik | ✅ |
| `cover` | Source audio → cover ulang | ✅ |
| `repaint` | Regenerate segmen spesifik | ✅ |
| `extract` | Stem separation (vocal/drum) | ❌ (non-turbo) |
| `lego` | Tambah track ke audio existing | ❌ (non-turbo) |
| `complete` | Orkestrasi dari motif | ❌ (non-turbo) |

### Batasan MLX
- `voice` parameter ada tapi **"not yet implemented"** — voice cloning tidak bisa
- `reference_audio` tidak ada di MLX — timbre/style reference tidak bisa
- `_prepare_timbre()` selalu feed silence — reference audio tidak difungsikan

### Plan: Repaint untuk Stitching Chunk

Gantikan crossfade 2s dengan repaint pada overlap region:

```
Chunk 0 ──────────────────────┐
                               ├── [28s–34s] → repaint task
Chunk 1 ──────────────────────┘
```

Langkah:
1. Generate chunk 0 (0–32s) dan chunk 1 (30–62s) seperti biasa
2. Concat kasar, ambil region overlap (28–34s) sebagai `source_audio`
3. Panggil `model.generate(task_type="repaint", source_audio=overlap_audio, duration=6)`
4. Model repair transisi — hasil seamless tanpa crossfade

### Test Plan (sebelum implementasi)
1. Generate audio 30s `text2music` → save
2. Load sebagai `source_audio`, generate ulang dengan `task_type="cover"`, prompt baru
3. Validasi: apakah cover mode berfungsi di MLX turbo?
4. Jika ya: implementasi repaint stitching ke chunking

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
