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
| `--verbose`, `-v` | `False` | Output detail |

## Examples

```bash
acegen generate -t "jazz piano" -d 30 -o jazz.wav
acegen generate -t "pop song" -l lyrics.txt -d 60 -L id
acegen generate -t "ambient" -o ambient.wav
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
