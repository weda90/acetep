# ACE-Step1.5 Music Generation (MLX 4-bit)

Generate music from text prompts using ACE-Step1.5 MLX 4-bit quantized model on Apple Silicon.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install mlx-audio
pip install git+https://github.com/Blaizzy/mlx-audio.git
```

**Apply fixes** (see `catatan-perbaikan.md`):
- `ace_step.py`: Add `_quantize_module()` + enable 5Hz LM for all tasks

## Usage

```python
from mlx_audio.tts.utils import load_model
import soundfile as sf

model = load_model("mlx-community/ACE-Step1.5-MLX-4bit")

for result in model.generate(
    text="Indie pop song with male vocals, emotional, Indonesian",
    lyrics="""[verse]\nLangit mulai redup\nKota kembali diam\n\n[chorus]\nDan aku masih di sini""",
    duration=45.0,
    vocal_language="id",
    use_lm=True,  # required for quality
):
    sf.write("output.wav", result.audio, result.sample_rate)
```

Run existing script:

```bash
source venv/bin/activate && python3 song.py
```

## Fixes

See [`catatan-perbaikan.md`](catatan-perbaikan.md) for full documentation.

1. **Quantized Linear** — Converts `nn.Linear` → `nn.QuantizedLinear` for encoder/decoder before loading 4-bit weights.
2. **5Hz LM** — Enables LM hints for `text2music` (was restricted to `cover` only).

## Model

- **HF:** `mlx-community/ACE-Step1.5-MLX-4bit`
- **Format:** 4-bit quantized, group_size=64
- **Cache:** `~/.cache/huggingface/hub/models--mlx-community--ACE-Step1.5-MLX-4bit/`
- **Output:** 48kHz stereo WAV
