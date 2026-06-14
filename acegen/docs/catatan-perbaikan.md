# Catatan Perbaikan ACE-Step1.5 MLX 4-bit

Model: `mlx-community/ACE-Step1.5-MLX-4bit` (HF)
Framework: `mlx-audio` PR #498 (Blaizzy/mlx-audio)

---

## Masalah 1: Shape Mismatch di Text Projector

**Error:**
```
ValueError: [mlx] Linear: input shape last dimension must match weight shape second dimension.
Expected 2048 but got 1024 for input (1, 65, 1024) and weight (128, 2048)
```

**Akar Masalah:**
PR #498 mendefinisikan semua layer sebagai `nn.Linear`, tetapi model yang di-cache (`mlx-community/ACE-Step1.5-MLX-4bit`) menyimpan bobot dalam format **4-bit quantized** untuk `nn.QuantizedLinear`. Bobot quantized disimpan dalam format packed:
- `weight`: `(out_features, in_features / 8)` — packed 4-bit (8 values per uint32)
- `scales`: `(out_features, in_features / group_size)`
- `biases`: `(out_features, in_features / group_size)`

Saat `model.update()` menyalin bobot quantized (shape `(2048, 128)`) ke `nn.Linear` biasa (ekspektasi shape `(2048, 1024)`), dimensi tidak cocok dan matrix multiply gagal.

**Perbaikan (`ace_step.py`):**

1. Tambah fungsi `_quantize_module` sebelum class `Model`:
```python
def _quantize_module(module, group_size=64, bits=4):
    """Recursively convert nn.Linear to nn.QuantizedLinear in a module tree."""
    for key, value in module.items():
        if isinstance(value, nn.Linear) and not isinstance(value, nn.QuantizedLinear):
            setattr(module, key, nn.QuantizedLinear.from_linear(value, group_size, bits))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, nn.Linear) and not isinstance(item, nn.QuantizedLinear):
                    value[i] = nn.QuantizedLinear.from_linear(item, group_size, bits)
                elif isinstance(item, nn.Module):
                    _quantize_module(item, group_size, bits)
        elif isinstance(value, nn.Module):
            _quantize_module(value, group_size, bits)
```

2. Panggil fungsi ini di `from_pretrained`, setelah konstruksi model dan sebelum loading bobot:
```python
quantization = config_dict.get("quantization")
if quantization:
    _quantize_module(
        model.encoder,
        group_size=quantization.get("group_size", 64),
        bits=quantization.get("bits", 4),
    )
    _quantize_module(
        model.decoder,
        group_size=quantization.get("group_size", 64),
        bits=quantization.get("bits", 4),
    )
```

**File yang dimodifikasi:**
- `venv/lib/python3.14/site-packages/mlx_audio/tts/models/ace_step/ace_step.py`

**Komponen yang dikonversi:**
- `model.encoder` (ConditionEncoder): text_projector, lyric_encoder (embed_tokens + 8x EncoderLayer), timbre_encoder (embed_tokens + 4x EncoderLayer)
- `model.decoder` (DiTModel): condition_embedder, time_embed/time_embed_r (3x Linear), 24x DiTLayer (self_attn q/k/v/o, cross_attn q/k/v/o, mlp gate/up/down)

**Tidak dikonversi** (tetap nn.Linear biasa):
- `model.tokenizer` (AudioTokenizer)
- `model.detokenizer` (AudioTokenDetokenizer)

Sesuai config: `"quantized_components": ["decoder", "encoder"]`

---

## Masalah 2: 5Hz LM Tidak Dipakai untuk text2music

**Akar Masalah:**
Di method `generate()`, kode hanya mengaktifkan 5Hz LM untuk task `"cover"`:
```python
elif use_lm and task_type == "cover":
    if verbose:
        print("5Hz LM enabled for cover task")
```

Untuk task `"text2music"`, LM tidak pernah dijalankan meskipun `use_lm=True`. Akibatnya diffusion berjalan tanpa LM hints dan menghasilkan audio noise/garbage.

**Perbaikan (`ace_step.py`):**

Ubah kondisional untuk menjalankan LM untuk semua task type:
```python
elif use_lm:
    if verbose:
        print("Generating 5Hz LM hints...")
    lm_hints = self._generate_lm_hints(
        caption=text,
        lyrics=lyrics,
        duration=int(duration),
        language=vocal_language,
        target_len=latent_len,
        model_size=lm_model_size,
        verbose=verbose,
    )
    if lm_hints is not None:
        is_covers = mx.ones((1,), dtype=self.dtype)
```

**File yang dimodifikasi:**
- `venv/lib/python3.14/site-packages/mlx_audio/tts/models/ace_step/ace_step.py`

---

## Struktur Model

```
Model
├── encoder (ConditionEncoder) ← QUANTIZED
│   ├── text_projector: Linear(1024→2048)
│   ├── lyric_encoder (LyricEncoder)
│   │   ├── embed_tokens: Linear(1024→2048)
│   │   └── layers[0..7]: EncoderLayer (Attention + MLP)
│   └── timbre_encoder (TimbreEncoder)
│       ├── embed_tokens: Linear(64→2048)
│       └── layers[0..3]: EncoderLayer (Attention + MLP)
├── decoder (DiTModel) ← QUANTIZED
│   ├── condition_embedder: Linear(2048→2048)
│   ├── time_embed (TimestepEmbedding): 3x Linear
│   ├── time_embed_r (TimestepEmbedding): 3x Linear
│   └── layers[0..23]: DiTLayer (SelfAttn + CrossAttn + MLP)
├── tokenizer (AudioTokenizer) ← STANDARD
│   ├── audio_acoustic_proj: Linear(64→2048)
│   ├── attention_pooler (AttentionPooler)
│   └── quantizer (ResidualFSQ)
├── detokenizer (AudioTokenDetokenizer) ← STANDARD
│   ├── embed_tokens: Linear(2048→2048)
│   ├── layers[0..1]: EncoderLayer
│   └── proj_out: Linear(2048→64)
├── vae (AutoencoderOobleck) ← bobot terpisah di vae/
└── text_encoder (TextEncoder) ← Qwen3-Embedding-0.6B
```

## Model Cache Location

```
~/.cache/huggingface/hub/models--mlx-community--ACE-Step1.5-MLX-4bit/
└── snapshots/<hash>/
    ├── config.json          # model config + quantization params
    ├── model.safetensors    # bobot quantized (encoder + decoder)
    ├── silence_latent.npy   # silence latent untuk text2music
    ├── Qwen3-Embedding-0.6B/ # text encoder
    └── vae/
        ├── config.json
        ├── mlx_weights.safetensors  # bobot VAE
        └── diffusion_pytorch_model.safetensors
```

## Lokasi File yang Dimodifikasi

```
venv/lib/python3.14/site-packages/mlx_audio/tts/models/ace_step/
├── ace_step.py    ← modifikasi: _quantize_module + LM fix
├── encoders.py    ← (tidak dimodifikasi, quantize via post-processing)
├── dit.py         ← (tidak dimodifikasi)
└── modules.py     ← (tidak dimodifikasi)
```

## Cara Reproduksi

```python
from mlx_audio.tts.utils import load_model
import soundfile as sf

model = load_model("mlx-community/ACE-Step1.5-MLX-4bit")

for result in model.generate(
    text="Indie pop song with male vocals, emotional",
    lyrics="""[verse]\nLyrics here\n[chorus]\nChorus here""",
    duration=30.0,
    vocal_language="en",
    use_lm=True,          # HARUS True untuk kualitas bagus
):
    sf.write("output.wav", result.audio, result.sample_rate)
```
