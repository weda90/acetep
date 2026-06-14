import time
from pathlib import Path

import soundfile as sf
from mlx_audio.tts.utils import load_model

MODEL_ID = "mlx-community/ACE-Step1.5-MLX-4bit"

_model_cache = None


def get_model(model_id: str = MODEL_ID):
    global _model_cache
    if _model_cache is None:
        print(f"Loading model {model_id}...")
        _model_cache = load_model(model_id)
    return _model_cache


def load_lyrics(path_or_text: str) -> str:
    if not path_or_text:
        return ""
    if path_or_text == "-":
        import sys
        return sys.stdin.read().strip()
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return p.read_text().strip()
    return path_or_text


def generate(
    text: str,
    lyrics: str = "",
    duration: float = 30.0,
    output: str = "output.wav",
    vocal_language: str = "unknown",
    use_lm: bool = True,
    lm_model_size: str = "0.6B",
    num_steps: int = 8,
    shift: float = 3.0,
    guidance_scale: float = 1.0,
    bpm=None,
    keyscale=None,
    seed=None,
    model_id: str = MODEL_ID,
    verbose: bool = True,
):
    model = get_model(model_id)
    lyrics_text = load_lyrics(lyrics)

    start = time.time()
    count = 0
    for i, result in enumerate(model.generate(
        text=text,
        lyrics=lyrics_text,
        duration=duration,
        vocal_language=vocal_language,
        use_lm=use_lm,
        lm_model_size=lm_model_size,
        num_steps=num_steps,
        shift=shift,
        guidance_scale=guidance_scale,
        bpm=bpm,
        keyscale=keyscale,
        seed=seed,
        verbose=verbose,
    )):
        out = output if i == 0 else output.replace(".wav", f"_{i}.wav")
        sf.write(out, result.audio, result.sample_rate)
        elapsed = time.time() - start
        print(f"saved: {out} ({elapsed:.1f}s)")
        count += 1

    return count
