import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
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


def _latent_len(duration: float) -> int:
    pool = 5
    n = int(duration * 25)
    if n % pool != 0:
        n = (n // pool + 1) * pool
    return n


def _split_lyrics(lyrics: str, n_chunks: int) -> list[str]:
    if not lyrics:
        return [""] * n_chunks
    lines = lyrics.split("\n")
    total = len(lines)
    out = []
    for i in range(n_chunks):
        a = int(i * total / n_chunks)
        b = int((i + 1) * total / n_chunks)
        out.append("\n".join(lines[a:b]))
    return out


def _crossfade(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
    ol = min(overlap, len(a), len(b))
    if ol <= 0:
        return np.concatenate([a, b])
    if a.ndim == 1:
        fade_out = np.linspace(1, 0, ol)
        fade_in = np.linspace(0, 1, ol)
    else:
        fade_out = np.linspace(1, 0, ol).reshape(-1, 1)
        fade_in = np.linspace(0, 1, ol).reshape(-1, 1)
    blended = a[-ol:] * fade_out + b[:ol] * fade_in
    return np.concatenate([a[:-ol], blended, b[ol:]])


def _chunked_generate(model, text: str, lyrics: str, duration: float,
                      vocal_language: str, use_lm: bool, lm_model_size: str,
                      num_steps: int, shift: float, guidance_scale: float,
                      bpm, keyscale, seed, overlap: float,
                      chunk_duration: float, consistent: bool,
                      verbose: bool) -> tuple[np.ndarray, int]:
    n_chunks = math.ceil(duration / chunk_duration)
    chunk_len = (duration + overlap * (n_chunks - 1)) / n_chunks

    print(f"Chunked mode: {n_chunks} chunks x {chunk_len:.1f}s + {overlap}s overlap")

    lyrics_chunks = _split_lyrics(lyrics, n_chunks)

    full_hints = None
    if use_lm and consistent:
        print("Generating song blueprint (5Hz LM)...")
        try:
            full_hints = model._generate_lm_hints(
                caption=text,
                lyrics=lyrics,
                duration=math.ceil(duration),
                language=vocal_language,
                target_len=_latent_len(duration),
                seed=seed,
                model_size=lm_model_size,
                verbose=verbose,
            )
            if full_hints is not None:
                print(f"  Blueprint: {full_hints.shape[1]} frames at 25Hz")
            else:
                print("  LM returned None, falling back to per-chunk LM")
        except Exception as e:
            print(f"  LM failed ({e}), falling back to per-chunk LM")
            full_hints = None

    audio_chunks = []
    sample_rate = None
    non_overlap = chunk_len - overlap

    for i in range(n_chunks):
        print(f"  Chunk {i + 1}/{n_chunks}...", end=" ", flush=True)
        chunk_seed = seed + i if seed is not None else None

        if full_hints is not None:
            chunk_latent_len = _latent_len(chunk_len)
            start_frame = int(i * non_overlap * 25)
            end_frame = start_frame + chunk_latent_len
            if end_frame > full_hints.shape[1]:
                chunk_hints = full_hints[:, start_frame:, :]
                pad_len = end_frame - full_hints.shape[1]
                pad = mx.repeat(chunk_hints[:, -1:, :], pad_len, axis=1)
                chunk_hints = mx.concatenate([chunk_hints, pad], axis=1)
            else:
                chunk_hints = full_hints[:, start_frame:end_frame, :]
        else:
            chunk_hints = None

        for result in model.generate(
            text=text,
            lyrics=lyrics_chunks[i],
            duration=chunk_len,
            lm_precomputed_hints=chunk_hints,
            use_lm=(full_hints is None and use_lm),
            vocal_language=vocal_language,
            lm_model_size=lm_model_size,
            num_steps=num_steps,
            shift=shift,
            guidance_scale=guidance_scale,
            bpm=bpm,
            keyscale=keyscale,
            seed=chunk_seed,
            verbose=False,
        ):
            audio_chunks.append(result.audio)
            if sample_rate is None:
                sample_rate = result.sample_rate
            print("done")
            break

    result = audio_chunks[0]
    overlap_samples = int(overlap * sample_rate)
    for i in range(1, n_chunks):
        result = _crossfade(result, audio_chunks[i], overlap_samples)

    return result, sample_rate


def generate(
    text: str,
    lyrics: str = "",
    duration: float = 30.0,
    output: str = "output.wav",
    vocal_language: str = "en",
    use_lm: bool = True,
    lm_model_size: str = "0.6B",
    num_steps: int = 8,
    shift: float = 3.0,
    guidance_scale: float = 1.0,
    bpm=None,
    keyscale=None,
    seed=None,
    model_id: str = MODEL_ID,
    chunk_duration: float = 0,
    consistent: bool = True,
    verbose: bool = True,
):
    model = get_model(model_id)
    lyrics_text = load_lyrics(lyrics)

    if chunk_duration > 0 and duration > chunk_duration:
        overlap = 2.0
        audio, sr = _chunked_generate(
            model, text, lyrics_text, duration,
            vocal_language, use_lm, lm_model_size,
            num_steps, shift, guidance_scale,
            bpm, keyscale, seed, overlap, chunk_duration, consistent, verbose,
        )
        sf.write(output, audio, sr)
        print(f"saved: {output} ({audio.shape[0] / sr:.1f}s)")
        return 1

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
