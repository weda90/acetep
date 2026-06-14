from mlx_audio.tts.utils import load_model
import soundfile as sf

print("Loading model...")
model = load_model("mlx-community/ACE-Step1.5-MLX-4bit")

lyrics = """[verse]
Langit mulai redup
Kota kembali diam

[chorus]
Dan aku masih di sini
Menunggu pagi kembali"""

print("Generating...")

for i, result in enumerate(
    model.generate(
        text="Indie pop song with male vocals, emotional, Indonesian",
        lyrics=lyrics,
        duration=120.0,
        vocal_language="id",
        use_lm=True,
        lm_model_size="0.6B",
    )
):
    out = f"hasil_{i}.wav"
    sf.write(out, result.audio, result.sample_rate)
    print("saved:", out)
