# api_server.py
import io
import sys
import hashlib
from pathlib import Path

import torch
import torchaudio as ta
import soundfile as sf
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, Field

# ----------------- Locate chatterbox src -----------------
REPO_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_DIR / "src"
if not SRC_DIR.exists():
    raise RuntimeError(f"Could not find src/ directory at {SRC_DIR}")

sys.path.append(str(SRC_DIR))

# ----------------- Import Chatterbox -----------------
from chatterbox.tts import ChatterboxTTS  # type: ignore

# ----------------- App + Global State -----------------
app = FastAPI(title="Chatterbox URL-Reference API")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Using device: {device}")

print("[*] Loading ChatterboxTTS model...")
tts = ChatterboxTTS.from_pretrained(device=device)
SR = tts.sr
print(f"[*] Model loaded. Sample rate: {SR}")

TMP_REFS_DIR = REPO_DIR / "tmp_refs"
TMP_REFS_DIR.mkdir(exist_ok=True)


class SynthesisRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    reference_url: HttpUrl = Field(..., description="URL to reference audio (Firebase, etc.)")
    exaggeration: float = Field(
        0.5,
        description="Voice exaggeration (matches your Colab usage)",
        ge=0.0,
        le=2.0,
    )


def url_to_cache_path(url: str) -> Path:
    """
    Map a URL to a deterministic local WAV path, so repeated
    URLs reuse the same processed reference file.
    """
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return TMP_REFS_DIR / f"{h}.wav"


def download_and_prepare_reference(url: str) -> Path:
    """
    Download audio from URL, convert to WAV at the model sample rate,
    and cache it under tmp_refs/.
    """
    cache_path = url_to_cache_path(url)
    if cache_path.exists():
        # Already processed/cached
        return cache_path

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download reference audio from URL: {e}",
        )

    # Load audio from bytes
    buf = io.BytesIO(resp.content)
    try:
        waveform, sr = ta.load(buf)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode audio from URL: {e}",
        )

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, n)

    if sr != SR:
        print(f"[*] Resampling reference audio from {sr} Hz to {SR} Hz")
        resampler = ta.transforms.Resample(sr, SR)
        waveform = resampler(waveform)

    # Save as WAV for Chatterbox
    ta.save(str(cache_path), waveform, SR)
    print(f"[*] Saved prepared reference to {cache_path} from {url}")
    return cache_path


def generate_wav_bytes(
    text: str,
    reference_url: str,
    exaggeration: float = 0.5,
) -> bytes:
    """
    Run ChatterboxTTS.generate using an audio prompt from a URL,
    return WAV bytes.
    """
    ref_path = download_and_prepare_reference(reference_url)

    wav_tensor = tts.generate(
        text,
        audio_prompt_path=str(ref_path),
        exaggeration=exaggeration,
    )

    # [channels, samples] -> [samples, channels]
    wav_np = wav_tensor.detach().cpu().numpy().T

    out_buf = io.BytesIO()
    sf.write(out_buf, wav_np, SR, format="WAV")
    out_buf.seek(0)
    return out_buf.read()


# ----------------- Routes -----------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "sample_rate": SR,
    }


@app.post("/synthesize")
async def synthesize(req: SynthesisRequest):
    """
    Stateless synthesis: text + reference_url in a single call.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    try:
        audio_bytes = generate_wav_bytes(
            text=req.text,
            reference_url=str(req.reference_url),
            exaggeration=req.exaggeration,
        )
    except HTTPException:
        # pass through as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="tts_output.wav"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
