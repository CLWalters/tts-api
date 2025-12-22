


import io
import sys
import hashlib
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio as ta
import soundfile as sf
import requests
import anyio
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

# Optional: reduce CPU oversubscription (helps memory spikes on multi-core hosts)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

print("[*] Loading ChatterboxTTS model...")
tts = ChatterboxTTS.from_pretrained(device=device)
SR = tts.sr
print(f"[*] Model loaded. Sample rate: {SR}")

TMP_REFS_DIR = REPO_DIR / "tmp_refs"
TMP_REFS_DIR.mkdir(exist_ok=True)

# ----------------- Queue / Worker Configuration -----------------
# Hard limit so pending requests don't pile up and eat RAM
MAX_QUEUE_SIZE = int(os.environ.get("TTS_MAX_QUEUE_SIZE", "32"))
# Optional: timeout waiting in queue (seconds). 0 disables.
QUEUE_WAIT_TIMEOUT_S = float(os.environ.get("TTS_QUEUE_WAIT_TIMEOUT_S", "0"))

# Single queue, single worker => only 1 synth runs at a time
_synth_queue: "asyncio.Queue[SynthJob]" = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
_worker_task: Optional[asyncio.Task] = None


class SynthesisRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    reference_url: HttpUrl = Field(..., description="URL to reference audio (Firebase, etc.)")
    exaggeration: float = Field(
        0.5,
        description="Voice exaggeration (matches your Colab usage)",
        ge=0.0,
        le=2.0,
    )


@dataclass
class SynthJob:
    text: str
    reference_url: str
    exaggeration: float
    future: "asyncio.Future[bytes]"


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
    return MP3 bytes (low quality for speed).
    """
    ref_path = download_and_prepare_reference(reference_url)

    # Keep inference memory lower / avoid autograd graphs
    with torch.inference_mode():
        wav_tensor = tts.generate(
            text,
            audio_prompt_path=str(ref_path),
            exaggeration=exaggeration,
        )

    # [channels, samples]
    audio = wav_tensor.detach().cpu()

    # --- lowest "sane" quality for smaller files ---
    # 1) force mono
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)  # (1, samples)

    # 2) downsample for smaller size (e.g. 16 kHz)
    TARGET_SR = 16000
    out_sr = TARGET_SR
    if SR != TARGET_SR:
        print(f"[*] Downsampling output from {SR} Hz to {TARGET_SR} Hz")
        resampler = ta.transforms.Resample(SR, TARGET_SR)
        audio = resampler(audio)
    # audio shape: (1, samples)

    # 3) convert to [samples, channels] for soundfile
    wav_np = audio.numpy().T  # (samples, 1)

    out_buf = io.BytesIO()
    # NOTE: requires libsndfile built with MP3 support.
    sf.write(out_buf, wav_np, out_sr, format="MP3")
    out_buf.seek(0)

    # Proactively release GPU cached memory (helps on bursty traffic)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return out_buf.read()


async def _synth_worker() -> None:
    """
    Background worker that processes synth jobs sequentially (1 at a time).
    Heavy work runs in a worker thread so the FastAPI event loop stays responsive.
    """
    print("[*] Synth worker started (single concurrency).")
    while True:
        job = await _synth_queue.get()
        try:
            audio_bytes = await anyio.to_thread.run_sync(
                generate_wav_bytes,
                job.text,
                job.reference_url,
                job.exaggeration,
            )
            if not job.future.cancelled():
                job.future.set_result(audio_bytes)
        except HTTPException as e:
            if not job.future.cancelled():
                job.future.set_exception(e)
        except Exception as e:
            if not job.future.cancelled():
                job.future.set_exception(
                    HTTPException(status_code=500, detail=f"TTS generation failed: {e}")
                )
        finally:
            _synth_queue.task_done()


@app.on_event("startup")
async def _on_startup():
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_synth_worker())


@app.on_event("shutdown")
async def _on_shutdown():
    global _worker_task
    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except Exception:
            pass


# ----------------- Routes -----------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "sample_rate": SR,
        "queue_size": _synth_queue.qsize(),
        "queue_max": MAX_QUEUE_SIZE,
        "worker_running": _worker_task is not None and not _worker_task.done(),
    }


@app.post("/synthesize")
async def synthesize(req: SynthesisRequest):
    """
    Stateless synthesis: text + reference_url in a single call.
    All requests are queued and processed strictly one-at-a-time.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    # Backpressure: if queue is full, reject quickly instead of OOM-ing later
    if _synth_queue.full():
        raise HTTPException(
            status_code=429,
            detail=f"Server busy (queue full: max {MAX_QUEUE_SIZE}). Try again shortly.",
        )

    loop = asyncio.get_running_loop()
    fut: "asyncio.Future[bytes]" = loop.create_future()

    job = SynthJob(
        text=req.text,
        reference_url=str(req.reference_url),
        exaggeration=req.exaggeration,
        future=fut,
    )

    await _synth_queue.put(job)

    try:
        if QUEUE_WAIT_TIMEOUT_S and QUEUE_WAIT_TIMEOUT_S > 0:
            audio_bytes = await asyncio.wait_for(fut, timeout=QUEUE_WAIT_TIMEOUT_S)
        else:
            audio_bytes = await fut
    except asyncio.TimeoutError:
        # If the caller timed out waiting, we cancel their future.
        fut.cancel()
        raise HTTPException(
            status_code=504,
            detail="Timed out waiting in queue. Try again with shorter text or later.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="tts_output.mp3"'},
    )


if __name__ == "__main__":
    import uvicorn
    import os

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


























# # api_server.py
# import io
# import sys
# import hashlib
# from pathlib import Path

# import torch
# import torchaudio as ta
# import soundfile as sf
# import requests
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel, HttpUrl, Field

# # ----------------- Locate chatterbox src -----------------
# REPO_DIR = Path(__file__).resolve().parent
# SRC_DIR = REPO_DIR / "src"
# if not SRC_DIR.exists():
#     raise RuntimeError(f"Could not find src/ directory at {SRC_DIR}")

# sys.path.append(str(SRC_DIR))

# # ----------------- Import Chatterbox -----------------
# from chatterbox.tts import ChatterboxTTS  # type: ignore

# # ----------------- App + Global State -----------------
# app = FastAPI(title="Chatterbox URL-Reference API")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"[*] Using device: {device}")

# print("[*] Loading ChatterboxTTS model...")
# tts = ChatterboxTTS.from_pretrained(device=device)
# SR = tts.sr
# print(f"[*] Model loaded. Sample rate: {SR}")

# TMP_REFS_DIR = REPO_DIR / "tmp_refs"
# TMP_REFS_DIR.mkdir(exist_ok=True)


# class SynthesisRequest(BaseModel):
#     text: str = Field(..., description="Text to synthesize")
#     reference_url: HttpUrl = Field(..., description="URL to reference audio (Firebase, etc.)")
#     exaggeration: float = Field(
#         0.5,
#         description="Voice exaggeration (matches your Colab usage)",
#         ge=0.0,
#         le=2.0,
#     )


# def url_to_cache_path(url: str) -> Path:
#     """
#     Map a URL to a deterministic local WAV path, so repeated
#     URLs reuse the same processed reference file.
#     """
#     h = hashlib.sha1(url.encode("utf-8")).hexdigest()
#     return TMP_REFS_DIR / f"{h}.wav"


# def download_and_prepare_reference(url: str) -> Path:
#     """
#     Download audio from URL, convert to WAV at the model sample rate,
#     and cache it under tmp_refs/.
#     """
#     cache_path = url_to_cache_path(url)
#     if cache_path.exists():
#         # Already processed/cached
#         return cache_path

#     try:
#         resp = requests.get(url, timeout=20)
#         resp.raise_for_status()
#     except Exception as e:
#         raise HTTPException(
#             status_code=502,
#             detail=f"Failed to download reference audio from URL: {e}",
#         )

#     # Load audio from bytes
#     buf = io.BytesIO(resp.content)
#     try:
#         waveform, sr = ta.load(buf)
#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Failed to decode audio from URL: {e}",
#         )

#     if waveform.ndim == 1:
#         waveform = waveform.unsqueeze(0)  # (1, n)

#     if sr != SR:
#         print(f"[*] Resampling reference audio from {sr} Hz to {SR} Hz")
#         resampler = ta.transforms.Resample(sr, SR)
#         waveform = resampler(waveform)

#     # Save as WAV for Chatterbox
#     ta.save(str(cache_path), waveform, SR)
#     print(f"[*] Saved prepared reference to {cache_path} from {url}")
#     return cache_path


# def generate_wav_bytes(
#     text: str,
#     reference_url: str,
#     exaggeration: float = 0.5,
# ) -> bytes:
#     """
#     Run ChatterboxTTS.generate using an audio prompt from a URL,
#     return MP3 bytes (low quality for speed).
#     """
#     ref_path = download_and_prepare_reference(reference_url)

#     wav_tensor = tts.generate(
#         text,
#         audio_prompt_path=str(ref_path),
#         exaggeration=exaggeration,
#     )

#     # [channels, samples]
#     audio = wav_tensor.detach().cpu()

#     # --- lowest "sane" quality for smaller files ---
#     # 1) force mono
#     if audio.ndim == 2 and audio.shape[0] > 1:
#         audio = audio.mean(dim=0, keepdim=True)  # (1, samples)

#     # 2) downsample for smaller size (e.g. 16 kHz)
#     TARGET_SR = 16000
#     out_sr = TARGET_SR
#     if SR != TARGET_SR:
#         print(f"[*] Downsampling output from {SR} Hz to {TARGET_SR} Hz")
#         resampler = ta.transforms.Resample(SR, TARGET_SR)
#         audio = resampler(audio)
#     # audio shape: (1, samples)

#     # 3) convert to [samples, channels] for soundfile
#     wav_np = audio.numpy().T  # (samples, 1)

#     out_buf = io.BytesIO()
#     # NOTE: requires libsndfile built with MP3 support.
#     # If this errors, you'll need ffmpeg/pydub; but structure stays the same.
#     sf.write(out_buf, wav_np, out_sr, format="MP3")
#     out_buf.seek(0)
#     return out_buf.read()


# # ----------------- Routes -----------------


# @app.get("/health")
# async def health():
#     return {
#         "status": "ok",
#         "device": device,
#         "sample_rate": SR,
#     }


# @app.post("/synthesize")
# async def synthesize(req: SynthesisRequest):
#     """
#     Stateless synthesis: text + reference_url in a single call.
#     """
#     if not req.text.strip():
#         raise HTTPException(status_code=400, detail="Text must not be empty")

#     try:
#         audio_bytes = generate_wav_bytes(
#             text=req.text,
#             reference_url=str(req.reference_url),
#             exaggeration=req.exaggeration,
#         )
#     except HTTPException:
#         # pass through as-is
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

#     return StreamingResponse(
#         io.BytesIO(audio_bytes),
#         media_type="audio/mpeg",
#         headers={"Content-Disposition": 'inline; filename="tts_output.mp3"'},
#     )


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(
#         "api_server:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=False,
#     )
































# api_server.py
# import io
# import sys
# import hashlib
# from pathlib import Path

# import torch
# import torchaudio as ta
# import soundfile as sf
# import requests
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel, HttpUrl, Field

# # ----------------- Locate chatterbox src -----------------
# REPO_DIR = Path(__file__).resolve().parent
# SRC_DIR = REPO_DIR / "src"
# if not SRC_DIR.exists():
#     raise RuntimeError(f"Could not find src/ directory at {SRC_DIR}")

# sys.path.append(str(SRC_DIR))

# # ----------------- Import Chatterbox -----------------
# from chatterbox.tts import ChatterboxTTS  # type: ignore

# # ----------------- App + Global State -----------------
# app = FastAPI(title="Chatterbox URL-Reference API")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"[*] Using device: {device}")

# print("[*] Loading ChatterboxTTS model...")
# tts = ChatterboxTTS.from_pretrained(device=device)
# SR = tts.sr
# print(f"[*] Model loaded. Sample rate: {SR}")

# TMP_REFS_DIR = REPO_DIR / "tmp_refs"
# TMP_REFS_DIR.mkdir(exist_ok=True)


# class SynthesisRequest(BaseModel):
#     text: str = Field(..., description="Text to synthesize")
#     reference_url: HttpUrl = Field(..., description="URL to reference audio (Firebase, etc.)")
#     exaggeration: float = Field(
#         0.5,
#         description="Voice exaggeration (matches your Colab usage)",
#         ge=0.0,
#         le=2.0,
#     )


# def url_to_cache_path(url: str) -> Path:
#     """
#     Map a URL to a deterministic local WAV path, so repeated
#     URLs reuse the same processed reference file.
#     """
#     h = hashlib.sha1(url.encode("utf-8")).hexdigest()
#     return TMP_REFS_DIR / f"{h}.wav"


# def download_and_prepare_reference(url: str) -> Path:
#     """
#     Download audio from URL, convert to WAV at the model sample rate,
#     and cache it under tmp_refs/.
#     """
#     cache_path = url_to_cache_path(url)
#     if cache_path.exists():
#         # Already processed/cached
#         return cache_path

#     try:
#         resp = requests.get(url, timeout=20)
#         resp.raise_for_status()
#     except Exception as e:
#         raise HTTPException(
#             status_code=502,
#             detail=f"Failed to download reference audio from URL: {e}",
#         )

#     # Load audio from bytes
#     buf = io.BytesIO(resp.content)
#     try:
#         waveform, sr = ta.load(buf)
#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Failed to decode audio from URL: {e}",
#         )

#     if waveform.ndim == 1:
#         waveform = waveform.unsqueeze(0)  # (1, n)

#     if sr != SR:
#         print(f"[*] Resampling reference audio from {sr} Hz to {SR} Hz")
#         resampler = ta.transforms.Resample(sr, SR)
#         waveform = resampler(waveform)

#     # Save as WAV for Chatterbox
#     ta.save(str(cache_path), waveform, SR)
#     print(f"[*] Saved prepared reference to {cache_path} from {url}")
#     return cache_path


# def generate_wav_bytes(
#     text: str,
#     reference_url: str,
#     exaggeration: float = 0.5,
# ) -> bytes:
#     """
#     Run ChatterboxTTS.generate using an audio prompt from a URL,
#     return WAV bytes.
#     """
#     ref_path = download_and_prepare_reference(reference_url)

#     wav_tensor = tts.generate(
#         text,
#         audio_prompt_path=str(ref_path),
#         exaggeration=exaggeration,
#     )

#     # [channels, samples] -> [samples, channels]
#     wav_np = wav_tensor.detach().cpu().numpy().T

#     out_buf = io.BytesIO()
#     sf.write(out_buf, wav_np, SR, format="WAV")
#     out_buf.seek(0)
#     return out_buf.read()


# # ----------------- Routes -----------------


# @app.get("/health")
# async def health():
#     return {
#         "status": "ok",
#         "device": device,
#         "sample_rate": SR,
#     }


# @app.post("/synthesize")
# async def synthesize(req: SynthesisRequest):
#     """
#     Stateless synthesis: text + reference_url in a single call.
#     """
#     if not req.text.strip():
#         raise HTTPException(status_code=400, detail="Text must not be empty")

#     try:
#         audio_bytes = generate_wav_bytes(
#             text=req.text,
#             reference_url=str(req.reference_url),
#             exaggeration=req.exaggeration,
#         )
#     except HTTPException:
#         # pass through as-is
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

#     return StreamingResponse(
#         io.BytesIO(audio_bytes),
#         media_type="audio/wav",
#         headers={"Content-Disposition": 'inline; filename="tts_output.wav"'},
#     )


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(
#         "api_server:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=False,
#     )
