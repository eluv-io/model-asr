import torch
import librosa
import io
import ffmpeg
from typing import Tuple

SAMPLE_RATE = 16000

def audio_file_to_tensor(fname: str) -> Tuple[torch.Tensor, float]:
    """
    Convert audio file bytes to tensor
    
    Args:
        audio_bytes: Raw audio file bytes (any format supported by ffmpeg)
    
    Returns:
        Tuple of (audio_tensor, duration_in_seconds)
        audio_tensor has shape (1, num_samples)
    """
    # Convert to WAV first
    with open(fname, 'rb') as f:
        audio_bytes = f.read()

    wav_bytes = _to_wav(audio_bytes)
    
    # Load as numpy array
    audio, sr = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=True)
    duration = librosa.get_duration(y=audio, sr=sr)
    
    # Convert to tensor
    audio_tensor = torch.Tensor(audio).unsqueeze(0)
    
    return audio_tensor, duration

def _to_wav(audio: bytes) -> bytes:
    """Convert any audio format to WAV using ffmpeg"""
    process = (
        ffmpeg
        .input('pipe:0', f='mov,mp4,m4a,3gp,3g2,mj2')
        .output('pipe:1', format='wav')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    out, err = process.communicate(input=audio)
    if process.returncode != 0:
        raise Exception(f"ffmpeg error: {err.decode()}")
    return out