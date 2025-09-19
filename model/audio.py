import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Union, Optional

from .ultils import exact_div

import numpy as np
import torch
from torch.nn import functional as fn

# hard-coded audio hyperparameters
SAMPLE_RATE = 1600
N_FFT= 400
HOP_LENTGH = 160
CHUNK_LENGTH =30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE # 48000 samples
N_FRAMES = exact_div(SAMPLE_RATE, HOP_LENTGH) # 3000 frames

N_SAMPLES_PER_TOKEN = HOP_LENTGH  * 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENTGH)
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)

def load_audio(file:str, sr:int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    # FFmpeg command to decode an audio file to raw mono PCM at the target sample rate:
    # - Uses all CPU threads for speed
    # - Reads the given input file
    # - Downmixes to 1 channel (mono)
    # - Resamples to the specified sample rate (sr)
    # - Outputs raw 16‑bit PCM little‑endian data (s16le) via stdout (pipe)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    
    # -1 1 normalizationhh
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length:int=N_SAMPLES, *, axis:int=-1):
    # i cant do Union tensor and ndarray because of statc error checking 
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """

    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array=array.index_select(dim=axis, index=torch.arange(length, device=array.device))
        
        if array.shape[axis] < length:
            pad_widths = [(0,0)] * array.ndim
            pad_widths[axis] = (0, length-array.shape[axis])
            array = fn.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0,0)] * array.ndim
            pad_widths[axis] = (0, length-array.shape[axis])
            array = np.pad(array, pad_widths)


    return array
        

@lru_cache(maxsize=None)
def mel_filters(device, n_mels:int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128},  f"Unsupported n_mels: {n_mels}"

    filter_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "mel_filters.npz")
    with np.load(filter_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding:int = 0,
    device: Optional[Union[str, torch.device]]= None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """    
    # skip the load from str
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding >0:
        audio = fn.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENTGH, window=window, return_complex=True)
    magnitudes = stft[...,:-1].abs() ** 2

    mel_filter = mel_filters(audio.device, n_mels)
    mel_spec = mel_filter @ magnitudes

    log_spec= torch.clamp(mel_spec, min=1e-10)
    log_spec = torch.maximum(log_spec, log_spec.max() -8.0)
    # same as log_spec = torch.clamp(log_spec, min= log_spec.max()-8.0); untested
    log_spec = (log_spec+4.0)/4.0
    return log_spec
