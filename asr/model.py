import os
import torch
import librosa  
from typing import List, Tuple
from loguru import logger
import io
import ffmpeg

import nemo.collections.asr as nemo_asr
from ctcdecode import CTCBeamDecoder
from deepmultilingualpunctuation import PunctuationModel

from .utils import postprocess
from asr.tags import VideoTag

TOKEN_OFFSET = 100
FRAME_SIZE = .04
SAMPLE_RATE = 16000

class EnglishSTT():
    def __init__(self, asr_path: str, lm_path: str):
        self.device = 'cuda'
        load_path = asr_path
        self.model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            load_path, map_location=self.device).eval()
        logger.info(f"Loaded model from {load_path}")
        self.ids_to_text_func = self.model.tokenizer.ids_to_text
        self.ids_to_tokens_func = self.model.tokenizer.ids_to_tokens
        vocab = self.model.decoder.vocabulary + ["_"]
        lm = lm_path
        self.punctuation_model = PunctuationModel()

        self.decoder = CTCBeamDecoder(
            [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab))],
            model_path=lm,
            beam_width=32,
            alpha=0.25,
            beta=0.5,
            blank_id=128,
            num_processes=max(os.cpu_count(), 1),
        )
        logger.debug(f"loading weights from {lm_path} ...")

    def _load_audio(self, audio_bytes: bytes) -> Tuple[torch.Tensor, float]:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        audio_length = librosa.get_duration(y=audio, sr=sr)
        audio = torch.Tensor(audio).to(self.device)
        audio = audio.unsqueeze(0)
        return audio, audio_length

    def _compute_probs(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.to(self.device)
        audio_length = torch.Tensor([audio.size(1)]).to(self.device)
        with torch.no_grad():
            logits, _, _ = self.model(
                input_signal=audio, input_signal_length=audio_length)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def _beamsearch(self, logits: torch.Tensor) -> Tuple[str, float, List[int], List[str]]:
        logits = logits.to(self.device)
        beams, scores, timesteps, out_lens = self.decoder.decode(logits)
        best_candidate = beams[0][0]
        seq_length = out_lens[0][0].item()
        score = scores[0][0].item()
        timesteps = timesteps[0][0]
        timesteps = timesteps[:seq_length] * FRAME_SIZE
        timesteps = timesteps.tolist()
        best_candidate = best_candidate[:seq_length].tolist()
        proxy_chars_seq = [self.decoder._labels[idx] for idx in best_candidate]
        converted_best_candidate = [
            ord(c)-TOKEN_OFFSET for c in proxy_chars_seq]
        tokens = self.ids_to_tokens_func(converted_best_candidate)
        pred_text = self.ids_to_text_func(converted_best_candidate)

        return pred_text, score, timesteps, tokens

    """
    audio_path: path to audio file
    returns: transcription, audio length in seconds (float), list of timestamps per word, 
    """
    def _tag(self, audio_bytes: bytes) -> Tuple[str, float, List[Tuple[str, int]]]:
        as_wav = self.to_wav(audio_bytes)
        audio, audio_length = self._load_audio(as_wav)
        probs = self._compute_probs(audio)
        prediction, _, timesteps, tokens = self._beamsearch(probs)
        timesteps_in_milliseconds = [t*1000 for t in timesteps]
        word_level_timestamps = self._get_word_level_timestamps(
            timesteps_in_milliseconds, tokens)
        prediction, word_level_timestamps = postprocess(prediction, word_level_timestamps)
        timesteps_w_words = list(
            zip(prediction.split(), word_level_timestamps))
        return prediction, audio_length, timesteps_w_words
    
    def to_wav(self, audio: bytes) -> bytes:
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

    def tag(self, audio_bytes: bytes) -> List[VideoTag]:
        tags = []
        _, _, timesteps_w_words = self._tag(audio_bytes)

        for word, ts in timesteps_w_words:
            ts = round(ts) 
            tags.append(VideoTag(
                start_time=ts,
                end_time=ts,
                text=word, 
            ))

        return tags

    def correct_text(self, text: str) -> str:
        if text == "":
            return text
        res = self.capitalize_proper_nouns(text)
        res = self.punctuation_model.restore_punctuation(res)
        if not res.endswith("."):
            res += "."
        sentence_delimiters = ['.', '?', '!']
        # iterate through first character of each sentence and capitalize it
        capitalized = []
        for i, c in enumerate(res):
            if i == 0 or i > 1 and res[i-2] in sentence_delimiters:
                capitalized.append(c.upper())
            else:
                capitalized.append(c)
        return ''.join(capitalized)
    
    def capitalize_proper_nouns(self, sentence: str) -> str:
        # TODO: add back spacy
        return sentence

    def _get_word_level_timestamps(self, timestamps: list, tokens: list) -> list:
        logger.debug(f"get_word_level_timestamps")
        word_timestamps = []
        for ts, tok in zip(timestamps, tokens):
            if tok.startswith('▁'):
                word_timestamps.append(ts)
        return word_timestamps
