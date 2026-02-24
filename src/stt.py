import os
import torch
from typing import List, Tuple
from loguru import logger

import nemo.collections.asr as nemo_asr
from ctcdecode import CTCBeamDecoder

from .utils import postprocess
from src.tags import ModelTag

TOKEN_OFFSET = 100
FRAME_SIZE = .04

class EnglishSTT():
    """Pure STT model - takes tensor, outputs word-level tags"""
    
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

    def _get_word_level_timestamps(self, timestamps: list, tokens: list) -> list:
        logger.debug(f"get_word_level_timestamps")
        word_timestamps = []
        for ts, tok in zip(timestamps, tokens):
            if tok.startswith('▁'):
                word_timestamps.append(ts)
        return word_timestamps

    def tag(self, audio_tensor: torch.Tensor) -> List[ModelTag]:
        """
        Core STT: tensor -> word-level tags (no prettification)
        
        Args:
            audio_tensor: torch.Tensor of shape (1, num_samples)
        
        Returns:
            List of ModelTag with word-level timestamps
        """
        probs = self._compute_probs(audio_tensor)
        prediction, _, timesteps, tokens = self._beamsearch(probs)
        timesteps_in_milliseconds = [t*1000 for t in timesteps]
        word_level_timestamps = self._get_word_level_timestamps(
            timesteps_in_milliseconds, tokens)
        prediction, word_level_timestamps = postprocess(prediction, word_level_timestamps)
        
        tags = []
        for word, ts in zip(prediction.split(), word_level_timestamps):
            if word.lower() == "d":
                continue
            ts = round(ts)
            tags.append(ModelTag(
                start_time=ts,
                end_time=int(ts+FRAME_SIZE*1000),
                text=word,
            ))
        
        # return sorted by start_time
        tags.sort(key=lambda t: t.start_time)
        return tags