
from num2words import num2words
from word2number import w2n
from typing import Tuple
from copy import deepcopy

def nested_update(original: dict, updates: dict) -> dict:
    original = deepcopy(original)
    def helper(original, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                helper(original[key], value)
            else:
                original[key] = value
    helper(original, updates)
    return original

#check if w represents a number e.g "five"
def _is_numeric_word(w: str) -> bool:
    try:
        w2n.word_to_num(w)
    except Exception:
        return False
    return True

#check if p represents multiword numberical expression such as "one thousand nine hundred and fifty one"
def _is_numeric_phrase(p: list) -> bool:
    try:
        for item in p:
            if not _is_numeric_word(item) and not item == "and":
                return False
        w2n.word_to_num(' '.join(p))
    except Exception:
        return False
    return True

#postprocess stt output... will REMOVE this function when the model is retrained with more robust preprocessing
#Fixing two issues: 
#1. ordinal numbers (i.e fifth, fiftieth) separating from suffix (i.e. five th and fifty th)
#2. years get transcribed algorithmically in a way that doesn't correspond to spoken language, e.g. 1964 -> "one thousand nine hundred and sixy four"
#   we want to transcribe them numerically instead. In extremely rare cases, non dates might get picked up, but this shouldn't affect the perceived quality too much if at all. 
def postprocess(transcript: str, timesteps: list) -> Tuple[str, list]:
    transcript = transcript.split()
    timesteps = timesteps[:]
    min_year = 1100
    max_year = 2500
    min_date_phrase_length = 4 #since, dates are mis-transcribed as long numerical expressions, this min length helps filter out innocent non-dates from postprocess
    
    seek = 1
    idx = 0
    while idx < len(transcript):
        if _is_numeric_word(transcript[idx]):
            while idx+seek <= len(transcript) and _is_numeric_phrase(transcript[idx:idx+seek]):
                seek = seek+1
            seek -= 1
            while seek > 2 and transcript[idx+seek-1] == "and": #"and" can appear in a numeric phrase, but shouldn't appear at the end.
                seek -= 1
            as_number = w2n.word_to_num(' '.join(transcript[idx:idx+seek]))
            if seek > min_date_phrase_length and min_year < as_number < max_year: #then we've probably found a year. 
                transcript[idx] = as_number
                del transcript[idx+1:idx+seek]
                del timesteps[idx+1:idx+seek]
                seek = 1
        idx, seek = idx+seek, 1

    #fix improper ordinal numbers, i.e five th -> fifth
    ordinal_suffixes = ["th", "nd"]
    idx = 0
    while idx < len(transcript)-1:
        if _is_numeric_word(transcript[idx]) and transcript[idx+1] in ordinal_suffixes:
            transcript[idx] = num2words(w2n.word_to_num(transcript[idx]), to="ordinal")
            transcript.pop(idx+1)
            timesteps.pop(idx+1)
        idx += 1

    transcript = list(map(str, transcript))

    return ' '.join(transcript), timesteps