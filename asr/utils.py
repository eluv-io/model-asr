from pycaption import SCCReader, SRTReader, DFXPWriter
from bs4 import BeautifulSoup
import re
from num2words import num2words
from word2number import w2n
import datetime
from typing import Tuple
from loguru import logger as LOG
from copy import deepcopy

SAMPLE_RATE = 16000

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


class CaptionLine():
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text

    @staticmethod
    def from_xml(xml: dict):
        start = time_to_seconds(xml["begin"])
        end = time_to_seconds(xml["end"])
        return CaptionLine(start, end, xml.text)

    @staticmethod
    def from_union(lines: list):
        start = lines[0].start
        end = lines[-1].end
        text = ''.join(line.text for line in lines)

        assert start == min(line.start for line in lines)
        assert end == max(line.end for line in lines)

        return CaptionLine(start, end, text)


def remove_extension(filename):
    if filename.endswith('.scc') or filename.endswith('.srt'):
        return filename[0:len(filename)-4]
    else:
        return filename


def time_to_seconds(time: str) -> float:
    start = datetime.datetime.strptime(time, "%H:%M:%S.%f")
    zero_time = datetime.datetime(start.year, start.month, start.day)
    return (start - zero_time).total_seconds()


def re_fn_replace(regexp: str, string: str, fn) -> str:
    pattern = re.compile(regexp, re.DOTALL)
    while True:
        match = pattern.search(string)
        if match is None:
            break
        prefix = string[:match.start()]
        suffix = string[match.end():]
        string = fn(prefix, string[match.start():match.end()], suffix)
    return string


def time_replace(string: str):
    def _time_to_words(prefix, term, suffix):
        left, right = term.split(":")
        left, right = int(left), int(right)
        left = num2words(left)
        if right == 0:
            term = f"{left} o'clock"
        else:
            right = num2words(right)
            term = f"{left} {right}"
        return f"{prefix} {term} {suffix}"

    return re_fn_replace(
        r"[0-9]?[0-9]:[0-9][0-9]",
        string,
        _time_to_words,
    )


CURRENCY_MAP = {
    r"£": "pounds",
    r"\$": "dollars",
}

CURRENCY_SUFFIXES = {
    "trillion",
    "billion",
    "million",
    "thousand"
    "hundred",
}

NUMBER_REGEXP = r"[0-9]+(\.[0-9]*)?|\.[0-9]+"


def currency_replace(currency_map, string):
    def _currency_num_to_word(currency, prefix, term, suffix):
        prefix = prefix.split()
        suffix = suffix.split()

        num = float(term[1:])
        term = num2words(num)

        if term.startswith("one"):
            if len(prefix) > 0 and (prefix[-1] == "a" or prefix[-1] == "one"):
                term = term[4:]
            else:
                term = f"a{term[3:]}"

        # handles special case "$10 hundred million" -> "ten hundred million dollars"
        while len(suffix) > 0 and suffix[0] in CURRENCY_SUFFIXES:
            term = f"{term} {suffix[0]}"
            suffix = suffix[1:]

        suffix = " ".join(suffix)
        prefix = " ".join(prefix)

        return f"{prefix} {term} {currency} {suffix}"

    for symbol, name in currency_map.items():
        string = re_fn_replace(
            symbol + NUMBER_REGEXP,
            string,
            lambda *x: _currency_num_to_word(name, *x),
        )
    return string


def slash_replace(string):
    def _half_or_quarter_replace(prefix, term, suffix):
        #assert not (len(suffix) > 0 and suffix[0].isdigit())

        term = term.strip()
        if len(term) == 3:
            trailing = ""
        else:
            trailing = term[-1]
            term = term[:-1]

        if term == "1/2":
            term = f"a half{trailing}"
        elif term == "1/4":
            term = f"a quarter{trailing}"
        elif term == "3/4":
            term = f"three quarters{trailing}"
        elif term == "1/3":
            term = f"one third"
        elif term == "2/3":
            term = f"two thirds"
        else:
            LOG.warning(f"Don't understand fraction: {string}")
            return ""

        prefix = prefix.lstrip()
        if len(prefix) > 0 and prefix[-1].isdigit():
            term = f"and {term}"

        return f"{prefix} {term} {suffix}"

    # 1. replace 1/2, 3/4 and 1/4 by proper english words
    string = re_fn_replace(
        r"( |^)[1-3]/[2-4]([^0-9]|$)",
        string,
        _half_or_quarter_replace,
    )

    def _saying_slash(prefix, term, suffix):
        assert term[1] == "/"
        return f"{prefix}{term[0]} slash {term[2]}{suffix}"

    # 2. say "slash" when "/" is present between two words
    string = re_fn_replace(
        r"[a-z]/[a-z]",
        string,
        _saying_slash,
    )

    # 3. replace "/" by space when in between two numbers
    # examples :
    #   "24/7" -> "24 7"
    #   "50/50" -> "50 50"
    def _slash_to_space(prefix, term, suffix):
        assert term[1] == "/"
        return f"{prefix}{term[0]} {term[2]}{suffix}"

    string = re_fn_replace(
        r"[0-9]/[0-9]",
        string,
        _slash_to_space,
    )

    # 4. remove all remaining slashes
    string = re.sub(r"/", " ", string)

    return string


def pound_sign_replace(string):
    def _pound_sign_to_number(prefix, term, suffix):
        assert term[0] == "#"
        return f"{prefix} number {term[1:]}{suffix}"
    # 1. replace by "number"
    string = re_fn_replace(
        r"#( )*[0-9]",
        string,
        _pound_sign_to_number,
    )

    # 2. remove pound sign
    string = re.sub(r"#", " ", string)

    return string


def number_replace(string):
    def _convert_number(prefix, term, suffix):
        term = num2words(term)
        term = re.sub(r",", "", term)
        term = re.sub(r"-", " ", term)
        return f"{prefix} {term} {suffix}"

    string = re_fn_replace(
        NUMBER_REGEXP,
        string,
        _convert_number,
    )

    return string


def load_caption_lines(path: str) -> list:
    dw = DFXPWriter()
    if path.endswith('.scc'):
        reader = SCCReader()
    else:
        reader = SRTReader()

    try:
        with open(path, "rt") as f:
            caps = dw.write(reader.read(f.read()))
    except ValueError:
        # some of the caption files are not properly formatted
        LOG.warning(
            f"captions from \"{path}\" couldn't be read"
        )
        return []

    lines = BeautifulSoup(caps, "xml").find_all("p")
    lines = [
        CaptionLine.from_xml(xml)
        for xml in lines
    ]
    return lines

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