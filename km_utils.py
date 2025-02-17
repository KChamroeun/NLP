from itertools import chain
import km_constant as kc
import re 

def is_accept_char(text: str) -> bool:
    khmer_characters = (
        kc.KM_CONSONANT 
        | kc.ENGU_CONSONANT 
        | kc.ENGL_CONSONANT
        | kc.KM_DEP_VOWEL
        | kc.KM_INDEP_VOWEL
        | kc.KM_NUMBER
        | kc.EN_NUMBER
        | kc.KM_SUP
        | kc.KM_SUB
    )
    regex = re.compile(r"^[{}]+$".format(khmer_characters))
    return regex.match(text) is not None

def get_km_only(list_text: list) -> list:
    return [split_str_on_number_eng(text) for text in list_text if is_accept_char(text)]

def get_accepted_only(list_text: list) -> list:
    return [split_str_on_number_eng(text) for text in list_text]

def rm_zero_w_space(word: str) -> str:
    # return ''.join(c for c in word if c != '\u200b' and c != '\ufeff').strip()
    text = re.sub(r'\u200b', '', word)
    return re.sub(r'\ufeff', '', text).strip()


def rm_all_space(word: str) -> str:
    return re.sub(r'\s', '', word).strip()

def split_str_on_number_eng(text: str) -> list:
    parts = split_str_on_number(text)
    return flatten([split_str_on_eng(part) for part in parts])

def split_str_on_number(text: str) -> list:
    number = kc.KM_NUMBER | kc.EN_NUMBER
    regex = re.compile(r"[{}]+".format(number))
    # regex = re.compile(r"^(?:[+-]|\()?\$?\d+(?:,\d+)*(?:\.\d+)?\)?$")
    matches = regex.finditer(text)
    num_idxs = []
    for match in matches:
        start = match.start()
        end = match.end()
        num_idxs.append(start)
        num_idxs.append(end)
    if len(num_idxs) == 0:
        return [text]
    if 0 not in num_idxs:
        num_idxs.insert(0, 0)
    if len(text) not in num_idxs:
        num_idxs.append(len(text))
    parts = []
    for a, b in zip(num_idxs[:-1], num_idxs[1:]):
        parts.append(text[a:b])
    return parts

def split_str_on_eng(text: str) -> list:
    eng = kc.ENGL_CONSONANT | kc.ENGU_CONSONANT
    regex = re.compile(r"[{}]+".format(eng))
    # regex = re.compile(r"^(?:[+-]|\()?\$?\d+(?:,\d+)*(?:\.\d+)?\)?$")
    matches = regex.finditer(text)
    num_idxs = []
    for match in matches:
        start = match.start()
        end = match.end()
        num_idxs.append(start)
        num_idxs.append(end)
    if len(num_idxs) == 0:
        return [text]
    if 0 not in num_idxs:
        num_idxs.insert(0, 0)
    if len(text) not in num_idxs:
        num_idxs.append(len(text))
    parts = []
    for a, b in zip(num_idxs[:-1], num_idxs[1:]):
        parts.append(text[a:b])
    return parts

def flatten(list2d: list) -> list:
    return list(chain(*list2d))