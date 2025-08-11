import re
import unicodedata
from pylatinize import PyLatinize, Normalization, default_mapping, emoji_mapping


def triplet_tokenizer(text, clear_spaces=False, ascii_only=False, latanize=False):
    # use normalized unicode
    text_unicode = unicodedata.normalize("NFKC", text)
    # use only ASCII characters
    text_ascii_only = text_unicode.encode("ascii", "ignore").decode() if ascii_only else text_unicode
    # remove special whitespace & newlines
    text_without_whitespaces = re.sub(r"[\n|\r|\t|\xa0]", " ", text_ascii_only)
    # remove questionmarks
    text_without_questionmarks = re.sub(r"\?", "", text_without_whitespaces)
    # remove fullstops
    text_without_full_stops = re.sub(r"[._]", "", text_without_questionmarks)
    # clear numbers
    text_without_numbers = re.sub(r"\d", "", text_without_full_stops)
    text_space_processsed = re.sub(r"\s", "", text_without_numbers) if clear_spaces else re.sub(r"\s+", " ", text_without_numbers).strip()

    # latinize all letters?
    latinizer_default = PyLatinize((default_mapping, emoji_mapping)) if latanize else None

    text_latinized = latinizer_default.decompose(text_space_processsed) if latanize else text_space_processsed
    # convert to lower case
    text_lower_case = text_latinized.lower()

    block_size = 3
    length = len(text_lower_case)
    trigram_list = []
    for idx in range(0, length-2):
        trigram = text_lower_case[idx] + text_lower_case[idx+1] + text_lower_case[idx+2]
        trigram_list.append(trigram)
    return trigram_list
