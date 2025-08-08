import re
import unicodedata


def triplet_tokenizer(text, clear_spaces=False):
    # use normalized unicode
    text_unicode = unicodedata.normalize("NFKC", text)
    # use only ASCII characters
    text_ascii_only = text_unicode.encode("ascii", "ignore").decode()
    # remove special whitespace & newlines
    text_without_whitespaces = re.sub(r"[\n\r\t\xa0]", " ", text_ascii_only)
    # remove questionmarks
    text_without_questionmarks = re.sub(r"\?", "", text_without_whitespaces)
    # remove fullstops
    text_without_full_stops = re.sub(r"[._]", "", text_without_questionmarks)
    # clear numbers
    text_without_numbers = re.sub(r"\d", "", text_without_full_stops)
    text_space_processsed = re.sub(r"\s", "", text_without_numbers) if clear_spaces else re.sub(r"\s+", " ", text_without_numbers).strip()
    # convert to lower case
    text_lower_case = text_space_processsed.lower()

    block_size = 3
    length = len(text_lower_case)
    trigram_list = []
    for idx in range(0, length-2):
        trigram = text_lower_case[idx] + text_lower_case[idx+1] + text_lower_case[idx+2]
        trigram_list.append(trigram)
    return trigram_list
