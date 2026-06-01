import re


PHONE_NUMBER_RE = re.compile(r"0[38]43\s*222\s*1234")
WHITESPACE_RE = re.compile(r"\s+")


def normalise_incident_text(text):
    if not text:
        return ""

    text = PHONE_NUMBER_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()
