"""Readable names for TfL's station area codes (``BookH``, ``CenEB``...)."""

import re

LINE_PREFIXES = {
    "bak": "Bakerloo",
    "cen": "Central",
    "cir": "Circle",
    "dis": "District",
    "ham": "Hammersmith & City",
    "jub": "Jubilee",
    "met": "Metropolitan",
    "nor": "Northern",
    "pic": "Piccadilly",
    "vic": "Victoria",
    "wat": "Waterloo & City",
}
LINE_WORDS = {
    "bakerl": "Bakerloo", "centr": "Central", "circl": "Circle", "distr": "District",
    "hamme": "Hammersmith & City", "jubil": "Jubilee", "metro": "Metropolitan",
    "north": "Northern", "picca": "Piccadilly", "victo": "Victoria",
}
DIRECTIONS = {"eb": "eastbound", "wb": "westbound", "nb": "northbound", "sb": "southbound"}
_LINES = "|".join(LINE_PREFIXES)

# Street furniture outside the station. Dropping these keeps the diagram
# readable and does not cut any station off (checked against the full file).
_DROPPED = re.compile(r"^(bus|taxi)", re.IGNORECASE)

# First match wins, so the specific line-and-direction rules come first.
_RULES = [
    (re.compile(r"^el ?(eb|wb)$"), lambda m: f"Elizabeth line {DIRECTIONS[m[1]]} platform"),
    (re.compile(rf"^({_LINES})(eb|wb|nb|sb)$"),
     lambda m: f"{LINE_PREFIXES[m[1]]} line {DIRECTIONS[m[2]]} platform"),
    (re.compile(rf"^({_LINES})pl$"), lambda m: f"{LINE_PREFIXES[m[1]]} line platforms"),
    (re.compile(r"^(book|th$|.*bh$)"), lambda m: "Ticket hall"),
    # "ent" anywhere: CEntr, MEntr, ELEnt, EntrW are entrances.
    (re.compile(r"^(i?ent|[a-z][-+]|.*entr|.*ent$)"), lambda m: "Entrance"),
    (re.compile(r"^(upcon|up)$"), lambda m: "Upper concourse"),
    (re.compile(r"^locon$"), lambda m: "Lower concourse"),
    (re.compile(r"^micon$"), lambda m: "Middle concourse"),
    (re.compile(r"^con"), lambda m: "Concourse"),
    (re.compile(r"^(fb|fbdg|footb|bridg|brdge)"), lambda m: "Footbridge"),
    (re.compile(r"^sub"), lambda m: "Subway"),
    (re.compile(r"^(rpl|rly|met$|dlr)"), lambda m: "Platform level"),
    # Only the Elizabeth line codes seen in TfL's file: ELVER, ELM or ELTH
    # are other places that happen to start with EL.
    (re.compile(r"^(elel|elwl|ell)$"), lambda m: "Elizabeth line level"),
]

# TfL truncates line names to five or six letters for a line's level
# ("Jubil", "Picca"). Matched whole and in TfL's capitalisation, so that
# "CEntr" (an entrance) or "North" (a direction) aren't taken for lines.
_LINE_WORD_CODE = re.compile(
    r"(Bakerl|Centr|Circl|Distr|Hamme|Jubil|Metro|Picca|Victo)"
    r"|(BAKERL|CENTR|CIRCL|DISTR|HAMME|JUBIL|METRO|PICCA|VICTO)"
)


def is_dropped(code):
    """True for bus stops and taxi ranks, which are not part of the station."""
    return bool(_DROPPED.match(code.strip()))


def place_label(code):
    """Readable name for an area code, or None when we have no rule for it."""
    match = _LINE_WORD_CODE.fullmatch(code.strip())
    if match:
        return f"{LINE_WORDS[match[0].lower()]} line level"
    key = code.strip().lower()
    for pattern, make in _RULES:
        match = pattern.match(key)
        if match:
            return make(match)
    return None
