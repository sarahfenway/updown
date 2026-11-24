"""Colours for the step-free pages: each line's colour, as on the Tube map,
so people can find their line at a glance."""

LINE_COLOURS = {
    "Bakerloo line": "#B36305",
    "Central line": "#E32017",
    "Circle line": "#FFD300",
    "District line": "#00782A",
    "Hammersmith & City line": "#F3A9BB",
    "Jubilee line": "#A0A5A9",
    "Metropolitan line": "#9B0056",
    "Northern line": "#000000",
    "Piccadilly line": "#003688",
    "Victoria line": "#0098D4",
    "Waterloo & City line": "#95CDBA",
    "Elizabeth line": "#6950A1",
    "DLR": "#00A4A7",
    "Liberty line": "#61686B",
    "Lioness line": "#FAA61A",
    "Mildmay line": "#437EC1",
    "Suffragette line": "#5BB972",
    "Weaver line": "#893B67",
    "Windrush line": "#D22730",
    "Overground": "#EE7C0E",
}
DEFAULT_COLOUR = "#1C3F94"


def line_colour(line):
    return LINE_COLOURS.get(line, DEFAULT_COLOUR)
