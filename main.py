from score_to_tokens import MusicXML_to_tokens
from tokens_to_score import tokens_to_score

tokens = MusicXML_to_tokens(
    r"C:\Users\VIPLAB\Downloads\clementi-sonatina-no-1-op-36.mxl",
    bar_major=True,           # bar-major style (recommended)
    note_name=True,           # use note names like C4, not MIDI numbers
    tokenize_chord_symbols=True
)

s = tokens_to_score(tokens)
s.write('musicxml', 'output_score')  # saves output_score.musicxml