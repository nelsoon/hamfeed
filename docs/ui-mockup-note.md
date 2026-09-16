# Feed UI note (Slice 1)

The feed is a single-column, newest-first list of message cards (dark
theme, system fonts). Approved structure per card, top to bottom:

1. Header: monospace timestamp, frequency label, duration chip.
2. Transcript in the largest type, in its source language.
3. Inline audio player with a 1x / 0.75x speed toggle. Never autoplays.
4. Footer: language + confidence badges (green ok / amber low / red
   failed). Failed cards show an error line plus Keep / Drop / Retry /
   Flag buttons instead of a transcript.

Behavior: a sticky filter bar (text search, from/to dates, hide-noise);
a "N new — jump to live" pill when scrolled up instead of yanking
position; a pause-live toggle; "load more" pagination for history.

Identity rows (callsign + name) and FOR-YOU / EMERGENCY signal banners
arrive in a later slice; the card layout already reserves room for them.
