# Tender data story

`make_story.py` builds one self-contained HTML page that follows Tender X-ray
(SSRL BL 6-2a) data from a raw detector image to an interpreted spectrum, one
chapter per reduction step. Every figure is computed from real `.sif` data with
`tender_analysis` and embedded as a PNG, so the page loads nothing from anywhere
(no fonts, no scripts, no CDN).

```bash
python docs/story/make_story.py --out story.html                      # bundled data only
python docs/story/make_story.py --real /path/to/story_data --out story.html
```

- `--bundled DIR` (default `data/Na2SO4`): a RIXS series; drives chapters 1-10.
- `--real DIR` (optional) may contain `Na2SO4_pellet/` (a second series, for
  averaging), `Ag2S/`, `AgNO3/`, `P12S/` (Ag L3-valence XES) and `elastic_BN/`
  (elastic scans for the pixel to energy calibration). Chapters whose folder is
  missing are skipped with a one-line note.
- `--out FILE`: the HTML to write.

Needs numpy, scipy and matplotlib; xraylarch (or chemcat's `xas_core`) for the
normalisation. A full run with all the real data takes about 3 minutes and
peaks near 6 GB of memory (the elastic and XES files are ~100-frame stacks).

The generated HTML is not committed.
