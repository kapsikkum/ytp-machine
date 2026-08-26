# Source packs

Drop corpus bundles here — `something.tar.zst` or `something.tar.gz` — and the
container installs each one on start as a corpus named after the file.
`attenborough.tar.gz` becomes the corpus `attenborough`, selectable in the
"voice" dropdown.

Build one from a working corpus with:

```bash
python scripts/corpus.py pack --out packs/attenborough.tar.zst
```

Installing is skipped when a corpus of that name already exists, so leaving
bundles here costs nothing on restart. Bundles themselves are gitignored; this
file is here so the directory exists for the compose mount.
