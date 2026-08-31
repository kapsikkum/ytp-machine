"""Where each phoneme actually is, by forced alignment against the sounds.

The splicer cuts sub-word units -- the "at" out of "fat", the Z off the end of
"calls" -- and to do that it needs a time for each phoneme. Until now it got
one by aligning the word's *letters* to the audio and then guessing which
letters make which phoneme, greedily merging a list of known digraphs.

That guess is wrong for a third of the words in a typical corpus, and wrong in
a way that silently shifts every phoneme after the mistake. "calls" is five
letters (c a l l s) and four phonemes (K AO L Z), so asking for the Z lands on
the second L: you ask for a Z and hear the tail of an L. Doubled consonants,
silent E and any digraph missing from the list all do the same.

So the phonemes are aligned directly instead. Not recognised -- recognition on
an isolated 300ms clip is unreliable, and we already know what was said.
*Forced* alignment: the dictionary supplies the sequence, and the model only
has to say where each one falls, which is the part it is good at.

The model is far too big for the server (a 1.2GB model in a container capped
at 3.2GB), so this runs at corpus build time and the result is stored per
clip. The server reads times; it never loads a model.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# ARPAbet (what the dictionary speaks) to the model's espeak IPA (what its
# vocabulary speaks). Stress digits are already stripped everywhere in this
# project, so each entry is one symbol.
#
# AH is the awkward one: CMU uses it for both the stressed "cup" vowel and the
# schwa, and having dropped stress we cannot tell them apart. It maps to the
# schwa, which is the commoner of the two by a wide margin -- and since only
# the boundary is wanted, landing on a neighbouring vowel quality costs a few
# milliseconds rather than the wrong sound.
_ARPA_TO_IPA = {
    "AA": "ɑː", "AE": "æ",  "AH": "ə",  "AO": "ɔː", "AW": "aʊ", "AY": "aɪ",
    "B":  "b",  "CH": "tʃ", "D":  "d",  "DH": "ð",  "EH": "ɛ",  "ER": "ɚ",
    "EY": "eɪ", "F":  "f",  "G":  "ɡ",  "HH": "h",  "IH": "ɪ",  "IY": "iː",
    "JH": "dʒ", "K":  "k",  "L":  "l",  "M":  "m",  "N":  "n",  "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P":  "p",  "R":  "ɹ",  "S":  "s",  "SH": "ʃ",
    "T":  "t",  "TH": "θ",  "UH": "ʊ",  "UW": "uː", "V":  "v",  "W":  "w",
    "Y":  "j",  "Z":  "z",  "ZH": "ʒ",
}

# Second choices, tried when the first is not in the model's vocabulary.
_FALLBACK = {"ɚ": "ɜː", "ɡ": "g", "ɹ": "r", "ɔː": "ɔ", "ɑː": "ɑ",
             "iː": "i", "uː": "u", "ə": "ʌ"}

_STRIDE = 320          # wav2vec2 sees 16kHz audio in 20ms frames
_SR = 16000

_bundle: dict[str, Any] | None = None


def available() -> bool:
    """Can this process align at all? False on the server, and that is fine."""
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _load(device: str | None = None) -> dict[str, Any]:
    """The model, its vocabulary, and the ARPAbet→id table. Loaded once.

    A pass over a whole corpus is one forward per clip, which is the sort of
    work a GPU finishes in minutes and a CPU in the best part of an hour, so
    it honours --device / MRS_DEVICE like every other model in the project.
    """
    global _bundle
    if _bundle is not None:
        return _bundle

    import io
    import json

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor

    # The bundled tokenizer imports phonemizer (and wants an espeak binary)
    # only to turn text into phones, which is the direction we are not going.
    # Reading the model's output needs nothing but its vocabulary.
    from app.device import resolve
    dev = resolve(device)

    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    model = AutoModelForCTC.from_pretrained(MODEL).eval().to(dev)
    vocab = json.load(io.open(hf_hub_download(MODEL, "vocab.json"),
                              encoding="utf-8"))

    ids: dict[str, int] = {}
    missing: list[str] = []
    for arpa, ipa in _ARPA_TO_IPA.items():
        for cand in (ipa, _FALLBACK.get(ipa)):
            if cand and cand in vocab:
                ids[arpa] = vocab[cand]
                break
        else:
            missing.append(arpa)
    if missing:
        log.warning("no model symbol for %s", ", ".join(missing))

    log.info("phoneme aligner on %s", dev)
    _bundle = {"fe": fe, "model": model, "ids": ids, "device": dev,
               "blank": vocab.get("<pad>", 0)}
    return _bundle


def align(wav, phones: list[str],
          device: str | None = None) -> list[tuple[str, float, float]] | None:
    """Place *phones* in *wav* (1-D float32 at 16kHz). Times in seconds.

    Returns one (phone, start, end) per phoneme, or None if the audio is too
    short to hold them or the model has no symbol for one of them.
    """
    import torch
    import torchaudio.functional as F

    if not phones:
        return None
    b = _load(device)
    try:
        targets = [b["ids"][p] for p in phones]
    except KeyError as exc:
        log.warning("cannot align %s: no symbol for %s", " ".join(phones), exc)
        return None

    # CTC needs at least one frame per target, plus room for the blanks it
    # puts between repeats. Short of that, forced_align raises rather than
    # returning a bad answer, which is the right way round but not a crash we
    # want to see per clip.
    if wav.numel() < _STRIDE * (len(targets) + 2):
        return None

    dev = b["device"]
    with torch.inference_mode():
        values = b["fe"](wav, sampling_rate=_SR,
                         return_tensors="pt").input_values.to(dev)
        logits = b["model"](values).logits
        logp = torch.log_softmax(logits, dim=-1)
        try:
            spans, _scores = F.forced_align(
                logp, torch.tensor([targets], dtype=torch.int32, device=dev),
                blank=b["blank"])
        except Exception as exc:
            log.warning("forced alignment failed: %s", exc)
            return None

    # forced_align returns one label per frame, and CTC alignments are spiky:
    # a token is emitted on a frame or two and the rest are blank. So a run's
    # length is not the length of the sound -- taking it as one put the end of
    # the Z in "because" 180ms early, cutting off most of the Z.
    #
    # What a spike does say is where the phoneme *begins*. Each one therefore
    # runs to the start of the next, and the last runs to the end of the
    # audio, which is the ordinary way to read a CTC alignment as segments.
    per_frame = spans[0].cpu().tolist()
    sec = _STRIDE / _SR
    starts: list[int] = []
    i = 0
    for want in targets:
        while i < len(per_frame) and per_frame[i] != want:
            i += 1
        if i >= len(per_frame):
            return None
        starts.append(i)
        while i < len(per_frame) and per_frame[i] == want:
            i += 1

    bounds = starts + [len(per_frame)]
    return [(p, bounds[k] * sec, bounds[k + 1] * sec)
            for k, p in enumerate(phones)]
