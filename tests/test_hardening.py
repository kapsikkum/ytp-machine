#!/usr/bin/env python3
"""The fixes from the security and concurrency audit, each held in place.

    python tests/test_hardening.py

A corpus whose database points outside itself, a cache dropped while a
sentence is using it, a menu looking up another voice mid-sentence, a queue
filled up, and a write from another site. The HTTP half needs FastAPI and
steps aside without it, as test_splice_editor's does.
"""
import os
import sqlite3
import sys
import tempfile

data = tempfile.mkdtemp(prefix="ytp-hard-")
os.environ["MRS_DATA_DIR"] = data
os.environ["MRS_CORPUS"] = "a"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database as db  # noqa: E402
import app.generate as g  # noqa: E402
from app import jobs  # noqa: E402

failures: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        failures.append(label)


secret = os.path.join(data, "secret.json")
open(secret, "w").write("{}")
for slug, word in (("a", "alpha"), ("b", "beta")):
    d = os.path.join(data, "corpora", slug)
    os.makedirs(os.path.join(d, "downloads"))
    conn = sqlite3.connect(os.path.join(d, "corpus.db"))
    conn.executescript(f"""
        CREATE TABLE sources (id INTEGER PRIMARY KEY, video_id TEXT, source_file TEXT, title TEXT, url TEXT);
        CREATE TABLE word_clips (id INTEGER PRIMARY KEY, source_id INTEGER, word TEXT,
                                 start_time REAL, end_time REAL, source_file TEXT);
        INSERT INTO sources VALUES (1, 'v', 'downloads/v.mp4', '{slug} video', '');
        INSERT INTO sources VALUES (2, 'x', '{secret}', 'escape', '');
        INSERT INTO word_clips VALUES (1, 1, '{word}', 1.0, 1.5, 'downloads/v.mp4');
        INSERT INTO word_clips VALUES (2, 2, 'stolen', 1.0, 1.5, '{secret}');
        INSERT INTO word_clips VALUES (3, 2, 'climbed', 1.0, 1.5, '../../secret.json');
    """)
    conn.commit()
    conn.close()

# ── a stored path stays inside its corpus ──────────────────────────────────
base = os.path.join(data, "corpora", "a")
check("a relative path is resolved into the corpus",
      db.inside(base, "downloads/v.mp4"), os.path.realpath(os.path.join(base, "downloads", "v.mp4")))
for label, bad in (("an absolute path", secret), ("a climb out", "../../secret.json"),
                   ("a Windows-style climb", "downloads\\..\\..\\..\\secret.json")):
    try:
        db.inside(base, bad)
        check(f"{label} is refused", "accepted", "refused")
    except ValueError:
        check(f"{label} is refused", "refused", "refused")
try:
    db.resolve_path(secret)
    check("resolve_path refuses an outside path", "accepted", "refused")
except ValueError:
    check("resolve_path refuses an outside path", "refused", "refused")

g._ensure_cache()
check("the pool leaves out clips pointing outside", sorted(g._clips_by_word_cache), ["alpha"])
check("and the other voice's pool too",
      sorted(g._pool(g._corpus_for("b"))["clips_by_word"]), ["beta"])

# ── the cache is dropped, not pulled out from under a sentence ──────────────
installed = g._clips_by_word_cache
g.invalidate_cache()
check("globals still hold a pool after invalidating", g._clips_by_word_cache is installed, True)
g._ensure_cache()
check("and the next use builds a fresh one", g._clips_by_word_cache is not installed, True)
check("with the same words", sorted(g._clips_by_word_cache), ["alpha"])

# A pool built while it was being invalidated is used, not kept.
real_connect = sqlite3.connect
def racing_connect(*a, **k):
    g._epoch += 1                       # an edit lands mid-read
    return real_connect(*a, **k)
sqlite3.connect = racing_connect
g._POOLS.clear()
pool = g._pool(g._corpus_for("b"))
sqlite3.connect = real_connect
check("a pool read during an edit is not cached", "b" in g._POOLS, False)
check("but is still returned whole", sorted(pool["clips_by_word"]), ["beta"])

# ── a menu asking about another voice does not install it ──────────────────
g._ensure_cache()
before = g._clips_by_word_cache
got = g.clip_choices("beta", voice="b")
check("another voice's recordings", [c["source"] for c in got], ["b video"])
check("without swapping the voice in use", g._clips_by_word_cache is before, True)

check("an effect named twice is applied once", g.parse_fx("shake,shake,spin,shake")["glitch"],
      ["shake", "spin"])

# ── the queue says when it is full ─────────────────────────────────────────
jobs._ensure_worker = lambda: None      # nothing runs; they only wait
for _ in range(jobs.MAX_QUEUED):
    jobs.submit("alpha")
try:
    jobs.submit("alpha")
    check("one past the limit is refused", "queued", "refused")
except jobs.QueueFull:
    check("one past the limit is refused", "refused", "refused")

# ── HTTP: limits, and writes from another site ─────────────────────────────
try:
    from fastapi.testclient import TestClient
except Exception as exc:                                    # noqa: BLE001
    print(f"SKIP http checks: {exc}")
    TestClient = None

if TestClient:
    os.chdir(data)                      # main.py makes output/ and downloads/ here
    os.makedirs("frontend", exist_ok=True)
    import main  # noqa: E402
    web = TestClient(main.app)
    r = web.post("/api/generate", json={"text": "a" * 5000})
    check("too long a sentence is refused", (r.status_code, "characters" in r.json()["detail"]["message"]),
          (400, True))
    r = web.post("/api/generate", json={"text": "alpha"})
    check("a full queue answers 503", r.status_code, 503)
    r = web.post("/api/rate", json={"word": "alpha", "clips": list(range(300))})
    check("a vote on 300 clips is refused", r.status_code, 422)
    r = web.delete("/api/edit/recipe/x", headers={"sec-fetch-site": "cross-site"})
    check("a write from another site is refused", r.status_code, 403)
    r = web.post("/api/reload", headers={"sec-fetch-site": "same-origin"})
    check("the page's own write goes through", r.status_code, 200)
    r = web.get("/api/stats", headers={"sec-fetch-site": "cross-site"})
    check("a read from another site still works", r.status_code, 200)
    r = web.get("/api/edit/source/2/video")
    check("a source outside the corpus is not served", r.status_code, 404)
    check("nor is where it points said", secret in r.text, False)
    r = web.get("/api/edit/source/1/envelope?start=nan&end=5")
    check("a NaN span is refused, not handed to ffmpeg", r.status_code, 400)
    r = web.post("/api/edit/splice/x/preview", json={"groups": [{"phones": ["B"]}] * 33})
    check("a splice plan of 33 pieces is refused", r.status_code, 422)
    main._MAX_BODY = 1000               # small, rather than sending 4 GB
    r = web.post("/api/corpus/import", files={"bundle": ("b.tar", b"x" * 5000)}, data={"name": "n"})
    check("a body over the limit is refused before it is read", r.status_code, 413)
    # Sent chunked, with no length to refuse up front: counted as it arrives.
    # FastAPI reports a body it could not finish reading as a parse error.
    form = (b'--zz\r\nContent-Disposition: form-data; name="name"\r\n\r\nn\r\n'
            b'--zz\r\nContent-Disposition: form-data; name="bundle"; filename="b.tar"\r\n'
            b'Content-Type: application/octet-stream\r\n\r\n' + b"x" * 5000 + b"\r\n--zz--\r\n")

    def chunked():
        return web.post("/api/corpus/import", content=(form[i:i + 500] for i in range(0, len(form), 500)),
                        headers={"content-type": "multipart/form-data; boundary=zz"}).text
    check("a chunked body over the limit is cut off", "parsing the body" in chunked(), True)
    main._MAX_BODY = 5 << 30
    check("and under it, the same body is read through", "could not read that bundle" in chunked(), True)
    # A bundle forced in over a corpus already in use: every cache made from
    # the old file, and the note that it was set up, has to go.
    from scripts import corpus as corpus_pack
    bundle = corpus_pack.pack_bundle("b", os.path.join(data, "b-bundle.tar.gz"))
    db.init_db()
    g._pool(g._corpus_for("b"))
    with open(bundle, "rb") as fh:
        r = web.post("/api/corpus/import", files={"bundle": ("b.tar.gz", fh.read())},
                     data={"name": "b", "force": "true"})
    check("a forced import is accepted", r.status_code, 200)
    check("and nothing cached from before it survives", (len(db._ready), "b" in g._POOLS), (0, False))
    check("no CORS header offered", "access-control-allow-origin" in
          web.options("/api/stats", headers={"origin": "https://evil.example",
                                             "access-control-request-method": "DELETE"}).headers, False)

print()
print(f"{len(failures)} failure(s)" if failures else "all ok")
sys.exit(1 if failures else 0)
