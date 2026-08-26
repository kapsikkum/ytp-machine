#!/bin/sh
# Bring the data directory up, then hand over to whatever was asked for.
#
# The corpus -- the database and the source videos it points at -- is not in
# the image. On a fresh volume there is nothing to serve, so this seeds it from
# a bundle if one is reachable, and otherwise says plainly what is missing
# rather than starting an app that would answer every request with "no clips".
set -eu

DATA="${MRS_DATA_DIR:-/app/data}"
DB="${MRS_DB_PATH:-$DATA/michael_rosen.db}"
PORT="${PORT:-8765}"

mkdir -p "$DATA/output" "$DATA/corpora"

# Corpora live one per directory under $DATA/corpora. An older install has its
# database loose in $DATA with downloads/ beside it; that layout still works
# and shows up as the corpus named "default", so nothing is moved and nothing
# breaks. New installs land in the tidy layout from the start.
SEED_DIR="$DATA/corpora/${CORPUS_NAME:-michael-rosen}"

seed() {
  # A loose corpus directory mounted at /corpus wins: that is what a clone of
  # this repo has, since the database and the videos are committed and a single
  # packed bundle is not (103 MB, and GitHub refuses anything over 100).
  #
  # corpus.db is the name every corpus uses now; michael_rosen.db is accepted
  # so a volume seeded from an older checkout still works.
  SRC_DB=""
  if [ -f /corpus/corpus.db ]; then
    SRC_DB=/corpus/corpus.db
  elif [ -f /corpus/michael_rosen.db ]; then
    SRC_DB=/corpus/michael_rosen.db
  fi

  if [ -n "$SRC_DB" ]; then
    echo "seeding corpus into $SEED_DIR from the files mounted at /corpus"
    mkdir -p "$SEED_DIR/downloads"
    cp "$SRC_DB" "$SEED_DIR/corpus.db"
    # Written as `if` rather than `[ -d x ] && cp ...` on purpose: under set -e
    # a bare test-and-command list that fails its test returns non-zero as a
    # statement, and the script exits. A missing transcripts directory would
    # have killed the container instead of being skipped.
    if [ -d /corpus/downloads ]; then
      cp -r /corpus/downloads/. "$SEED_DIR/downloads/"
    fi
    if [ -d /corpus/transcripts ]; then
      mkdir -p "$SEED_DIR/transcripts"
      cp -r /corpus/transcripts/. "$SEED_DIR/transcripts/"
    fi
    echo "  $(ls "$SEED_DIR/downloads" | wc -l) videos"
    return 0
  fi

  # A packed bundle, which is how a corpus travels to a server.
  for bundle in /corpus/*.tar.zst /corpus/*.tar.gz; do
    [ -e "$bundle" ] || continue
    echo "seeding corpus into $SEED_DIR from $bundle"
    mkdir -p "$SEED_DIR"
    python /app/scripts/corpus.py unpack "$bundle" --into "$SEED_DIR"
    return 0
  done

  if [ -n "${CORPUS_URL:-}" ]; then
    echo "seeding corpus into $SEED_DIR from \$CORPUS_URL"
    mkdir -p "$SEED_DIR"
    python /app/scripts/corpus.py unpack "$CORPUS_URL" --into "$SEED_DIR"
    return 0
  fi
  return 1
}

# Extra source packs, installed each start so dropping a bundle in the packs
# directory is all it takes to add a voice. Installing is skipped when the
# corpus already exists, so this is cheap on every restart but the first.
install_packs() {
  [ -d /packs ] || return 0
  for bundle in /packs/*.tar.zst /packs/*.tar.gz; do
    [ -e "$bundle" ] || continue
    name=$(basename "$bundle"); name=${name%%.tar.*}
    if [ -d "$DATA/corpora/$name" ]; then
      continue
    fi
    echo "installing source pack: $name"
    python /app/scripts/corpus.py install "$bundle" --name "$name" ||       echo "  failed, skipping $name"
  done
}

# Anything already installed under corpora/ counts, as does a legacy database.
have_corpus() {
  [ -f "$DB" ] && return 0
  for d in "$DATA"/corpora/*/; do
    [ -e "$d" ] || continue
    for f in "$d"*.db; do
      [ -e "$f" ] && return 0
    done
  done
  return 1
}

case "${1:-serve}" in
  serve)
    # Fold any legacy loose install into corpora/<name>/ before looking for a
    # corpus, so a volume created by an older image comes up in the same shape
    # as a fresh one. No-op once done, and a no-op on a volume that never had
    # the old layout, so it costs nothing to run every start.
    python /app/scripts/corpus.py migrate --apply --quiet \
      --name "${CORPUS_NAME:-michael-rosen}" || echo "  migration skipped"

    # Only serving needs a corpus. Requiring one to run anything at all meant
    # `podman run ... python somescript.py` died before it started, which is
    # exactly the situation where you are trying to build or repair a corpus.
    if ! have_corpus; then
      if ! seed; then
        echo "----------------------------------------------------------------"
        echo "No corpus found and none to seed from."
        echo
        echo "  $DB does not exist, no corpus is installed under $DATA/corpora,"
        echo "  no bundle is mounted at /corpus, and CORPUS_URL is unset."
        echo
        echo "Build one from a working copy with"
        echo "    python scripts/corpus.py pack"
        echo "then mount it read-only at /corpus, or point CORPUS_URL at it."
        echo "----------------------------------------------------------------"
        exit 1
      fi
    fi
    install_packs
    exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
    ;;
  corpus)
    shift
    exec python /app/scripts/corpus.py "$@"
    ;;
  *)
    # Anything else runs as given, so the container doubles as the place to run
    # the ingest scripts against the same corpus it serves.
    exec "$@"
    ;;
esac
