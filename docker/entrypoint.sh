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

mkdir -p "$DATA/downloads" "$DATA/output"

seed() {
  # A loose corpus mounted at /corpus wins: that is what a clone of this repo
  # has, since the database and the videos are committed and a single packed
  # bundle is not (103 MB, and GitHub refuses anything over 100).
  if [ -f /corpus/michael_rosen.db ]; then
    echo "seeding corpus from the files mounted at /corpus"
    cp /corpus/michael_rosen.db "$DATA/michael_rosen.db"
    # Written as `if` rather than `[ -d x ] && cp ...` on purpose: under set -e
    # a bare test-and-command list that fails its test returns non-zero as a
    # statement, and the script exits. A missing transcripts directory would
    # have killed the container instead of being skipped.
    if [ -d /corpus/downloads ]; then
      cp -r /corpus/downloads/. "$DATA/downloads/"
    fi
    if [ -d /corpus/transcripts ]; then
      mkdir -p "$DATA/transcripts"
      cp -r /corpus/transcripts/. "$DATA/transcripts/"
    fi
    echo "  $(ls "$DATA/downloads" | wc -l) videos"
    return 0
  fi

  # A packed bundle, which is how a corpus travels to a server.
  for bundle in /corpus/*.tar.zst /corpus/*.tar.gz; do
    [ -e "$bundle" ] || continue
    echo "seeding corpus from $bundle"
    python /app/scripts/corpus.py unpack "$bundle" --into "$DATA"
    return 0
  done

  if [ -n "${CORPUS_URL:-}" ]; then
    echo "seeding corpus from \$CORPUS_URL"
    python /app/scripts/corpus.py unpack "$CORPUS_URL" --into "$DATA"
    return 0
  fi
  return 1
}

if [ ! -f "$DB" ]; then
  if ! seed; then
    echo "----------------------------------------------------------------"
    echo "No corpus found and none to seed from."
    echo
    echo "  $DB does not exist, no bundle is mounted at /corpus, and"
    echo "  CORPUS_URL is unset."
    echo
    echo "Build one from a working copy with"
    echo "    python scripts/corpus.py pack"
    echo "then mount it read-only at /corpus, or point CORPUS_URL at it."
    echo "----------------------------------------------------------------"
    exit 1
  fi
fi

case "${1:-serve}" in
  serve)
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
