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
  # A mounted bundle wins over a URL: it needs no network and is what a host
  # with a copied-in corpus will have.
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
