#!/bin/sh
# Run inside a disposable application-image container. Mount this directory at
# /bench/benchmarks/pyav and pass benchmark.py arguments after this script.
set -eu
python -m pip install --quiet --only-binary=:all: --target /tmp/pyav-benchmark-deps \
  -r /bench/benchmarks/pyav/requirements.txt
PYTHONPATH=/tmp/pyav-benchmark-deps:/app exec python /bench/benchmarks/pyav/benchmark.py "$@"
