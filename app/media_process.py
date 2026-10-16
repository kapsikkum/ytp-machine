"""Run media commands with bounded waits and cooperative cancellation."""

from __future__ import annotations

import subprocess
import time
import threading
from contextlib import contextmanager


def stop(proc: subprocess.Popen) -> None:
    """Kill and reap a child before closing pipes or removing its files."""
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)
    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except OSError:
            pass


def run(cmd, *, check_cancel=None, timeout=300, **kwargs):
    """subprocess.run(), checking a cancellation callback while it runs."""
    if check_cancel is None:
        return subprocess.run(cmd, timeout=timeout, **kwargs)
    check_cancel()
    capture = kwargs.pop("capture_output", False)
    if capture:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc = subprocess.Popen(cmd, **kwargs)
    deadline = time.monotonic() + timeout
    try:
        while True:
            check_cancel()
            left = deadline - time.monotonic()
            if left <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout)
            try:
                stdout, stderr = proc.communicate(timeout=min(0.2, left))
                return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                pass
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()
        raise


def wait(proc: subprocess.Popen, *, check_cancel=None, timeout=300) -> int:
    """Wait for a piped encoder's final flush, without losing cancellation."""
    deadline = time.monotonic() + timeout
    while True:
        if check_cancel:
            check_cancel()
        left = deadline - time.monotonic()
        if left <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        try:
            return proc.wait(timeout=min(0.2, left))
        except subprocess.TimeoutExpired:
            pass


@contextmanager
def cancellation_watch(proc: subprocess.Popen, check_cancel=None):
    """Unblock a raw-video pipe write when its consumer stalls on cancellation."""
    if check_cancel is None:
        yield
        return
    ended = threading.Event()
    errors = []

    def watch():
        while not ended.wait(0.2):
            try:
                check_cancel()
            except BaseException as exc:
                errors.append(exc)
                try:
                    if proc.poll() is None:
                        proc.kill()
                except OSError:
                    pass
                return

    worker = threading.Thread(target=watch, name="ytp-encoder-cancel", daemon=True)
    worker.start()
    try:
        yield
    finally:
        ended.set()
        worker.join(timeout=1)
        if errors:
            raise errors[0]
