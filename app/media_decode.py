"""Optional in-process media decoding with the CLI as a compatibility path.

MRS_AUDIO_DECODER=pyav enables the audio backend; the default is ffmpeg.
Cancellation exceptions deliberately bypass compatibility fallback.
"""
from __future__ import annotations

from fractions import Fraction
import gc
import logging
import os
import time

import numpy as np

log = logging.getLogger(__name__)


class DecodeError(RuntimeError):
    """A backend could not decode this media; the CLI may still support it."""


class _Cancelled(BaseException):
    def __init__(self, original):
        self.original = original


def audio(path, start, duration, sr, pcm, check_cancel, fallback):
    def native(check):
        try:
            return _audio_pyav(path, start, duration, sr, pcm, check)
        finally:
            # Short instrument graphs are young. A full collection would walk
            # the entire pronunciation dictionary on every keyboard audition.
            gc.collect(1 if duration is not None and duration <= 3 else 2)
    return _optional("MRS_AUDIO_DECODER", check_cancel, fallback, native)


def video(path, start, duration, w, h, fit, pixel_aspect, fps, check_cancel, fallback):
    def native(check):
        try:
            return _video_pyav(path, start, duration, w, h, fit, pixel_aspect, fps, check)
        finally:
            # Filter contexts have Python cycles owning native worker pools.
            # Release those pools even if cancellation exits the frame loop.
            gc.collect(1 if duration <= 3 else 2)
    return _optional("MRS_VIDEO_DECODER", check_cancel, fallback, native)


def _optional(setting, check_cancel, fallback, native):
    backend = os.environ.get(setting, "ffmpeg").lower()
    if backend not in {"ffmpeg", "pyav"}:
        raise ValueError(f"{setting} must be ffmpeg or pyav")
    if backend == "ffmpeg":
        return fallback()
    deadline = time.monotonic() + 30

    def check():
        if check_cancel:
            try:
                check_cancel()
            except BaseException as exc:
                raise _Cancelled(exc) from None
        if time.monotonic() >= deadline:
            raise TimeoutError("Media decode exceeded 30 seconds")

    try:
        check()
        try:
            result = native(check)
        except (ImportError, DecodeError) as exc:
            check()
            log.debug("%s compatibility fallback: %s", setting, exc)
            return fallback()
        check()
        return result
    except _Cancelled as exc:
        raise exc.original


def _audio_pyav(path, start, duration, sr, pcm, check):
    import av

    if pcm not in {"f32le", "s16le"}:
        raise DecodeError("Unsupported PCM format")
    first = Fraction(f"{max(0, start or 0):.4f}")
    if duration is not None and duration < 0:
        raise DecodeError("Negative duration")
    last = first + Fraction(f"{duration:.4f}") if duration is not None else None
    if last == first:
        return np.empty(0, dtype=np.float32)
    pieces = []
    graph = None

    def push(frame):
        nonlocal graph
        if graph is None:
            if frame is None:
                return
            # AudioResampler's internal graph uses the host's default worker
            # count. Build the same abuffer/aformat chain with one worker so
            # concurrent recommendations respect a container's CPU/PID budget.
            graph = av.filter.Graph()
            graph.threads = 1
            source = graph.add("abuffer", sample_rate=str(frame.sample_rate),
                sample_fmt=frame.format.name, channel_layout=frame.layout.name,
                time_base=str(frame.time_base))
            target = graph.add("aformat", sample_rates=str(sr),
                sample_fmts="s16" if pcm == "s16le" else "flt", channel_layouts="mono")
            sink = graph.add("abuffersink")
            source.link_to(target)
            target.link_to(sink)
            graph.configure()
        graph.push(frame)
        while True:
            check()
            try:
                pieces.append(graph.pull().to_ndarray().reshape(-1).copy())
            except (av.error.BlockingIOError, av.error.EOFError):
                break

    try:
        with av.open(str(path), timeout=(5, 5)) as container:
            if not container.streams.audio:
                raise DecodeError("No audio stream")
            stream = container.streams.audio[0]
            stream.codec_context.thread_count = 1
            # Container-default seeking was verified for video recordings and
            # PCM. AAC-only containers use a different CLI priming/seek policy.
            if not container.streams.video and not stream.codec_context.name.startswith("pcm_"):
                raise DecodeError("Compressed audio-only timestamp policy")
            if stream.codec_context.name == "aac":
                # Keep unbounded reads and requests crossing encoded padding
                # on the CLI until its end-of-stream policy is reproduced.
                if last is None or stream.duration is None or last >= stream.duration * stream.time_base:
                    raise DecodeError("AAC end-of-stream timestamp policy")
            # Non-zero stream origins and discontinuous timestamps need the
            # CLI's timestamp policy. Do not silently change the clip window.
            if container.start_time not in (None, 0):
                raise DecodeError("Non-zero container start")
            if first:
                container.seek(round(first * av.time_base), backward=True)
            clock = 0
            for frame in container.decode(stream):
                check()
                if frame.pts is None or frame.time_base is None:
                    raise DecodeError("Audio has no timestamp")
                origin = frame.pts * frame.time_base
                left = max(0, round((first - origin) * frame.sample_rate))
                right = (min(frame.samples, round((last - origin) * frame.sample_rate))
                         if last is not None else frame.samples)
                if right > left:
                    data = frame.to_ndarray()
                    if frame.format.is_planar:
                        data = data[:, left:right]
                    else:
                        channels = len(frame.layout.channels)
                        data = data[:, left * channels:right * channels]
                    cropped = av.AudioFrame.from_ndarray(np.ascontiguousarray(data),
                        format=frame.format.name, layout=frame.layout.name)
                    cropped.sample_rate = frame.sample_rate
                    cropped.time_base = Fraction(1, frame.sample_rate)
                    cropped.pts = clock
                    clock += right - left
                    push(cropped)
                if last is not None and origin + Fraction(frame.samples, frame.sample_rate) >= last:
                    break
            push(None)
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise DecodeError(str(exc)) from exc
    data = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
    return data.astype(np.float32) / 32768.0 if pcm == "s16le" else data


def _video_pyav(path, start, duration, w, h, fit, pixel_aspect, fps, check):
    import av

    first = Fraction(f"{start:.3f}")
    last = first + Fraction(f"{duration + .1:.3f}")
    if (first * fps).denominator != 1:
        # The CLI's CFR output synchronizer duplicates the initial frame for
        # some fractional seeks. Keep that timing policy until reproduced.
        raise DecodeError("Fractional video seek requires CLI frame synchronization")
    arrays = []
    graph = None
    try:
        with av.open(str(path), timeout=(5, 5)) as container:
            if not container.streams.video or container.start_time not in (None, 0):
                raise DecodeError("Unsupported video origin or missing stream")
            stream = container.streams.video[0]
            if stream.base_rate and stream.average_rate and stream.base_rate != stream.average_rate:
                raise DecodeError("Variable frame-rate synchronization requires CLI")
            if stream.metadata.get("rotate", "0") not in ("0", "0.0"):
                raise DecodeError("Video rotation requires CLI autorotation")
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = 2
            if first:
                container.seek(int(first / stream.time_base), stream=stream, backward=True)

            def drain():
                while True:
                    check()
                    try:
                        arrays.append(graph.pull().to_ndarray(format="rgb24"))
                    except (av.error.BlockingIOError, av.error.EOFError):
                        break

            for frame in container.decode(stream):
                check()
                if frame.pts is None or frame.time_base is None:
                    raise DecodeError("Video has no timestamp")
                if av.sidedata.sidedata.Type.DISPLAYMATRIX in frame.side_data:
                    raise DecodeError("Video display matrix requires CLI autorotation")
                timestamp = frame.pts * frame.time_base
                if timestamp < first:
                    continue
                if timestamp >= last:
                    break
                frame.pts -= round(first / frame.time_base)
                if graph is None:
                    graph = av.filter.Graph()
                    graph.threads = 2
                    sar = stream.sample_aspect_ratio or Fraction(1)
                    rate = stream.average_rate or Fraction(fps)
                    source = graph.add("buffer", args=(
                        f"video_size={frame.width}x{frame.height}:pix_fmt={frame.format.name}"
                        f":time_base={frame.time_base}:pixel_aspect={sar}:frame_rate={rate}"))
                    bound = "max" if fit == "cover" else "min"
                    width = f"max(1,{bound}({w},{h}*dar/{pixel_aspect:.12g}))"
                    height = f"max(1,{bound}({h},{w}*{pixel_aspect:.12g}/dar))"
                    nodes = [source, graph.add("format", pix_fmts="rgb24"),
                             graph.add("scale", w=width, h=height)]
                    if fit == "cover":
                        nodes.append(graph.add("crop", w=str(w), h=str(h), exact="1"))
                    else:
                        nodes.append(graph.add("pad", w=str(w), h=str(h),
                                               x="(ow-iw)/2", y="(oh-ih)/2", color="black"))
                    nodes.extend([graph.add("setsar", sar="1"), graph.add("fps", fps=str(fps)),
                                  graph.add("buffersink")])
                    for previous, following in zip(nodes, nodes[1:]):
                        previous.link_to(following)
                    graph.configure()
                graph.push(frame)
                drain()
            if graph is not None:
                graph.push(None)
                drain()
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise DecodeError(str(exc)) from exc
    # The CLI's output -t limits the post-fps stream in frame time units.
    # Flushing fps alone can emit an additional copy of the final frame.
    limit = max(0, int(float(last - first) * fps + .5))
    arrays = arrays[:limit]
    return np.stack(arrays) if arrays else np.zeros((1, h, w, 3), np.uint8)
