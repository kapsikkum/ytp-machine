"""Compare real application decode paths with isolated PyAV prototypes.

No application monkeypatches, corpus writes, or production dependency changes.
Fixtures and JSON output belong in output/ or another scratch directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.ytpmv import pitch, samples


@dataclass(frozen=True)
class AudioCase:
    name: str
    source: Path
    start: float
    duration: float
    pcm: str = 'f32le'
    sr: int = 44100


@dataclass(frozen=True)
class VideoCase:
    name: str
    source: Path
    start: float
    duration: float
    width: int
    height: int
    fit: str = 'contain'
    pixel_aspect: float = 1


def ffmpeg_audio(case: AudioCase) -> np.ndarray:
    return pitch.decode_audio(str(case.source), case.start, case.duration, case.sr, case.pcm)


def pyav_audio(case: AudioCase) -> np.ndarray:
    """Open/seek/decode/resample each call, matching the application's PCM API."""
    start = float(f'{max(0, case.start):.4f}')
    duration = float(f'{case.duration:.4f}')
    first_time = Fraction(str(start))
    last_time = first_time + Fraction(str(duration))
    pieces = []
    with av.open(str(case.source)) as container:
        stream = container.streams.audio[0]
        # Accurate seeking happens before resampling, as in the FFmpeg CLI.
        # Cropping already-resampled audio shifts its phase on fractional
        # output-sample boundaries and changes the start/end filter response.
        if start:
            # Use the container's default seek stream, as the CLI does. An
            # audio-specific seek can start decoding at a different packet.
            container.seek(round(start * av.time_base), backward=True)
        resampler = av.AudioResampler(format='s16' if case.pcm == 's16le' else 'flt',
                                      layout='mono', rate=case.sr)
        source_clock = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Decoded audio has no timestamp')
            origin = frame.pts * frame.time_base
            left = max(0, round((first_time-origin) * frame.sample_rate))
            right = min(frame.samples, round((last_time-origin) * frame.sample_rate))
            if right > left:
                data = frame.to_ndarray()
                if frame.format.is_planar:
                    data = data[:,left:right]
                else:
                    channels = len(frame.layout.channels)
                    data = data[:,left*channels:right*channels]
                cropped = av.AudioFrame.from_ndarray(np.ascontiguousarray(data),
                    format=frame.format.name, layout=frame.layout.name)
                cropped.sample_rate = frame.sample_rate
                cropped.time_base = Fraction(1, frame.sample_rate)
                cropped.pts = source_clock
                source_clock += right-left
                for converted in resampler.resample(cropped):
                    pieces.append(converted.to_ndarray().reshape(-1).copy())
            if origin + Fraction(frame.samples,frame.sample_rate) >= last_time:
                break
        for converted in resampler.resample(None):
            pieces.append(converted.to_ndarray().reshape(-1).copy())
    dtype = np.int16 if case.pcm == 's16le' else np.float32
    data = np.concatenate(pieces) if pieces else np.empty(0, dtype=dtype)
    return data.astype(np.float32) / 32768 if case.pcm == 's16le' else data


def sample_for(case: VideoCase):
    return samples.Sample(case.name, 0, 1, None, str(case.source), case.start,
                          SimpleNamespace(duration=case.duration))


def ffmpeg_video(case: VideoCase) -> np.ndarray:
    return sample_for(case).frames(case.width, case.height, fit=case.fit, pixel_aspect=case.pixel_aspect)


def _pyav_video(case: VideoCase) -> np.ndarray:
    """Use the same format/scale/pad-or-crop/SAR/FPS filter sequence."""
    start = float(f'{case.start:.3f}')
    end = start + float(f'{case.duration+.1:.3f}')
    arrays = []
    graph = None
    with av.open(str(case.source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 2
        if start:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)

        def drain():
            while True:
                try:
                    arrays.append(graph.pull().to_ndarray(format='rgb24'))
                except (av.error.BlockingIOError, av.error.EOFError):
                    break

        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Video has no timestamp')
            timestamp = float(frame.pts * frame.time_base)
            if timestamp + 1e-8 < start:
                continue
            if timestamp >= end - 1e-8:
                break
            frame.pts -= round(Fraction(str(start)) / frame.time_base)
            if graph is None:
                graph = av.filter.Graph()
                graph.threads = 2
                sar = stream.sample_aspect_ratio or Fraction(1)
                rate = stream.average_rate or Fraction(samples.FPS)
                source = graph.add('buffer', args=(
                    f'video_size={frame.width}x{frame.height}:pix_fmt={frame.format.name}'
                    f':time_base={frame.time_base}:pixel_aspect={sar}:frame_rate={rate}'))
                bound = 'max' if case.fit == 'cover' else 'min'
                width = f'max(1,{bound}({case.width},{case.height}*dar/{case.pixel_aspect:.12g}))'
                height = f'max(1,{bound}({case.height},{case.width}*{case.pixel_aspect:.12g}/dar))'
                nodes = [source, graph.add('format', pix_fmts='rgb24'), graph.add('scale', w=width, h=height)]
                if case.fit == 'cover':
                    nodes.append(graph.add('crop', w=str(case.width), h=str(case.height), exact='1'))
                else:
                    nodes.append(graph.add('pad', w=str(case.width), h=str(case.height),
                                           x='(ow-iw)/2', y='(oh-ih)/2', color='black'))
                nodes += [graph.add('setsar', sar='1'), graph.add('fps', fps=str(samples.FPS)), graph.add('buffersink')]
                for previous, following in zip(nodes, nodes[1:]):
                    previous.link_to(following)
                graph.configure()
            graph.push(frame)
            drain()
        if graph is not None:
            graph.push(None)
            drain()
    return np.stack(arrays) if arrays else np.zeros((1,case.height,case.width,3),np.uint8)


def pyav_video(case: VideoCase) -> np.ndarray:
    try:
        return _pyav_video(case)
    finally:
        # Filter graphs own native worker pools and Python context cycles.
        # Release them before another decode; include this cost in timing.
        gc.collect()


def compare(a, b, kind):
    common = min(len(a), len(b))
    left, right = a[:common].astype(np.float64), b[:common].astype(np.float64)
    difference = np.abs(left-right)
    mean = float(difference.mean()) if difference.size else None
    maximum = float(difference.max()) if difference.size else None
    # This is a prototype acceptance gate, not a byte-identity assertion.
    # Record raw measurements so tighter thresholds can be evaluated later.
    equivalent = bool(difference.size and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all())
    if equivalent:
        equivalent &= (mean <= 1e-5 and maximum <= 2/32768) if kind == 'audio' else (mean <= 1 and maximum <= 8)
    return {'shape_ffmpeg':list(a.shape), 'shape_pyav':list(b.shape),
            'mean_absolute_error':mean, 'max_absolute_error':maximum,
            'reference_rms':float(np.sqrt(np.mean(left*left))) if left.size else None,
            'byte_identical':a.shape == b.shape and a.tobytes() == b.tobytes(),
            'equivalent':bool(equivalent), 'output_bytes_ffmpeg':a.nbytes, 'output_bytes_pyav':b.nbytes}


def measure(case, kind, repeats):
    functions = {'ffmpeg':ffmpeg_audio if kind == 'audio' else ffmpeg_video,
                 'pyav':pyav_audio if kind == 'audio' else pyav_video}
    results, first, timings = {}, {}, {key:[] for key in functions}
    for name, function in functions.items():
        if kind == 'video': samples.clear_cache()
        begin = time.perf_counter()
        results[name] = function(case)
        first[name] = (time.perf_counter()-begin)*1000
    correctness = compare(results['ffmpeg'], results['pyav'], kind)
    # Alternate order, using a fixed seed, to reduce systematic warmup bias.
    rng = random.Random(0)
    for _ in range(repeats):
        order = list(functions); rng.shuffle(order)
        for name in order:
            if kind == 'video': samples.clear_cache()  # outside the timed region
            begin = time.perf_counter(); functions[name](case)
            timings[name].append((time.perf_counter()-begin)*1000)
    median = {key:statistics.median(value) for key,value in timings.items()}
    result = {'kind':kind, 'name':case.name, 'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in asdict(case).items() if k!='source'},
              'first_call_ms':first, 'median_ms':median, 'samples_ms':timings,
              'decode_speedup':median['ffmpeg']/median['pyav'], 'correctness':correctness}
    if kind == 'video':
        samples.clear_cache(); ffmpeg_video(case)
        hit=[]
        for _ in range(repeats*10):
            begin=time.perf_counter(); ffmpeg_video(case); hit.append((time.perf_counter()-begin)*1000)
        result['existing_frame_cache_hit_median_ms']=statistics.median(hit)
    label='PASS' if correctness['equivalent'] else 'DIFF'
    print(f"{label} {kind} {case.name}: FFmpeg {median['ffmpeg']:.2f}ms, PyAV {median['pyav']:.2f}ms, "
          f"{result['decode_speedup']:.2f}x; MAE={correctness['mean_absolute_error']}, "
          f"shape={correctness['shape_ffmpeg']}/{correctness['shape_pyav']}",flush=True)
    return result


def fixtures(directory):
    directory.mkdir(parents=True,exist_ok=True)
    sources=[]
    for name,size,rate in [('landscape','480x270',25),('portrait','240x320',30)]:
        path=directory/(name+'.mp4')
        if not path.exists():
            subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i',
                f'testsrc2=size={size}:rate={rate}:duration=4','-f','lavfi','-i',
                'aevalsrc=0.2*sin(2*PI*440*t)|0.2*sin(2*PI*660*t):s=48000:d=4',
                '-c:v','libx264','-pix_fmt','yuv420p','-g',str(rate),'-bf','2',
                '-c:a','aac','-b:a','128k','-shortest',str(path)],check=True,capture_output=True)
        sources.append(path)
    wav=directory/'stereo.wav'
    if not wav.exists():
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(sources[0]),'-vn','-c:a','pcm_s16le',str(wav)],check=True,capture_output=True)
    return sources,wav


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,default=ROOT/'output/pyav-benchmark')
    parser.add_argument('--output',type=Path,default=ROOT/'output/pyav-benchmark/results.json')
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--source',type=Path,action='append',default=[],help='Optional real source video, read only')
    parser.add_argument('--case',action='append',default=[],help='Run only cases containing one of these strings')
    parser.add_argument('--baseline-commit',help='Identify the app image when running outside a Git checkout')
    parser.add_argument('--environment-label',default='local',help='Record resource limits or host role without identifying the machine')
    args=parser.parse_args()
    if not 1<=args.repeats<=100: parser.error('--repeats must be 1–100')
    videos,wav=fixtures(args.work_dir)
    cases=[]
    for path,name in [(videos[0],'aac'),(wav,'pcm'),*((p,f'real-{i+1}') for i,p in enumerate(args.source))]:
        for pcm in ['f32le','s16le']:
            windows = [(5.123,.08),(13.017,.35),(21.12,1.5)] if name.startswith('real-') else [(0,.08),(.123,.35),(1.12,1.5)]
            for start,duration in windows:
                cases.append((AudioCase(f'{name}-{pcm}-{start}-{duration}',path,start,duration,pcm),'audio'))
    for path in videos:
        for fit in ['contain','cover']:
            cases.append((VideoCase(f'{path.stem}-{fit}-square',path,0,.5,180,180,fit),'video'))
        cases.append((VideoCase(f'{path.stem}-seek-odd',path,.12,.5,181,91),'video'))
    cases.append((VideoCase('retro-pixel-aspect',videos[0],0,.5,128,112,'contain',1.5555555555555556),'video'))
    for i,path in enumerate(args.source):
        cases.append((VideoCase(f'real-{i+1}-sample',path,5.12,.5,240,135),'video'))
    if args.case: cases=[(case,kind) for case,kind in cases if any(text in case.name for text in args.case)]
    if not cases: parser.error('No cases matched')
    results=[]
    for case,kind in cases:
        try:
            results.append(measure(case,kind,args.repeats))
        except Exception as error:
            results.append({'kind':kind,'name':case.name,'error':f'{type(error).__name__}: {error}',
                            'correctness':{'equivalent':False,'byte_identical':False}})
            print(f'ERROR {kind} {case.name}: {error}',flush=True)
        finally:
            gc.collect()
    report={'created_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_commit':args.baseline_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'app_decode_sha256':{module.__name__:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in [pitch,samples]},
        'environment':{'label':args.environment_label,'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
                       'pyav':av.__version__,'pyav_libraries':av.library_versions,
                       'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]},
        'methodology':{'repeats':args.repeats,'pyav_video_threads':'AUTO with 2 decoder threads and 2 graph threads; graph-cycle collection included in each video call',
                       'timing':'In-process function wall time; interpreter imports and fixture creation excluded; OS filesystem cache uncontrolled.',
                       'cache':'Sample frame cache cleared outside timed misses; existing hit times reported separately.',
                       'scope':'Audio/video decoding only, not full render speed; correctness failures are not usable speedups.',
                       'memory':'Output array bytes only, not peak native or child-process RSS.'},
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*videos,wav,*args.source]},
        'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    passed=sum(r['correctness']['equivalent'] for r in results)
    print(f'{passed}/{len(results)} correctness gates passed; report: {args.output}',flush=True)
    return 0 if passed==len(results) else 2


if __name__=='__main__':
    raise SystemExit(main())
