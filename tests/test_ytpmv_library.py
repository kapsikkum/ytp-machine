"""Real MIDI library edits, cache invalidation and queued-source protection."""
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mido

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import api, jobs
from app.ytpmv import render


class MidiLibraryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(render, 'MIDI_DIR', str(self.root)))
        self.stack.enter_context(patch.object(render, '_songs', {}))
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.extend([mido.Message('note_on', note=60, velocity=100),
                      mido.Message('note_off', note=60, time=480)])
        data = BytesIO()
        midi.save(file=data)
        self.mid = render.save_midi(data.getvalue(), 'Original.mid')

    def test_rename_updates_parsed_song_and_library_cache_without_touching_midi(self):
        source = self.root / f'{self.mid}.mid'
        before = source.read_bytes()
        first = render.load_song(self.mid)
        self.assertEqual(render.library()[0]['title'], 'Original')
        renamed = api.ytpmv_rename(self.mid, api.YtpmvRenameRequest(title='  New <title>  '))
        self.assertEqual(renamed['title'], 'New <title>')
        self.assertEqual(render.library()[0]['title'], 'New <title>')
        self.assertEqual(render.load_song(self.mid).title, 'New <title>')
        self.assertIsNot(render.load_song(self.mid), first)
        self.assertEqual(source.read_bytes(), before)

    def test_delete_removes_source_and_metadata_but_keeps_other_files(self):
        render.library()
        artifact = self.root / 'existing-render.mp4'
        artifact.write_bytes(b'render')
        self.assertTrue(api.ytpmv_delete(self.mid)['deleted'])
        self.assertEqual(render.library(), [])
        self.assertFalse(list(self.root.glob(f'{self.mid}.mid*')))
        self.assertEqual(artifact.read_bytes(), b'render')
        with self.assertRaises(render.YtpmvError):
            render.load_song(self.mid)

    def test_invalid_titles_and_ids_cannot_change_files(self):
        for title in ['', ' ', 'x' * 81, 'two\nlines', 'null\x00byte']:
            with self.subTest(title=title), self.assertRaises(render.YtpmvError):
                render.rename_midi(self.mid, title)
        for mid in ['../outside', self.mid + '\n', 'X' * 12, 'C:/outside']:
            with self.subTest(mid=mid), self.assertRaises(render.YtpmvError):
                render.delete_midi(mid)
        self.assertEqual(render.load_song(self.mid).title, 'Original')

    def test_queued_render_source_cannot_be_deleted(self):
        with patch.object(jobs, '_ensure_worker'):
            job = jobs.submit_ytpmv({'midi_id': self.mid}, corpus='fixture')
        self.addCleanup(lambda: jobs.cancel(job.id))
        with self.assertRaises(api.HTTPException) as raised:
            api.ytpmv_delete(self.mid)
        self.assertEqual(raised.exception.status_code, 409)
        self.assertTrue((self.root / f'{self.mid}.mid').is_file())
        jobs.cancel(job.id)
        self.assertTrue(api.ytpmv_delete(self.mid)['deleted'])


if __name__ == '__main__':
    unittest.main()
