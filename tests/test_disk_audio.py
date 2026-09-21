import asyncio
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import server

if not hasattr(server.PromptServer, 'instance'):
    server.PromptServer.instance = SimpleNamespace(prompt_queue=SimpleNamespace())

from videohelpersuite import nodes, disk_audio
from videohelpersuite.utils import LazyAudioMap


class DiskAudioIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = disk_audio._backend()
        if self.backend is None:
            self.skipTest('Install VTS DiskAudio for integration tests')
        disk_audio.install_disk_audio(nodes.NODE_CLASS_MAPPINGS)
        self.native = {'waveform': torch.zeros(1, 2, 4800), 'sample_rate': 48000}
        self.disk = self.backend.save_audio(self.native, output_dir=self.temp.name)

    def test_lazy_video_audio_does_not_cache_a_waveform(self):
        source = LazyAudioMap(self.disk.files[0], 0, .05)
        with mock.patch('videohelpersuite.utils.get_audio', side_effect=AssertionError('eager load')):
            output = self.backend.save_audio(source, output_dir=self.temp.name)
        self.assertEqual(output['waveform'].shape[-1], 2400)
        self.assertIsNone(source._dict)

    def test_silent_video_keeps_absent_audio(self):
        path = Path(self.temp.name) / 'silent.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=32x32:duration=0.1',
                        '-an', str(path)], check=True)
        source = LazyAudioMap(str(path), 0, 0)
        self.assertIsNone(self.backend.save_audio(source, output_dir=self.temp.name))

    def test_file_loader_disk_mode_does_not_materialize(self):
        with mock.patch.object(nodes, 'get_audio', side_effect=AssertionError('eager load')):
            audio, duration = nodes.LoadAudio().load_audio(self.disk.files[0], audio_return_type='DiskAudio',
                                                         audio_output_dir=self.temp.name)
        self.assertIsInstance(audio, self.backend.DiskAudio)
        self.assertEqual(duration, .1)

    def test_unbatch_accepts_disk_audio(self):
        result = asyncio.run(nodes.Unbatch.execute([self.disk, self.disk]))
        self.assertEqual(result[0]['waveform'].shape, (2, 2, 4800))

    def test_optional_dependency_keeps_native_calls(self):
        def original(value):
            return value
        wrapped = disk_audio._without_backend(original)
        self.assertEqual(wrapped(7, audio_return_type='Tensor', audio_format='FLAC'), 7)
        with self.assertRaisesRegex(RuntimeError, 'ComfyUI-vts-nodes'):
            wrapped(7, audio_return_type='DiskAudio')

    def test_video_combine_uses_file_not_waveform(self):
        # Real FFmpeg mux, with waveform materialization forbidden.
        with mock.patch.object(nodes.folder_paths, 'get_output_directory', return_value=self.temp.name), \
             mock.patch.object(self.backend.DiskAudio, 'materialize', side_effect=AssertionError('eager load')):
            result = nodes.VideoCombine().combine_video(images=torch.zeros(2, 64, 64, 3), frame_rate=20,
                loop_count=0, filename_prefix='disk-audio', format='video/h264-mp4', pingpong=False,
                save_output=True, audio=self.disk, unique_id='test', pix_fmt='yuv420p', crf=23,
                save_metadata=False, trim_to_audio=False)
        self.assertTrue(any(str(path).endswith('-audio.mp4') for path in result['result'][0][1]))


if __name__ == '__main__':
    unittest.main()
