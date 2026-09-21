"""Run DiskAudio integration and existing loader regressions on CPU."""
import os
from pathlib import Path
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root.parents[1])]
sys.argv = ['vhs-audio-tests', '--cpu', '--disable-xformers']
import comfy.options
comfy.options.enable_args_parsing()
import comfy.model_management
sys.argv = ['vhs-audio-tests']
suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(root / 'tests'), pattern=pattern)
                           for pattern in ('test_disk_audio.py', 'test_load_video_diskimage.py'))
raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
