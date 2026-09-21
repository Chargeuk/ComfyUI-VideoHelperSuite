"""Optional native integration with the VTS DiskAudio storage implementation."""
import importlib
import functools
import sys
from pathlib import Path

import folder_paths


def _backend():
    for root in folder_paths.get_folder_paths('custom_nodes'):
        utils = Path(root) / 'ComfyUI-vts-nodes' / 'py' / 'vtsUtils'
        if (utils / 'vts_disk_audio.py').is_file():
            if str(utils) not in sys.path:
                sys.path.append(str(utils))
            return importlib.import_module('vts_disk_audio')
    return None


def disk_audio_from_file(path, start=0, duration=0, allow_missing=False):
    backend = _backend()
    if backend is None:
        raise RuntimeError('DiskAudio requires ComfyUI-vts-nodes with DiskAudio support installed')
    try:
        return backend.DiskAudio.from_file(path, start, duration)
    except backend.NoAudioStreamError:
        if allow_missing:
            return None
        raise


def _load_file(bound, controls):
    from .utils import strip_path, validate_path, is_url
    from .nodes import try_download_video
    arguments = bound.arguments
    extra = arguments.get('kwargs', {})
    if 'audio_file' in arguments:
        path = strip_path(arguments['audio_file'])
    else:
        path = folder_paths.get_annotated_filepath(strip_path(extra['audio']))
    if not path or validate_path(path) is not True:
        raise ValueError('Invalid audio file path')
    if is_url(path):
        path = try_download_video(path) or path
    audio = disk_audio_from_file(path, arguments.get('seek_seconds', arguments.get('start_time', 0)),
                                arguments.get('duration', 0))
    return audio, audio['waveform'].shape[-1] / audio['sample_rate']


def install_disk_audio(mapping):
    if _backend() is None:
        for cls in set(mapping.values()):
            function = getattr(cls, 'FUNCTION', None)
            if function:
                setattr(cls, function, _without_backend(getattr(cls, function)))
        return
    integrate = importlib.import_module('vts_audio_nodes').disk_audio_node
    for name, cls in mapping.items():
        if cls.__name__ in ('LoadAudio', 'LoadAudioUpload'):
            cls.DISK_AUDIO_EXECUTE = staticmethod(_load_file)
        integrate(cls, prefix=name, keep_disk_inputs=('audio',) if cls.__name__ == 'VideoCombine' else ())


def _without_backend(function):
    @functools.wraps(function)
    def execute(*args, **kwargs):
        if kwargs.get('audio_return_type', 'Tensor') != 'Tensor':
            raise RuntimeError('DiskAudio requires ComfyUI-vts-nodes with DiskAudio support installed')
        return function(*args, **{k: v for k, v in kwargs.items() if k not in (
            'audio_return_type', 'audio_format', 'audio_bitrate_kbps', 'audio_prefix',
            'audio_output_dir', 'audio_device_policy', 'audio_device')})
    return execute
