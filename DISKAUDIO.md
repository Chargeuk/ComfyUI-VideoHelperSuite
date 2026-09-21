# DiskAudio integration

This fork supports DiskAudio when the matching ComfyUI-vts-nodes extension is installed. Existing workflows retain native AUDIO output by default.

Audio loaders, video loaders with AUDIO output, and legacy audio adapters expose `audio_return_type`, `audio_format`, `audio_bitrate_kbps`, `audio_output_dir`, and `audio_prefix` where applicable. FLAC is the default disk format; Exact, Opus and MP3 are also available. Every new control and audio socket has a tooltip; the bitrate tooltip includes speech and stereo music guidance.

FLAC stores 24-bit PCM losslessly after converting floating-point samples to PCM. Exact preserves tensor values exactly. For all storage choices, naming, bitrate and device behavior, see [the VTS DiskAudio guide](https://github.com/Chargeuk/ComfyUI-vts-nodes/blob/main/docs/DISKAUDIO.md).

Video/audio loaders can save disk output without retaining a full waveform. Video Combine sends single-clip codec-backed DiskAudio directly to FFmpeg. Final video audio encoding remains controlled by the video format preset, independently of DiskAudio storage. Exact audio must be materialized for this muxing path. VHS Unbatch accepts DiskAudio and returns a native batched waveform.

Start time and duration apply to the referenced audio segment. Disk metadata reports decoded sample counts, including codec delay/padding handling. The VTS manifest and all payload files must be retained together.

Without the VTS dependency, normal VHS behavior remains available. A request for DiskAudio gives an installation error instead of silently returning a tensor.

Run `tests/run_disk_audio_tests.py` in the ComfyUI Python environment for CPU integration tests, including a real FFmpeg mux that forbids full waveform materialization.
