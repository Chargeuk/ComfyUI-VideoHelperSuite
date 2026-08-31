import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import server

if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = SimpleNamespace(
        prompt_queue=SimpleNamespace()
    )

from videohelpersuite import load_video_nodes as module


class FakeDiskImage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.shape = kwargs["image"].shape
        self.dtype = kwargs["image"].dtype
        self.ndim = kwargs["image"].ndim


class FakeMetaBatch:
    def __init__(self):
        self.inputs = {}


class LoadVideoDiskImageTests(unittest.TestCase):
    def setUp(self):
        self.save_calls = []

    def fake_save_images(self, **kwargs):
        self.save_calls.append(kwargs)
        return [
            f"{kwargs['prefix']}_{kwargs['start_sequence'] + index:06d}.{kwargs['format']}"
            for index in range(len(kwargs["image"]))
        ]

    @staticmethod
    def fake_resolve_output_identity(prefix, start_sequence):
        return prefix, start_sequence

    def support_patch(self):
        return mock.patch.object(
            module,
            "_get_vts_disk_image_support",
            return_value=(
                FakeDiskImage,
                self.fake_resolve_output_identity,
                self.fake_save_images,
            ),
        )

    def test_tensor_mode_preserves_original_result_without_loading_vts(self):
        images = torch.rand(2, 8, 8, 3)
        original = (images, 2, object(), {"loaded_frame_count": 2})
        node = module.LoadVideoUpload()

        with mock.patch.object(
            module.folder_paths,
            "get_annotated_filepath",
            return_value="/input/selected.mp4",
        ), mock.patch.object(
            module,
            "load_video",
            return_value=original,
        ) as load_mock, mock.patch.object(
            module,
            "_get_vts_disk_image_support",
        ) as support_mock:
            result = node.load_video(
                video="selected.mp4",
                force_rate=0,
                custom_width=0,
                custom_height=0,
                frame_load_cap=2,
                skip_first_frames=4,
                select_every_nth=3,
                vts_return_type="Tensor",
            )

        self.assertIs(result, original)
        support_mock.assert_not_called()
        load_kwargs = load_mock.call_args.kwargs
        self.assertEqual(load_kwargs["skip_first_frames"], 4)
        self.assertEqual(load_kwargs["select_every_nth"], 3)
        self.assertNotIn("vts_return_type", load_kwargs)

    def test_diskimage_saves_only_the_selected_result_frames(self):
        selected_frames = torch.rand(3, 8, 8, 3)
        original = (
            selected_frames,
            3,
            object(),
            {"source_frame_count": 90, "loaded_frame_count": 3},
        )
        node = module.LoadVideoUpload()

        with mock.patch.object(
            module.folder_paths,
            "get_annotated_filepath",
            return_value="/input/selected.mp4",
        ), mock.patch.object(
            module,
            "load_video",
            return_value=original,
        ), self.support_patch():
            result = node.load_video(
                video="selected.mp4",
                force_rate=0,
                custom_width=0,
                custom_height=0,
                frame_load_cap=3,
                skip_first_frames=10,
                select_every_nth=5,
                vts_return_type="DiskImage",
                vts_prefix="selected",
                vts_start_sequence=7,
                vts_output_dir="/output",
                vts_format="png",
                vts_num_workers=2,
                vts_compression_level=6,
                vts_quality=101,
            )

        self.assertIsInstance(result[0], FakeDiskImage)
        self.assertEqual(result[0].number_of_images, 3)
        self.assertEqual(result[0].start_sequence, 7)
        self.assertEqual(result[1:], original[1:])
        self.assertEqual(len(self.save_calls), 1)
        self.assertIs(self.save_calls[0]["image"], selected_frames)
        self.assertEqual(len(self.save_calls[0]["image"]), 3)
        self.assertIsNone(self.save_calls[0]["quality"])

    def test_diskimage_rejects_latent_output(self):
        with self.assertRaisesRegex(RuntimeError, "VAE is connected"):
            module._save_loaded_frames_to_disk(
                image={"samples": torch.rand(2, 4, 8, 8)},
                prefix="selected",
                start_sequence=0,
                output_dir="/output",
                format="png",
                num_workers=1,
                compression_level=6,
                quality=95,
            )

    def test_meta_batch_chunks_use_consecutive_sequence_numbers(self):
        meta_batch = FakeMetaBatch()
        first = torch.rand(2, 8, 8, 3)
        second = torch.rand(3, 8, 8, 3)

        with self.support_patch():
            first_disk_image = module._save_loaded_frames_to_disk(
                image=first,
                prefix="selected",
                start_sequence=10,
                output_dir="/output",
                format="png",
                num_workers=1,
                compression_level=6,
                quality=95,
                meta_batch=meta_batch,
                unique_id="load-node",
                reset_meta_batch_sequence=True,
            )
            second_disk_image = module._save_loaded_frames_to_disk(
                image=second,
                prefix="selected",
                start_sequence=10,
                output_dir="/output",
                format="png",
                num_workers=1,
                compression_level=6,
                quality=95,
                meta_batch=meta_batch,
                unique_id="load-node",
                reset_meta_batch_sequence=False,
            )

        self.assertEqual(first_disk_image.start_sequence, 10)
        self.assertEqual(second_disk_image.start_sequence, 12)
        self.assertEqual(
            [call["start_sequence"] for call in self.save_calls],
            [10, 12],
        )


if __name__ == "__main__":
    unittest.main()
