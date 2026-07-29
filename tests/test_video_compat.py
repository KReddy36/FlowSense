"""TEST-ONLY: verifies browser-compatible H.264 conversion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from flowsense.video_compat import (
    browser_video_bytes,
    convert_to_browser_mp4,
    is_h264_mp4,
)


class BrowserVideoTests(unittest.TestCase):
    def test_mp4v_video_is_converted_to_h264(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "browser.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (96, 72),
            )
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(3):
                    writer.write(
                        np.zeros((72, 96, 3), dtype=np.uint8)
                    )
            finally:
                writer.release()

            converted = convert_to_browser_mp4(source, output)

            self.assertEqual(converted, output)
            self.assertTrue(is_h264_mp4(output))
            capture = cv2.VideoCapture(str(output))
            try:
                success, _ = capture.read()
            finally:
                capture.release()
            self.assertTrue(success)

    def test_browser_bytes_temporarily_convert_mp4v_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(3):
                    writer.write(
                        np.zeros((48, 64, 3), dtype=np.uint8)
                    )
            finally:
                writer.release()

            preview = browser_video_bytes(source)

            self.assertIn(b"ftyp", preview[:64])
            self.assertTrue(b"avc1" in preview or b"avc3" in preview)


if __name__ == "__main__":
    unittest.main()
