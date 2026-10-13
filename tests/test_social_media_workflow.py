import io
import unittest

from PIL import Image

from social_media_workflow import (
    convert_to_tiktok_format,
    pixelate_image,
    validate_public_media_url,
)


class SocialMediaWorkflowTests(unittest.TestCase):
    def test_pixelate_image_preserves_dimensions_as_png(self):
        image = Image.new("RGB", (64, 48), "#cc5533")
        source = io.BytesIO()
        image.save(source, format="PNG")

        result = Image.open(io.BytesIO(pixelate_image(source.getvalue(), pixel_size=8)))

        self.assertEqual(result.size, (64, 48))
        self.assertEqual(result.format, "PNG")

    def test_validate_public_media_url_rejects_social_post_links(self):
        with self.assertRaisesRegex(ValueError, "direct media file URL"):
            validate_public_media_url("https://www.tiktok.com/@creator/video/123")

    def test_validate_public_media_url_accepts_public_https_media(self):
        self.assertEqual(
            validate_public_media_url("https://cdn.example.com/video.mp4"),
            "https://cdn.example.com/video.mp4",
        )

    def test_convert_to_tiktok_format_rejects_out_of_range_zoom_before_ffmpeg(self):
        with self.assertRaisesRegex(ValueError, "zoom must be between"):
            convert_to_tiktok_format(b"video", zoom=2.5)


if __name__ == "__main__":
    unittest.main()