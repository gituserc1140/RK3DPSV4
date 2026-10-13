"""Provider and local media helpers for the social animation workflow."""

import json
import math
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from PIL import Image


VIGGLE_API_URL = "https://apis.viggle.ai/v1"
BUFFER_API_URL = "https://api.buffer.com"
SOCIAL_MEDIA_PAGE_HOSTS = {
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtu.be",
    "youtube.com",
}


def pixelate_image(image_bytes, pixel_size=8):
    """Reduce image detail and enlarge with nearest-neighbor interpolation."""
    if not image_bytes:
        raise ValueError("Choose an image to convert to pixel art.")
    if not 2 <= int(pixel_size) <= 64:
        raise ValueError("Pixel size must be between 2 and 64.")

    import io

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        reduced_size = (
            max(1, image.width // int(pixel_size)),
            max(1, image.height // int(pixel_size)),
        )
        pixel_art = image.resize(reduced_size, Image.Resampling.BICUBIC).resize(
            image.size, Image.Resampling.NEAREST
        )
        output = io.BytesIO()
        pixel_art.save(output, format="PNG")
        return output.getvalue()


def animate_image(
    image_bytes,
    prompt,
    api_key,
    *,
    quality="low",
    duration_seconds=5,
    poll_interval=5,
    max_poll_attempts=60,
):
    """Generate an image-referenced MP4 with Viggle and return its bytes."""
    if quality not in {"low", "high"}:
        raise ValueError("Viggle quality must be either low or high.")
    if not 3 <= int(duration_seconds) <= 15:
        raise ValueError("Viggle video duration must be between 3 and 15 seconds.")
    if not prompt.strip():
        raise ValueError("Describe the video motion before generating.")
    if not image_bytes:
        raise ValueError("Provide an image to use as the animation reference.")

    created = _viggle_json_response(
        requests.post(
            f"{VIGGLE_API_URL}/videos",
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "prompt": prompt.strip(),
                "quality": quality,
                "duration_s": str(duration_seconds),
            },
            files={"reference_image": ("pixel-art.png", image_bytes, "image/png")},
            timeout=120,
        )
    )
    video_id = created.get("id")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("Viggle did not return a video ID.")

    status = None
    for attempt in range(max_poll_attempts):
        if attempt:
            time.sleep(poll_interval)
        video = _viggle_json_response(
            requests.get(
                f"{VIGGLE_API_URL}/videos/{video_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
        )
        status = video.get("status")
        if status in {"ready", "failed", "cancelled"}:
            break
    else:
        raise TimeoutError(f"Viggle video {video_id} was not ready in time.")

    if status != "ready":
        error = video.get("error")
        detail = error.get("message") if isinstance(error, dict) else None
        raise RuntimeError(detail or f"Viggle video generation ended with {status!r}.")

    video_url = video.get("video_url")
    if not isinstance(video_url, str) or not video_url.startswith("https://"):
        raise ValueError("Viggle returned no valid HTTPS video URL.")
    response = requests.get(video_url, timeout=180)
    response.raise_for_status()
    if not response.content:
        raise ValueError("Viggle returned an empty video file.")
    return response.content


def add_background_music(video_bytes, music_bytes, music_extension):
    """Mux a soundtrack into an MP4, trimming the audio at the video end."""
    if not video_bytes or not music_bytes:
        raise ValueError("Choose both a video and a soundtrack.")
    extension = str(music_extension).lower().lstrip(".")
    if extension not in {"mp3", "wav", "m4a", "aac", "ogg", "flac"}:
        raise ValueError("Use an MP3, WAV, M4A, AAC, OGG, or FLAC soundtrack.")

    import imageio_ffmpeg

    with tempfile.TemporaryDirectory(prefix="social-media-workflow-") as directory:
        video_path = Path(directory) / "video.mp4"
        music_path = Path(directory) / f"music.{extension}"
        output_path = Path(directory) / "with-music.mp4"
        video_path.write_bytes(video_bytes)
        music_path.write_bytes(music_bytes)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(music_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=180)
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not add the soundtrack: {details[-1000:]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Adding the soundtrack exceeded the 3-minute limit.") from exc
        return output_path.read_bytes()


def convert_to_tiktok_format(video_bytes, zoom=1.0):
    """Fit a video into a 1080x1920 frame over a sampled solid background."""
    if not video_bytes:
        raise ValueError("Choose a video to convert.")
    if not math.isfinite(zoom) or not 0.5 <= zoom <= 2.0:
        raise ValueError("Video zoom must be between 0.5 and 2.0.")

    import imageio_ffmpeg

    with tempfile.TemporaryDirectory(prefix="social-media-workflow-") as directory:
        video_path = Path(directory) / "video.mp4"
        output_path = Path(directory) / "portrait.mp4"
        video_path.write_bytes(video_bytes)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        sample_command = [
            ffmpeg,
            "-i",
            str(video_path),
            "-vf",
            "scale=1:1:flags=area,format=rgb24",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            sample = subprocess.run(
                sample_command, check=True, capture_output=True, timeout=30
            ).stdout
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not sample the video background: {details[-1000:]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Sampling the video background exceeded 30 seconds.") from exc
        if len(sample) != 3:
            raise ValueError("Could not read a video frame to sample its background color.")

        background_color = "".join(f"{channel:02x}" for channel in sample)
        zoom_text = f"{zoom:.2f}"
        video_filter = (
            "[0:v]null[foreground];"
            f"[foreground]scale=1080*{zoom_text}:1920*{zoom_text}:"
            "force_original_aspect_ratio=decrease[foreground];"
            f"color=c=0x{background_color}:s=1080x1920:r=30[background];"
            "[background][foreground]overlay=(W-w)/2:(H-h)/2:shortest=1,"
            "setsar=1,format=yuv420p[video]"
        )
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-map",
            "[video]",
            "-map",
            "0:a?",
            "-filter_complex",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not convert the video: {details[-1000:]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Video conversion exceeded the 10-minute limit.") from exc
        return output_path.read_bytes()


def get_organizations(api_key):
    data = _buffer_request(
        "query GetOrganizations { account { organizations { id name } } }", api_key
    )
    try:
        return data["account"]["organizations"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Buffer's response did not include organizations.") from exc


def get_channels(organization_id, api_key):
    query = (
        "query GetChannels($organizationId: OrganizationId!) { "
        "channels(input: { organizationId: $organizationId }) { id name service } }"
    )
    data = _buffer_request(query, api_key, {"organizationId": organization_id})
    try:
        return data["channels"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Buffer's response did not include channels.") from exc


def validate_public_media_url(url):
    """Require HTTPS and reject common social post/profile links as media."""
    try:
        parsed = urlsplit(url.strip())
        hostname = (parsed.hostname or "").lower().removeprefix("www.")
    except ValueError as exc:
        raise ValueError("Enter a valid public HTTPS media URL.") from exc
    if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
        raise ValueError("Media URLs must be public HTTPS file URLs.")
    if hostname in SOCIAL_MEDIA_PAGE_HOSTS or any(
        hostname.endswith(f".{domain}") for domain in SOCIAL_MEDIA_PAGE_HOSTS
    ):
        raise ValueError("Use a direct media file URL, not a social profile or post URL.")
    return parsed.geturl()


def create_post(
    *, channel_id, text, api_key, mode, due_at=None, media_url=None, media_kind=None
):
    """Queue a Buffer post or schedule it for a specific UTC time."""
    if mode not in {"addToQueue", "customScheduled"}:
        raise ValueError("Choose either the next queue slot or a custom scheduled time.")
    if mode == "customScheduled" and not due_at:
        raise ValueError("A scheduled post requires a date and time.")
    if media_url:
        media_url = validate_public_media_url(media_url)
        if media_kind not in {"image", "video"}:
            raise ValueError("Choose whether the attached media is an image or video.")

    fields = [
        f"text: {json.dumps(text)}",
        f"channelId: {json.dumps(channel_id)}",
        "schedulingType: automatic",
        f"mode: {mode}",
    ]
    if due_at:
        fields.append(f"dueAt: {json.dumps(due_at)}")
    if media_url:
        fields.append(
            f"assets: [{{ {media_kind}: {{ url: {json.dumps(media_url)} }} }}]"
        )
    query = (
        "mutation CreatePost { createPost(input: { "
        + " ".join(fields)
        + " }) { ... on PostActionSuccess { post { id text dueAt } } "
        "... on MutationError { message } } }"
    )
    result = _buffer_request(query, api_key).get("createPost")
    if not isinstance(result, dict):
        raise ValueError("Buffer returned no post result.")
    if result.get("message"):
        raise RuntimeError(f"Buffer could not create the post: {result['message']}")
    post = result.get("post")
    if not isinstance(post, dict):
        raise ValueError("Buffer's response did not include the created post.")
    return post


def _buffer_request(query, api_key, variables=None):
    response = requests.post(
        BUFFER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        messages = "; ".join(
            str(error.get("message", "Unknown GraphQL error"))
            for error in payload["errors"]
        )
        raise RuntimeError(f"Buffer API error: {messages}")
    if not isinstance(payload.get("data"), dict):
        raise ValueError("Buffer returned a response without GraphQL data.")
    return payload["data"]


def _viggle_json_response(response):
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code >= 400:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            message = error.get("message", response.reason or "Request rejected.")
        else:
            message = str(error) or response.reason or "Request rejected."
        raise RuntimeError(f"Viggle API returned HTTP {response.status_code}: {message}")
    if not isinstance(payload, dict):
        raise ValueError("Viggle returned an invalid response.")
    return payload