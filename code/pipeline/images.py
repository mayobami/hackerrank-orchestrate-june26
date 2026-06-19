"""Image resolution, format detection, and encoding for the evidence-review
pipeline.

Image files in this dataset are all named with a ".jpg" extension
regardless of their real format (JPEG, PNG, WebP, or AVIF), so the real
format must always be detected from file content (magic bytes / Pillow's
parsed format), never trusted from the filename.

JPEG / PNG / WebP are natively accepted by the Anthropic Messages API and
are passed through unchanged (base64-encoded) with the correct media_type.
AVIF is not accepted by the API and is transcoded to PNG bytes first.

Per the REQ_GENERAL_MULTI_IMAGE evidence requirement ("each submitted
image should be considered separately"), the loader returns a list of
per-image records rather than a single combined blob.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

try:
    # Registers AVIF support with Pillow as a side effect of import.
    import pillow_avif  # noqa: F401
except ImportError:  # pragma: no cover - environment guard
    pillow_avif = None


# Pillow format name -> Anthropic Messages API media_type for formats that
# are sent through unchanged (no transcoding required).
_NATIVE_MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

# Formats that the API does not accept directly and that we transcode to
# PNG before encoding.
_TRANSCODE_TO_PNG = {"AVIF"}

# Raw magic-byte signatures, used as a fast/independent cross-check ahead of
# (or instead of) Pillow's own format sniffing.
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"RIFF", "WEBP"),  # confirmed further below (bytes 8-12 == "WEBP")
)


@dataclass(frozen=True)
class ImageRecord:
    """One submitted image, resolved, format-sniffed, and encoded.

    On success: real_format/media_type/base64_data are populated and error
    is None. On failure (missing file, unreadable/corrupt image, unknown
    format): error is populated and the data fields are None, so a single
    bad image never crashes the rest of the pipeline.
    """

    image_id: str
    relative_path: str
    absolute_path: str
    real_format: Optional[str]
    media_type: Optional[str]
    base64_data: Optional[str]
    transcoded: bool
    error: Optional[str]

    @property
    def ok(self) -> bool:
        return self.error is None


def _sniff_magic_bytes(data: bytes) -> Optional[str]:
    """Best-effort raw magic-byte sniff, independent of Pillow. Returns a
    Pillow-style format name (JPEG/PNG/WEBP) or None if unrecognized.
    Used as a cross-check; Pillow's Image.open is the authoritative parse.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    # ISO base media file format box, used by AVIF/HEIF: bytes 4-8 are
    # "ftyp" and the following 4-byte brand indicates the subtype.
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis"):
            return "AVIF"
    return None


def resolve_image_path(relative_path: str, dataset_root: Path) -> Path:
    """Resolve an image_paths entry (e.g. "images/test/case_001/img_1.jpg")
    to an absolute path under the dataset root."""
    return (dataset_root / relative_path).resolve()


def _error_record(image_id: str, relative_path: str, absolute_path: Path, message: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        relative_path=relative_path,
        absolute_path=str(absolute_path),
        real_format=None,
        media_type=None,
        base64_data=None,
        transcoded=False,
        error=message,
    )


def load_image(image_id: str, relative_path: str, dataset_root: Path) -> ImageRecord:
    """Resolve, sniff, (if needed) transcode, and base64-encode a single
    image. Never raises; failures are captured in ImageRecord.error.
    """
    absolute_path = resolve_image_path(relative_path, dataset_root)

    if not absolute_path.is_file():
        return _error_record(
            image_id, relative_path, absolute_path, f"file not found: {absolute_path}"
        )

    try:
        raw_bytes = absolute_path.read_bytes()
    except OSError as exc:
        return _error_record(
            image_id, relative_path, absolute_path, f"failed to read file: {exc}"
        )

    if not raw_bytes:
        return _error_record(image_id, relative_path, absolute_path, "file is empty")

    magic_format = _sniff_magic_bytes(raw_bytes)

    try:
        with Image.open(absolute_path) as img:
            pillow_format = img.format
            # Force a full decode so truncated/corrupt files are caught
            # here rather than surfacing later when bytes are used.
            img.load()
    except UnidentifiedImageError:
        return _error_record(
            image_id,
            relative_path,
            absolute_path,
            f"unrecognized image format (magic-byte guess: {magic_format or 'none'})",
        )
    except OSError as exc:
        return _error_record(
            image_id, relative_path, absolute_path, f"failed to decode image: {exc}"
        )

    real_format = pillow_format or magic_format
    if real_format is None:
        return _error_record(
            image_id, relative_path, absolute_path, "could not determine image format"
        )

    if real_format in _TRANSCODE_TO_PNG:
        try:
            with Image.open(absolute_path) as img:
                img = img.convert("RGB") if img.mode in ("CMYK", "P") else img
                out_buffer = _encode_png(img)
        except Exception as exc:  # noqa: BLE001 - any transcode failure is an image-level error
            return _error_record(
                image_id, relative_path, absolute_path, f"AVIF transcode failed: {exc}"
            )
        encoded = base64.b64encode(out_buffer).decode("ascii")
        return ImageRecord(
            image_id=image_id,
            relative_path=relative_path,
            absolute_path=str(absolute_path),
            real_format=real_format,
            media_type="image/png",
            base64_data=encoded,
            transcoded=True,
            error=None,
        )

    media_type = _NATIVE_MEDIA_TYPES.get(real_format)
    if media_type is None:
        return _error_record(
            image_id,
            relative_path,
            absolute_path,
            f"unsupported image format for API submission: {real_format}",
        )

    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return ImageRecord(
        image_id=image_id,
        relative_path=relative_path,
        absolute_path=str(absolute_path),
        real_format=real_format,
        media_type=media_type,
        base64_data=encoded,
        transcoded=False,
        error=None,
    )


def _encode_png(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_images_for_claim(
    image_ids: tuple[str, ...],
    image_paths_list: tuple[str, ...],
    dataset_root: Path,
) -> list[ImageRecord]:
    """Load every image referenced by a claim's image_paths, in order,
    each as its own independent ImageRecord (per REQ_GENERAL_MULTI_IMAGE:
    each submitted image is considered separately). A failure on one image
    does not prevent the others from loading.
    """
    records: list[ImageRecord] = []
    for image_id, relative_path in zip(image_ids, image_paths_list):
        records.append(load_image(image_id, relative_path, dataset_root))
    return records
