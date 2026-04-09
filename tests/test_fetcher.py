"""Tests for the URL fetcher module."""

from mind_vault.fetcher import detect_type, ContentType, _extract_video_id


def test_detect_article():
    assert detect_type("https://example.com/blog/post") == ContentType.ARTICLE


def test_detect_youtube():
    assert detect_type("https://www.youtube.com/watch?v=abc123") == ContentType.YOUTUBE
    assert detect_type("https://youtu.be/abc123") == ContentType.YOUTUBE


def test_detect_pdf():
    assert detect_type("https://arxiv.org/pdf/1706.03762.pdf") == ContentType.PDF


def test_detect_unsupported_domain():
    assert detect_type("https://x.com/user/status/123") == ContentType.UNSUPPORTED
    assert detect_type("https://instagram.com/p/abc") == ContentType.UNSUPPORTED


def test_detect_unsupported_extension():
    assert detect_type("https://example.com/image.png") == ContentType.UNSUPPORTED
    assert detect_type("https://example.com/video.mp4") == ContentType.UNSUPPORTED


def test_extract_video_id_watch():
    assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_short():
    assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_embed():
    assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    assert _extract_video_id("https://example.com") == ""
