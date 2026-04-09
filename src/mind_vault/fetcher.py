"""Content fetcher — downloads and extracts text from URLs.

Handles articles (trafilatura + BS4 fallback), YouTube transcripts, and PDFs.
All extraction libraries are optional — import errors give clear messages.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

_MAX_CONTENT_LENGTH = 100_000
_FETCH_RETRIES = 3
_FETCH_RETRY_BACKOFF = 2.0

_UNSUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".mp3", ".zip"}
_UNSUPPORTED_DOMAINS = {"twitter.com", "x.com", "instagram.com", "tiktok.com"}
_YOUTUBE_DOMAINS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}


class ContentType(Enum):
    ARTICLE = "article"
    PDF = "pdf"
    YOUTUBE = "youtube"
    UNSUPPORTED = "unsupported"


@dataclass
class FetchResult:
    success: bool
    text: str = ""
    title: str = ""
    source_type: str = ""
    error: str = ""
    truncated: bool = False


def detect_type(url: str) -> ContentType:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path.lower()

    if hostname in _YOUTUBE_DOMAINS:
        return ContentType.YOUTUBE
    if hostname in _UNSUPPORTED_DOMAINS:
        return ContentType.UNSUPPORTED
    _, _, ext = path.rpartition(".")
    if ext and f".{ext}" in _UNSUPPORTED_EXTENSIONS:
        return ContentType.UNSUPPORTED
    if path.endswith(".pdf"):
        return ContentType.PDF
    return ContentType.ARTICLE


def _fetch_article(url: str) -> FetchResult:
    try:
        import trafilatura
    except ImportError:
        return FetchResult(success=False, error="trafilatura not installed. Install with: pip install trafilatura")

    last_error = ""
    for attempt in range(_FETCH_RETRIES):
        if attempt > 0:
            time.sleep(_FETCH_RETRY_BACKOFF ** attempt)

        html = trafilatura.fetch_url(url)
        if html is None:
            last_error = f"fetch_url returned None (attempt {attempt + 1})"
            continue

        text = trafilatura.extract(html)
        if text:
            metadata = trafilatura.extract_metadata(html)
            title = metadata.title if metadata and metadata.title else ""
            return FetchResult(success=True, text=text, title=title, source_type="article")

        # Fallback to BS4
        try:
            import httpx
            from bs4 import BeautifulSoup
            response = httpx.get(url, follow_redirects=True, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if text:
                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else ""
                return FetchResult(success=True, text=text, title=title, source_type="article")
        except ImportError:
            last_error = "bs4/httpx not installed for fallback"
        except Exception as exc:
            last_error = str(exc)

        last_error = "trafilatura and BS4 fallback both failed"

    return FetchResult(success=False, source_type="article", error=last_error)


def _fetch_youtube(url: str) -> FetchResult:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return FetchResult(success=False, error="youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")

    video_id = _extract_video_id(url)
    if not video_id:
        return FetchResult(success=False, source_type="youtube", error=f"Could not extract video ID from: {url}")

    try:
        yt = YouTubeTranscriptApi()
        transcript = yt.fetch(video_id)
        text = "\n".join(snippet.text for snippet in transcript.snippets)
        return FetchResult(success=True, text=text, title="", source_type="youtube")
    except Exception as exc:
        return FetchResult(success=False, source_type="youtube", error=str(exc))


def _fetch_pdf(url: str) -> FetchResult:
    try:
        import httpx
        import pymupdf
    except ImportError:
        return FetchResult(success=False, error="httpx and pymupdf required for PDF. Install with: pip install httpx pymupdf")

    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        doc = pymupdf.open(stream=response.content, filetype="pdf")
        pages = [page.get_text() for page in doc]
        text = "\n".join(pages)
        return FetchResult(success=True, text=text, title="", source_type="pdf")
    except Exception as exc:
        return FetchResult(success=False, source_type="pdf", error=str(exc))


def _extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname == "youtu.be":
        return parsed.path.lstrip("/")
    if "/embed/" in parsed.path:
        return parsed.path.split("/embed/")[1].split("/")[0]
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    return ""


def fetch_url(url: str) -> FetchResult:
    """Fetch and extract text content from a URL.

    Supports articles, YouTube videos, and PDFs.
    Returns a FetchResult with the extracted text or an error message.
    """
    content_type = detect_type(url)

    if content_type == ContentType.UNSUPPORTED:
        return FetchResult(success=False, source_type="unsupported", error=f"Unsupported content type for URL: {url}")

    handlers = {
        ContentType.ARTICLE: _fetch_article,
        ContentType.YOUTUBE: _fetch_youtube,
        ContentType.PDF: _fetch_pdf,
    }
    result = handlers[content_type](url)

    if result.success and len(result.text) > _MAX_CONTENT_LENGTH:
        result.text = result.text[:_MAX_CONTENT_LENGTH]
        result.truncated = True

    return result
