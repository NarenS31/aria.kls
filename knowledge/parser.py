"""
File text extraction for the ARIA knowledge base.
Supports: PDF, TXT, MD, MP4, MP3, YouTube URLs.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Chunk helper
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    """Split text into ~max_tokens-word chunks with overlap."""
    words = text.split()
    chunks = []
    step = max_tokens - 64  # 64-word overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_tokens])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def parse_pdf(path: str) -> List[Tuple[str, dict]]:
    """Return list of (chunk_text, metadata) tuples."""
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("pypdf2 not installed. Run: pip install pypdf2")

    results = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            for i, chunk in enumerate(chunk_text(text)):
                results.append((chunk, {"page_number": page_num, "chunk_index": i}))
    return results


# ---------------------------------------------------------------------------
# Plain text / Markdown
# ---------------------------------------------------------------------------

def parse_text(path: str) -> List[Tuple[str, dict]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return [(chunk, {"page_number": 1, "chunk_index": i})
            for i, chunk in enumerate(chunk_text(content))]


# ---------------------------------------------------------------------------
# Audio / Video (whisper transcription)
# ---------------------------------------------------------------------------

def parse_audio_video(path: str) -> List[Tuple[str, dict]]:
    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")

    model = whisper.load_model("base")
    result = model.transcribe(path)
    full_text = result.get("text", "")
    chunks = []
    for i, chunk in enumerate(chunk_text(full_text)):
        chunks.append((chunk, {"page_number": 1, "chunk_index": i, "timestamp": ""}))

    # Attach timestamps from segments if available
    segments = result.get("segments", [])
    if segments:
        chunks = []
        # Re-chunk using segment boundaries
        seg_texts = []
        for seg in segments:
            seg_texts.append((seg.get("start", 0), seg.get("text", "")))

        combined = " ".join(t for _, t in seg_texts)
        word_chunks = chunk_text(combined)
        for i, wc in enumerate(word_chunks):
            ts = seg_texts[min(i * 10, len(seg_texts) - 1)][0] if seg_texts else 0
            chunks.append((wc, {"page_number": 1, "chunk_index": i, "timestamp": f"{ts:.1f}s"}))

    return chunks


# ---------------------------------------------------------------------------
# YouTube URL
# ---------------------------------------------------------------------------

def parse_youtube(url: str) -> List[Tuple[str, dict]]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise ImportError(
            "youtube-transcript-api not installed. Run: pip install youtube-transcript-api"
        )

    video_id_match = re.search(
        r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url
    )
    if not video_id_match:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    video_id = video_id_match.group(1)
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join(item["text"] for item in transcript_list)

    chunks = []
    for i, chunk in enumerate(chunk_text(full_text)):
        idx = min(i * 10, len(transcript_list) - 1)
        ts = transcript_list[idx].get("start", 0) if transcript_list else 0
        chunks.append((chunk, {"page_number": 1, "chunk_index": i, "timestamp": f"{ts:.1f}s"}))
    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def parse_file(path: str) -> List[Tuple[str, dict]]:
    """Parse any supported file and return (chunk, base_metadata) pairs."""
    lower = path.lower()
    if lower.endswith(".pdf"):
        return parse_pdf(path)
    elif lower.endswith((".txt", ".md")):
        return parse_text(path)
    elif lower.endswith((".mp4", ".mp3", ".m4a", ".wav", ".ogg", ".webm")):
        return parse_audio_video(path)
    else:
        # Try plain text as fallback
        return parse_text(path)


def is_youtube_url(text: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", text))
