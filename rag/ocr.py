"""OCR for scanned pages and images.

Tesseract runs first (local, free). When it is missing, unsure, finds too
little text, or reads Hindi (which it gets wrong too often), the page goes to
Gemini vision. Results are cached on disk by
file hash and page number so a retry never pays twice.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings
from .errors import LLMError
from .llm import LLM

WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
MIN_DEVANAGARI_CHARS = 3


def is_devanagari(text: str) -> bool:
    """True when the text has Hindi (Devanagari) letters, not just a stray symbol."""
    return len(_DEVANAGARI.findall(text)) >= MIN_DEVANAGARI_CHARS


@dataclass
class OCRPage:
    text: str
    source_type: str  # "ocr" (Tesseract) or "vision" (Gemini)
    confidence: float | None = None  # Tesseract mean word confidence, 0-100
    low_confidence: bool = False


@dataclass
class TesseractResult:
    text: str
    confidence: float


class TesseractOCR:
    """Thin wrapper over pytesseract. `available` is False when the package, the
    Tesseract program, or every requested language pack is missing."""

    def __init__(self, langs: str = "eng+hin", cmd: str = "") -> None:
        self.available = False
        self._lang = ""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return
        path = cmd or shutil.which("tesseract") or (WINDOWS_TESSERACT if os.path.exists(WINDOWS_TESSERACT) else "")
        if path:
            pytesseract.pytesseract.tesseract_cmd = path
        try:
            installed = set(pytesseract.get_languages(config=""))
        except Exception:  # Tesseract program not found or not runnable
            return
        wanted = [code for code in langs.split("+") if code in installed]
        if not wanted:
            return
        self._pytesseract = pytesseract
        self._image_module = Image
        self._lang = "+".join(wanted)
        self.available = True

    def read(self, image: bytes) -> TesseractResult:
        img = self._image_module.open(io.BytesIO(image))
        data = self._pytesseract.image_to_data(img, lang=self._lang, output_type=self._pytesseract.Output.DICT)
        lines: dict[tuple, list[str]] = {}
        confidences: list[float] = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = float(data["conf"][i])
            if not word or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            confidences.append(conf)
        text = "\n".join(" ".join(words) for words in lines.values())
        mean = sum(confidences) / len(confidences) if confidences else 0.0
        return TesseractResult(text=text, confidence=mean)


class OCRCache:
    def __init__(self, root: str) -> None:
        self._root = Path(root)

    def _file(self, doc_id: str, page: int) -> Path:
        return self._root / doc_id / f"page_{page:04d}.json"

    def get(self, doc_id: str, page: int) -> OCRPage | None:
        try:
            return OCRPage(**json.loads(self._file(doc_id, page).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def put(self, doc_id: str, page: int, result: OCRPage) -> None:
        path = self._file(doc_id, page)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")

    def delete(self, doc_id: str) -> None:
        shutil.rmtree(self._root / doc_id, ignore_errors=True)


class OCRPipeline:
    def __init__(
        self,
        tesseract: TesseractOCR | None,
        llm: LLM | None,
        min_chars: int = 50,
        confidence_threshold: float = 60.0,
    ) -> None:
        self._tesseract = tesseract if tesseract is not None and tesseract.available else None
        self._llm = llm
        self._min_chars = min_chars
        self._threshold = confidence_threshold

    @property
    def available(self) -> bool:
        return self._tesseract is not None or self._llm is not None

    def process(self, image: bytes, mime: str) -> OCRPage | None:
        """Return the page text, or None if nothing readable was found.
        Raises LLMError only when Gemini failed and there is no other result."""
        local = None
        if self._tesseract is not None:
            try:
                local = self._tesseract.read(image)
            except Exception:
                local = None  # fall through to vision

        # Tesseract's Hindi output is often wrong while it still reports high confidence,
        # so Devanagari pages always go to vision when it is available.
        indic = local is not None and is_devanagari(local.text)
        if local and not indic and len(local.text.strip()) >= self._min_chars and local.confidence >= self._threshold:
            return OCRPage(local.text.strip(), "ocr", local.confidence)

        vision_error: LLMError | None = None
        if self._llm is not None:
            try:
                text = self._llm.ocr_image(image, mime)
                if text.strip():
                    return OCRPage(text.strip(), "vision")
            except LLMError as exc:
                vision_error = exc

        if local and local.text.strip():
            return OCRPage(local.text.strip(), "ocr", local.confidence, low_confidence=True)
        if vision_error is not None:
            raise vision_error
        return None


def build_ocr(settings: Settings, llm: LLM | None) -> OCRPipeline | None:
    """None when there is no way to read a scan (no Tesseract and no Gemini key)."""
    tesseract = TesseractOCR(settings.ocr_langs, settings.tesseract_cmd)
    pipeline = OCRPipeline(tesseract, llm, settings.min_page_chars, settings.ocr_confidence_threshold)
    return pipeline if pipeline.available else None
