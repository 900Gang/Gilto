import pytest

from rag.errors import LLMError
from rag.ocr import OCRCache, OCRPage, OCRPipeline, TesseractOCR, TesseractResult, is_devanagari


class FakeTesseract:
    def __init__(self, text="", confidence=0.0, error=None, available=True):
        self.available = available
        self._result = TesseractResult(text, confidence)
        self._error = error
        self.calls = 0

    def read(self, image):
        self.calls += 1
        if self._error:
            raise self._error
        return self._result


class FakeVisionLLM:
    def __init__(self, text="vision text", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def ocr_image(self, image, mime_type):
        self.calls.append(mime_type)
        if self.error:
            raise self.error
        return self.text


GOOD = "This is a clearly scanned page with enough words on it to count as real text."


def pipeline(tesseract=None, llm=None):
    return OCRPipeline(tesseract, llm, min_chars=50, confidence_threshold=60.0)


def test_confident_tesseract_result_is_used_without_calling_gemini():
    llm = FakeVisionLLM()
    result = pipeline(FakeTesseract(GOOD, 91.0), llm).process(b"img", "image/png")
    assert result.source_type == "ocr"
    assert result.text == GOOD
    assert result.confidence == 91.0
    assert not result.low_confidence
    assert llm.calls == []


def test_low_confidence_escalates_to_vision():
    llm = FakeVisionLLM("clean text from gemini")
    result = pipeline(FakeTesseract(GOOD, 30.0), llm).process(b"img", "image/png")
    assert result.source_type == "vision"
    assert result.text == "clean text from gemini"
    assert llm.calls == ["image/png"]


def test_too_little_text_escalates_to_vision():
    result = pipeline(FakeTesseract("hi", 99.0), FakeVisionLLM()).process(b"img", "image/jpeg")
    assert result.source_type == "vision"


def test_without_tesseract_vision_is_used():
    assert pipeline(None, FakeVisionLLM()).process(b"img", "image/png").source_type == "vision"
    unavailable = FakeTesseract(available=False)
    assert pipeline(unavailable, FakeVisionLLM()).process(b"img", "image/png").source_type == "vision"
    assert unavailable.calls == 0


def test_tesseract_crash_falls_back_to_vision():
    result = pipeline(FakeTesseract(error=RuntimeError("boom")), FakeVisionLLM()).process(b"img", "image/png")
    assert result.source_type == "vision"


def test_without_gemini_a_weak_tesseract_result_is_kept_and_flagged():
    result = pipeline(FakeTesseract(GOOD, 30.0), None).process(b"img", "image/png")
    assert result.source_type == "ocr"
    assert result.low_confidence


def test_vision_failure_keeps_weak_tesseract_result():
    llm = FakeVisionLLM(error=LLMError("Gemini is busy"))
    result = pipeline(FakeTesseract(GOOD, 30.0), llm).process(b"img", "image/png")
    assert result.source_type == "ocr"
    assert result.low_confidence


def test_vision_failure_with_nothing_else_raises():
    with pytest.raises(LLMError, match="busy"):
        pipeline(None, FakeVisionLLM(error=LLMError("Gemini is busy"))).process(b"img", "image/png")


def test_blank_page_returns_none():
    assert pipeline(FakeTesseract("", 0.0), FakeVisionLLM(text="")).process(b"img", "image/png") is None


HINDI = "वारंटी की अवधि खरीद की तारीख से बारह महीने है। वापसी तीस दिनों के भीतर स्वीकार की जाती है।"


def test_is_devanagari_needs_real_hindi_letters():
    assert is_devanagari(HINDI)
    assert not is_devanagari(GOOD)
    assert not is_devanagari("Price । only")  # a lone danda symbol is not Hindi text


def test_hindi_page_goes_to_vision_even_when_tesseract_is_confident():
    llm = FakeVisionLLM("वारंटी की अवधि खरीद की तारीख से बारह महीने है।")
    result = pipeline(FakeTesseract(HINDI, 95.0), llm).process(b"img", "image/png")
    assert result.source_type == "vision"
    assert llm.calls == ["image/png"]


def test_hindi_page_without_gemini_is_kept_but_flagged():
    result = pipeline(FakeTesseract(HINDI, 95.0), None).process(b"img", "image/png")
    assert result.source_type == "ocr"
    assert result.low_confidence


def test_hindi_page_keeps_tesseract_text_when_vision_fails():
    llm = FakeVisionLLM(error=LLMError("Gemini is busy"))
    result = pipeline(FakeTesseract(HINDI, 95.0), llm).process(b"img", "image/png")
    assert result.source_type == "ocr"
    assert result.low_confidence


def test_english_page_still_stays_local():
    llm = FakeVisionLLM()
    result = pipeline(FakeTesseract(GOOD, 95.0), llm).process(b"img", "image/png")
    assert result.source_type == "ocr"
    assert llm.calls == []


def test_available_reflects_engines():
    assert not pipeline(None, None).available
    assert not pipeline(FakeTesseract(available=False), None).available
    assert pipeline(FakeTesseract(), None).available
    assert pipeline(None, FakeVisionLLM()).available


def test_cache_round_trip_and_delete(tmp_path):
    cache = OCRCache(str(tmp_path / "ocr"))
    assert cache.get("abc", 1) is None
    page = OCRPage("हिन्दी text", "vision", None, False)
    cache.put("abc", 1, page)
    assert cache.get("abc", 1) == page
    assert cache.get("abc", 2) is None
    cache.delete("abc")
    assert cache.get("abc", 1) is None


def test_corrupt_cache_entry_is_ignored(tmp_path):
    cache = OCRCache(str(tmp_path))
    cache.put("abc", 1, OCRPage("x", "ocr"))
    (tmp_path / "abc" / "page_0001.json").write_text("{not json", encoding="utf-8")
    assert cache.get("abc", 1) is None


tess = TesseractOCR()


@pytest.mark.skipif(not tess.available, reason="Tesseract is not installed")
def test_real_tesseract_reads_a_rendered_page(png_factory):
    result = tess.read(png_factory("The warranty period is twelve months from the date of purchase."))
    assert "warranty" in result.text.lower()
    assert result.confidence > 60
