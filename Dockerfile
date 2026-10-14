FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN sed -i 's/\r$//' start.sh \
    && chmod +x start.sh \
    && useradd --create-home gilto \
    && mkdir -p /data \
    && chown -R gilto:gilto /app /data

USER gilto
EXPOSE 8501
CMD ["./start.sh"]