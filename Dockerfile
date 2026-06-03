FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pymupdf pypdf python-docx python-pptx celery[redis]

COPY . .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["bash", "start.sh"]
