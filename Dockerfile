# Container for hosting photofix, so the phone can reach it from anywhere.
FROM python:3.12-slim

# OpenCV's headless build still needs these two shared libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY photofix ./photofix
COPY webapp ./webapp

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# A 12 megapixel photo peaks around 280 MB, so a 512 MB instance has room for
# exactly one worker; threads handle the rest of the traffic, and the app
# itself only transforms one photo at a time. The long timeout is for a large
# batch arriving in a single request.
CMD ["sh", "-c", "gunicorn webapp.app:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 600"]
