FROM python:3.12-slim

WORKDIR /srv

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY features.py .
COPY app/ app/
COPY artifacts/ artifacts/

# Cloud Run sets $PORT and expects the container to listen on it.
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
