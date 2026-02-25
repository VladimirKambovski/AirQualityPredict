FROM python:3.11-slim

WORKDIR /app

# Install dependencies first — Docker caches this layer
# so code changes don't trigger a full reinstall
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs (not training scripts or data)
COPY app.py config.py ./
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
