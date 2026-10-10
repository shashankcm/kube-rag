FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-ui.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements-ui.txt

# Copy app
COPY ui/ ./ui/
COPY app/ ./app/

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
