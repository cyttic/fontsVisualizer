FROM python:3.11-slim

# opencv needs these
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py pipeline.py ./
COPY static ./static
# Store fonts under a different name so the entrypoint can seed the volume
COPY fonts ./fonts_seed

RUN mkdir -p fonts fonts_modified

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
