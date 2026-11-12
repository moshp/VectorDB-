FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

# Build from the repository root: docker build -f deploy/Dockerfile .
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    DQA_MODELS_DIR=/home/user/models \
    DQA_SESSIONS_DIR=/tmp/dqa-sessions
WORKDIR /home/user/app

# Build llama.cpp for generic x86 CPUs: the build machine's CPU may differ from the one that runs the app.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"
COPY --chown=user deploy/requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake the models into the image so a restart doesn't download 2 GB again.
RUN python -c "\
from huggingface_hub import hf_hub_download; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
hf_hub_download('bartowski/Llama-3.2-3B-Instruct-GGUF', 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', local_dir='/home/user/models'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user sarsvdb ./sarsvdb

EXPOSE 7860
CMD ["python", "-m", "sarsvdb", "serve", "--host", "0.0.0.0", "--port", "7860"]
