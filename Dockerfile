FROM python:3.11-slim
WORKDIR /app
ENV FASTEMBED_CACHE_PATH=/models PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake all three models into the image: startup and tests never hit the network.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/models'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/models')"
RUN python -c "from huggingface_hub import hf_hub_download as get; \
[get('Xenova/nli-deberta-v3-xsmall', f, revision='2a4f614a701367a02d51389039afc998faeda637', cache_dir='/models') \
 for f in ('config.json', 'tokenizer.json', 'onnx/model_quantized.onnx')]"
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
