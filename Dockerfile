FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /ChatBot

ENV HF_HOME=/ChatBot/data/huggingface_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

CMD ["tail", "-f", "/dev/null"]