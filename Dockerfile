FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime


WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install "numpy<2" ultralytics \
    scikit-learn pillow python-multipart tqdm joblib fastapi uvicorn \
    streamlit requests transformers accelerate

    EXPOSE 8501

    CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
