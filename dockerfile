#docker run --mount type=bind,source="$(pwd)/docker_output",target=/app/experiment_logs --mount type=bind,source="/Volumes/Storage/data_science/trained_NLP_models/model_folder",target=/app/model -e MODEL_PATH="model/" -e TOKENIZER_PATH="model/" -p 8080:5001 clincode-demo:latest bash

FROM python:3.7

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "demo.py"]
