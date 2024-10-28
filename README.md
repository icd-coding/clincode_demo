# clincode_demo
 First DEMO for ClinCode user study
 
NOTE: you will need your own clincial LLM, since we're not able to share this publicly yet.

Python 3.7.13

# Running instructions (locally)
Please use virtual environment to avoid problems with python dependencies!
```
pip install -r clincode_demo/requirements.txt
export MODEL_PATH="Path to the trained model on your local machine"
export ID2CAT_PATH="Path to id2cat.json"
export TOKENIZER_PATH="Path to tokenizer, typically the same same as MODEL_PATH"
python demo.py 
```
Open http://localhost:5001/

# Running in docker container
```
docker build -t clincode-demo ./clincode_demo/
docker run \
--mount type=bind,source="$(pwd)",target=/app/experiment_logs \
--mount type=bind,source="Path to the trained model on your local machine",target=/app/model \
-e MODEL_PATH="model/" \
-e TOKENIZER_PATH="model/" \
-p 8080:5001 clincode-demo:latest
```
Open http://localhost:8080/

# Redeploy instructions
1. Pull the latest changes from the repo
```
git pull
```
2. Build and run an updated docker container (model path may need to be updated!)
```
git checkout API
docker stop clincode-demo:latest
docker build -t clincode-demo ./clincode_demo/
sudo docker run -d --mount type=bind,source="/models/experiment_logs",target=/app/experiment_logs \
--mount type=bind,source="/models/model_dir",target=/app/model \
-e MODEL_PATH="model/" \
-e TOKENIZER_PATH="model/" \
-p 8080:5001 clincode-demo2:latest

```
Experiment logs are available in experiment_logs
