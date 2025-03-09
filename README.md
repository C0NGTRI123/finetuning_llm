# Finetuning R1 with LLM models

We conducted a speed-run on to investigate R1's paradigm in llm models after observing growing interest in R1 and studying the elegant implementation of the GRPO algorithm in open-r1 and trl.

## Installation
### Install uv
```shell
curl -LsSf https://astral.sh/uv/install.sh | less
# or
pip install uv
```

### Creare a environment python3.7 on uv
```shell
uv venv
source .venv/bin/activate
```

### Install the requirements
```shell
uv pip install -r requirements.txt
```

## Run the code train the model
```shell
python src/grpo_train.py
```