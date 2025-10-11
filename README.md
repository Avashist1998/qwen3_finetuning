# Setting up training machine 

### Clone the repo
```sh
git clone https://github.com/Avashist1998/qwen3_finetuning.git
```

#### create .venv

```sh
uv venv .venv;
source .venv/bin/activate;
uv pip install --upgrade pip;
uv pip install -r requirements.txt;
```

#### Copy the dataset to local
```sh
rsync -avzP  /Users/avashist/Documents/projects/role-embedding/datasets VastAI:/workspace/qwen3_finetuning/
```

#### Add the current checkpoints and logs
```sh
rsync -avzP /Users/avashist/Documents/projects/role-embedding/peft_lab_outputs VastAI:/workspace/qwen3_finetuning/
```

#### Tensorboard
```sh
nohup tensorboard --logdir ./ &
```

#### Pull Models from the machine 
```sh
rsync -avzP VastAI:/workspace/qwen3_finetuning/peft_lab_outputs /Users/avashist/Documents/projects/role-embedding/
rsync -avzP VastAI:/workspace/qwen3_finetuning/results /Users/avashist/Documents/projects/role-embedding/
```


#### Trainig Model
```sh
python3 src/quantized_finetuning_v2.py --run_name 500_sen_8196_r_4_1000 --dataset_path datasets/training_set/5000_sentence_based.json --num_of_epochs 16 --max_length 8196 --dataset_size 1000 --eval_dataset_path datasets/evaluation_set/500_sentence_based.json
```


## Serving the model in production

## Creating ta LORA adaptor file from Checkpoint
```sh
python3 convert_lora_to_gguf.py \
  ~/Documents/projects/role-embedding/peft_lab_outputs/test_run_base/checkpoint-21/ \
  --base /Users/avashist/Documents/models/Qwen3-Embedding-0.6B

```

## Running llm.cpp
llama_env needs to activate


#### Base
```sh
llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --port 11434 --embeddings
```

#### With LORA
```sh
llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --port 11434 --lora ~/Documents/projects/role-embedding/peft_lab_outputs/test_run_base/checkpoint-21/checkpoint-21-F16-LoRA.gguf --embeddings
```



## Running with ollama

### MakeFile

```MakeFile
FROM hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
ADAPTER /Users/avashist/Documents/projects/role-embedding/peft_lab_outputs/test_run_base/checkpoint-21/checkpoint-21-F16-LoRA.gguf
```

### Running the model
```sh
ollama create model_name -f ./Modelfile
```


## Running Training Pipeline 

```sh
python src/quantized_finetuning_v2.py \
  --run_name eval_test \
  --dataset_path datasets/training_set/10_ner.json \
  --num_of_epochs 10 --from_checkpoint true
```