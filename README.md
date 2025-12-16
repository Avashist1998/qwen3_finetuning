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
python3 src/quantized_finetuning_v2.py --run_name 500_sen_8196_r_4_1000 --dataset_path datasets/training_set/5000_sentence_based.json --num_of_epochs 16 --max_length 8196 --dataset_size 3000 --eval_dataset_path datasets/evaluation_set/500_sentence_based.json --from_checkpoint True
```
python3 src/quantized_finetuning_v2.py --run_name 5000_test_2048 --dataset_path datasets/training_set/5000_ner.json --num_of_epochs 8 --max_length 2048

#### Eval

```sh
python3 evaluation.py --type finetuned --peft_model_path peft_lab_outputs/500_sen_8196_r_4_1000/checkpoint-4500/ --max_tokens 8196
```

## Servning Model

### Creating ta LORA adaptor file from Pytorch Checkpoint
```sh
python3 convert_lora_to_gguf.py \
  ~/Documents/projects/role-embedding/peft_lab_outputs/test_run_base/checkpoint-21/ \
  --base /Users/avashist/Documents/models/Qwen3-Embedding-0.6B

```

### Running will llama.cpp
- llama_env needs to activate

#### Base
```sh
llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --port 11434 --embeddings
```

#### With LORA
```sh
llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --port 11434 --lora ~/Documents/projects/role-embedding/peft_lab_outputs/test_run_base/checkpoint-21/checkpoint-21-F16-LoRA.gguf --embeddings
```

llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --port 11434 --lora ~/Documents/projects/role-embedding/peft_lab_outputs/500_sen_8196_r_4_1000/checkpoint-2700/checkpoint-2700-Q8_0-LoRA.gguf --embeddings


llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --port 11434 --lora ~/Documents/projects/role-embedding/peft_lab_outputs/500_sen_8196_r_4_1000/checkpoint-4550/checkpoint-4550--LoRA.gguf --embeddings



Medium artical about the following code base

https://sarinsuriyakoon.medium.com/unsloth-lora-with-ollama-lightweight-solution-to-full-cycle-llm-development-edadb6d9e0f0