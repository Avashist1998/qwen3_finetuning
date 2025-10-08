# Important commands 


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