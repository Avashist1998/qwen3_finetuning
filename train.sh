# Script to start the training
# Set the PYTHONPATH to the current directory
export PYTHONPATH=$PWD
export KMP_DUPLICATE_LIB_OK=TRUE

MODEL_ID="Qwen/Qwen3-Embedding-0.6B"
# MODEL_ID="intfloat/multilingual-e5-small"
# MODEL_ID="ibm-granite/granite-4.0-h-small-GGUF"
R=1
MAX_LENGTH=512
# MAX_LENGTH=8196
NUM_OF_EPOCHS=60
DATASET_SIZE=30
EVAL_SET_SIZE=5
DATASET_PATH="datasets/training_set/5000_sentence_based.json"
EVAL_DATASET_PATH="datasets/evaluation_set/500_sentence_based.json"
FROM_CHECKPOINT=False

# R = 1 # r=2, r=4, # Change to 2 to make on mac

pixi run python src/quantized_finetuning_v2.py --model_id $MODEL_ID \
    --r $R --max_length $MAX_LENGTH --dataset_size $DATASET_SIZE \
    --evalset_size $EVAL_SET_SIZE \
    --eval_dataset_path $EVAL_DATASET_PATH \
    --num_of_epochs $NUM_OF_EPOCHS \
    --dataset_path $DATASET_PATH \
    --from_checkpoint $FROM_CHECKPOINT
