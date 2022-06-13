TASK=style_transfer
DATA_NAME=WNC

MODEL_TYPE=custom_transformer
TOKENIZER=T5
VOCAB_SIZE=8000
MAX_LEN=100
VARIATIONAL_MODE=9

BATCH_SIZE=16
NUM_EPOCHS=50
LR=1e-4
DEVICE=cuda:0

MODEL_NAME=0613_${MODEL_TYPE}_${DATA_NAME}_${TOKENIZER}_${VOCAB_SIZE}_VAR${VARIATIONAL_MODE}

clear

python main.py --preprocessing --task=$TASK --data_name=$DATA_NAME --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN --src_vocab_size=$VOCAB_SIZE --trg_vocab_size=$VOCAB_SIZE --model_type=$MODEL_TYPE --tokenizer=$TOKENIZER --isPreTrain=True --device=$DEVICE --sentencepiece_model=bpe
python main.py --training --task=$TASK --data_name=$DATA_NAME --model_name=$MODEL_NAME --variational_mode=$VARIATIONAL_MODE --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN --num_epochs=$NUM_EPOCHS --src_vocab_size=$VOCAB_SIZE --trg_vocab_size=$VOCAB_SIZE --batch_size=$BATCH_SIZE --lr=$LR --model_type=$MODEL_TYPE --tokenizer=$TOKENIZER --isPreTrain=True --device=$DEVICE --sentencepiece_model=bpe
python main.py --testing --task=$TASK --data_name=$DATA_NAME --model_name=$MODEL_NAME --variational_mode=$VARIATIONAL_MODE --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN --num_epochs=$NUM_EPOCHS --src_vocab_size=$VOCAB_SIZE --trg_vocab_size=$VOCAB_SIZE --batch_size=$BATCH_SIZE --lr=$LR --model_type=$MODEL_TYPE --tokenizer=$TOKENIZER --isPreTrain=True --device=$DEVICE