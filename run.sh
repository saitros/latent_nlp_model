TASK=style_transfer
DATA_NAME=WNC

MODEL_TYPE=T5
TOKENIZER=T5
VOCAB_SIZE=10000
MAX_LEN=300

BATCH_SIZE=16
NUM_EPOCHS=50
LR=1e-4

MODEL_NAME=tst_${MODEL_TYPE}+waegmm+sim2_${DATA_NAME}_${TOKENIZER}_${VOCAB_SIZE}

clear

python main.py --preprocessing --task=$TASK --data_name=$DATA_NAME --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN --src_vocab_size=$VOCAB_SIZE --trg_vocab_size=$VOCAB_SIZE --tokenizer=$TOKENIZER --isPreTrain=True
python main.py --training --task=$TASK --data_name=$DATA_NAME --model_name=$MODEL_NAME --variational_mode=8 --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN --num_epochs=$NUM_EPOCHS --src_vocab_size=$VOCAB_SIZE --trg_vocab_size=$VOCAB_SIZE --batch_size=$BATCH_SIZE --lr=$LR --tokenizer=$TOKENIZER --isPreTrain=True