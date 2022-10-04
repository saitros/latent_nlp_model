TASK=style_transfer
DATA_NAME=glue_cola

MODEL_TYPE=custom_transformer
TOKENIZER=bert
VOCAB_SIZE=8000
MAX_LEN=100
VARIATIONAL_MODE=9

BATCH_SIZE=16
NUM_EPOCHS=50
LR=1e-4
DEVICE=cuda:0

MODEL_NAME=0613_${MODEL_TYPE}_${DATA_NAME}_${TOKENIZER}_${VOCAB_SIZE}_VAR${VARIATIONAL_MODE}

clear

python main.py --preprocessing --data_name=$DATA_NAME --tokenizer=$TOKENIZER

# multirc idx issue
# record spm, plm issue


## Super Glue
# boolq : question, passage, label
# cb : premise, hypothesis, label
# copa : premise, choice1, choice2, question, label
# multirc : paragraph, question, answer, label / 여기 index 가 다른데?
# record : passage, query, entities, answers
# rte : premise, hypothesis, label
# wic : word, sentence1, sentence2, start1, start2, end1, end2, label
# wsc : text, span1_index, spna2_index, span1_text, span2_text, label
# wsc.fixed : text, span1_index, spna2_index, span1_text, span2_text, label
# axb(JUST TEST DATASET!!!) : sentence1, sentece2, label
# axg(JUST TEST DATASET!!!) : premise, hypothesis, label

# python main.py --preprocessing --task=$TASK --data_name=$DATA_NAME --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN --src_vocab_size=$VOCAB_SIZE --trg_vocab_size=$VOCAB_SIZE --model_type=$MODEL_TYPE --tokenizer=$TOKENIZER --isPreTrain=True --device=$DEVICE --sentencepiece_model=bpe
# python main.py --training --task=$TASK --data_name=$DATA_NAME --model_name=$MODEL_NAME --variational_mode=$VARIATIONAL_MODE --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN --num_epochs=$NUM_EPOCHS --src_vocab_size=$VOCAB_SIZE --trg_vocab_size=$VOCAB_SIZE --batch_size=$BATCH_SIZE --lr=$LR --model_type=$MODEL_TYPE --tokenizer=$TOKENIZER --isPreTrain=True --device=$DEVICE --sentencepiece_model=bpe
# python main.py --testing --task=$TASK --data_name=$DATA_NAME --model_name=$MODEL_NAME --variational_mode=$VARIATIONAL_MODE --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN --num_epochs=$NUM_EPOCHS --src_vocab_size=$VOCAB_SIZE --trg_vocab_size=$VOCAB_SIZE --batch_size=$BATCH_SIZE --lr=$LR --model_type=$MODEL_TYPE --tokenizer=$TOKENIZER --isPreTrain=True --device=$DEVICE