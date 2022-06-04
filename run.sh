TASK=style_transfer
DATA_NAME=WNC
MODEL_NAME=tst_basic
MAX_LEN=150

clear

python main.py --preprocessing --task=$TASK --data_name=$DATA_NAME --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN
python main.py --training --task=$TASK --data_name=$DATA_NAME --model_name=$MODEL_NAME --src_max_len=$MAX_LEN --trg_max_len=$MAX_LEN --variational=True