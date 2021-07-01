export CUDA_VISIBLE_DEVICES=0
bz=4
epn=3
sc=2
dfmm=0
model_type=bert
pm=bert-base-uncased
data_dir=/home/zehao.yu/workspace/py3/dr_ann/data/dr_aio_th1
nmd=/home/zehao.yu/workspace/py3/dr_ann/relation_models
pof=/home/zehao.yu/workspace/py3/dr_ann/result/predictions.txt
log=/home/zehao.yu/workspace/py3/dr_ann/logs/log_bert_relation.txt

python3 /home/zehao.yu/workspace/py3/dr_ann/ClinicalTransformerRelationExtraction/src/relation_extraction.py \
		--model_type $model_type \
		--data_format_mode $dfmm \
		--classification_scheme $sc \
		--pretrained_model $pm \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 512 \
		--cache_data \
		--do_train \
		--do_predict \
		--do_lower_case \
		--train_batch_size $bz \
		--eval_batch_size $bz \
		--learning_rate 1e-5 \
		--num_train_epochs $epn \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 0 \
		--log_file $log \


# example of testing and convert predictions to brat
export CUDA_VISIBLE_DEVICES=0
python3 /home/zehao.yu/workspace/py3/dr_ann/ClinicalTransformerRelationExtraction/src/relation_extraction.py \
		--model_type $model_type \
		--data_format_mode $dfmm \
		--classification_scheme $sc \
		--pretrained_model $pm \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 512 \
		--cache_data \
		--do_predict \
		--do_lower_case \
		--train_batch_size $bz \
		--eval_batch_size $bz \
		--learning_rate 1e-5 \
		--num_train_epochs $epn \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 0 \
		--log_file $log \

