model_name=MCRCN
seq_len=96
loss='mae'
random_seed=2024

root_path_name=./dataset/CO2/
data_path_name=China_MI.csv
model_id_name=China_MI
data_name=custom

random_seed=2024
for pred_len in 48 72 96 120
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --top_k 3 \
      --num_layer 1 \
      --kernel_size 4 \
      --conv2d_kernel 4 \
      --data $data_name \
      --seq_len $seq_len \
      --pred_len $pred_len \
            --loss $loss\
            --features M \
      --enc_in 7 \
         --target Total \
      --r 2 \
      --mam_layer 2 \
      --dim 20 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --target Total \
      --patchlen1 24\
      --patchlen2 16\
      --patchlen3 12\
      --patchlen4 25\
      --patchlen5 21\
      --patchlen6 8\
       --patchlen7 19\
      --patchlen8 18\
      --weight1 6880\
      --weight2 3069\
      --weight3 1995\
            --weight4 1390\
      --weight5 1354\
      --weight6 1293\
      --weight7 1143\
      --weight8 615\
      --dropout 0.0\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --train_epochs 150 \
      --patience 10 \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done