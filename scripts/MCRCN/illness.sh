model_name=MCRCN
seq_len=96
loss='mae'
random_seed=2024

root_path_name=./dataset/illness/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2024
for pred_len in 24 36 48 60
do

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --top_k 3 \
      --data $data_name \
      --seq_len $seq_len \
      --pred_len $pred_len \
            --loss $loss\
            --features M \
      --enc_in 7 \
      --e_layers 2 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --patchlen1 20\
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
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --lradj 'constant'\
      --train_epochs 150 \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.0025 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
