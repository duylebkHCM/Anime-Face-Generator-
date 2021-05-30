#!/usr/bin/env bash
python main.py --gan_type DCGAN --data_root dataset/raw/ --epoch 100 --batch_size 128 --input_size 64 --hdg 64 --hdd 64 \
                                    --save_dir checkpoints --result_dir results  --lrG 0.0002 --lrD 0.0002 --beta1 0.5 --beta2 0.999