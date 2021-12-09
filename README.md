# Extrapolação 

Scan (Simplified versions of the Comm AI Navigation tasks).

Lake, B. M. and Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. Proceedings of ICML 2018.


 Download Dataset:

```
wget https://raw.githubusercontent.com/facebookresearch/meta_seq2seq/master/data/tasks_train_length.txt`

wget https://raw.githubusercontent.com/facebookresearch/meta_seq2seq/master/data/tasks_test_length.txt

sed 's/IN:\ //' tasks_train_length.txt | sed 's/\ OUT:\ /\t/' > train.tsv

sed 's/IN:\ //' tasks_test_length.txt | sed 's/\ OUT:\ /\t/' > test.tsv
```
## Dependences
`pip install -q  transformers num2words pytorch_lightning neptune-client`

## Run the code
```
python -u main.py \
      --output_dir=. \
      --model_name_or_path=t5-base \
      --input_train=train.tsv \
      --input_val=test.tsv \
      --input_test=test.tsv \
      --easy_scan \
      --augment_scan=-1.0 \
      --seq2seq \
      --operation=addition \
      --orthography=character_random \
      --balance_train \
      --balance_val \
      --balance_test \
      --invert_question \
      --invert_answer \
      --train_size=10000 \
      --val_size=1000 \
      --test_size=10000 \
      --min_digits_train=2 \
      --max_digits_train=5 \
      --min_digits_test=2 \
      --max_digits_test=5 \
      --base_number=10 \
      --seed=1 \
      --train_batch_size=32 \
      --accumulate_grad_batches=4 \
      --val_batch_size=64 \
      --max_seq_length=512 \
      --num_workers=4 \
      --gpus=1 \
      --optimizer=AdamW \
      --lr=3e-4 \
      --weight_decay=5e-5 \
      --scheduler=StepLR \
      --t_0=2 \
      --t_mult=2 \
      --gamma=1.0 \
      --step_size=1000 \
      --max_epochs=2 \
      --amp_level=O0 \
      --precision=32 \
      --gradient_clip_val=1.0 \
      --check_val_every_n_epoch=1 \

```
