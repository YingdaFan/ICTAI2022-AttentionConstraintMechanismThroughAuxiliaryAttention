# Attention Constraint Mechanism 
code ICTAI2022 paper "Attention Constraint Mechanism Through Auxiliary Attention."


### Requirements
* [PyTorch](http://pytorch.org/) version == 1.9
* Python version >= 3.8.8   
* [Fairseq](https://github.com/facebookresearch/fairseq/) version == 0.10.1
### Train
Our method is based on [fairseq toolkit](https://github.com/pytorch/fairseq) for training and evaluating. 
The bilingual datasets should be first preprocessed into binary format and save in 'data-bin' file. We list the binay format for [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf) and [WMT'14 English to German dataset](http://www.statmt.org/wmt14/translation-task.html). 

'auxiliary-positive' and 'auxiliary-negative' are corresponding to our proposed two optimization methods for the axuliary attenton, which named as 'the positive', and 'the negative' as mentioned in section II-B and II-C.
```
cd ./fairseq
user_dir=./auxiliary-positive
data_bin=./data-bin/wmt14_en_de_bpe32k
model_dir=./models/auxiliary-positive
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup fairseq-train $data_bin \
        --user-dir $user_dir --criterion auxiliarycriterion --task auxiliary_translation_task --arch transformer_vaswani_wmt_en_de_big1 \
        --optimizer auxiliaryadam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --stop-min-lr 1e-09 \
        --weight-decay 0.0 --label-smoothing 0.1 \
        --max-tokens 2048 --no-progress-bar --max-update 150000 \
        --log-interval 100 --save-interval-updates 1000 --keep-interval-updates 30 --save-interval 10000 --seed 1111 \
        --ddp-backend no_c10d \
        --dropout 0.3 \
        --patience=20 \
        --update-freq 8 \
        -s en -t de --save-dir $model_dir \
        --mask-loss-weight 0.03 > auxiliary-positve.log 2>&1 &
```

### Inference
```
fairseq-generate ./wmt14_en_de_bpe32k \
        --user-dir ./auxiliary-positive \
        --criterion auxiliarycriterion \
	      --task auxiliary_translation_task \ 
        --path ./checkpoint.avg20.pt  \
        --remove-bpe -s en -t de --beam 5 --lenpen 0.6 > gen.out.txt
```

### Post-Process for English to German
```
grep ^H gen.out.txt | sort -n -k 2 -t '-' | cut -f 3 >H.txt
grep ^T gen.out.txt | sort -n -k 2 -t '-' | cut -f 2 >T.txt
./post-process.sh your-path/H.txt your-path/T.txt
```

