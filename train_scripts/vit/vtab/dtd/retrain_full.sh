source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	pruning/retrain_full.py ${VTAB_PATH}/dtd  --dataset dtd --num-classes 47 --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 300 -vb 128 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-7 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/dtd/pruning_retrain_full \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 1e-2\
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/vtab/dtd/dtd_2.13_0.1_0.2_221.tar
    # --lora-rank $4
	# --model-ema --model-ema-decay 0.9  \
