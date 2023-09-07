source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	pruning/retrain_full.py ${VTAB_PATH}/dsprites_ori  --dataset dsprites_ori --num-classes 16 --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 300 -vb 128 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/dsprites_ori/pruning_retrain_full \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 5e-3\
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/vtab/dsprites_ori/rec=0.85_7.56_238.pth.tar
    # --lora-rank $4
	# --model-ema --model-ema-decay 0.9  \
