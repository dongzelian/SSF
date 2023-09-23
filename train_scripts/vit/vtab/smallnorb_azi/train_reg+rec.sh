source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	pruning/train_reg+rec.py ${VTAB_PATH}/smallnorb_azi  --dataset smallnorb_azi --num-classes 18 --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 300 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-2 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/smallnorb_azi/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 1e-4  \
    --rec 0.5   \
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/vtab/smallnorb_azi/baseline-29.62-114.pth.tar   \
    --ratio5-epochs 20
	# --model-ema --model-ema-decay 0.9  \
