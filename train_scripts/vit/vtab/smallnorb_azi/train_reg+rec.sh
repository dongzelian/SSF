
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	pruning/train_reg+rec.py ${VTAB_PATH}/smallnorb_azi  --dataset smallnorb_azi --num-classes 18 --model vit_base_patch16_224_in21k  \
    --batch-size 128 -vb 128 --epochs 300 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/smallnorb_azi/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 5e-4  \
    --rec 0.8   \
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/vtab/smallnorb_azi/baseline-29.62-114.pth.tar   
	# --model-ema --model-ema-decay 0.9  \
