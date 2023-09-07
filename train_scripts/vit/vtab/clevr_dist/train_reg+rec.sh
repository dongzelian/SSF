
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	pruning/train_reg+rec.py ${VTAB_PATH}/clevr_dist  --dataset clevr_dist --num-classes 6 --model vit_base_patch16_224_in21k  \
    --batch-size 128 -vb 128 --epochs 300 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-7 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/clevr_dist/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 3e-3  \
    --rec 0.8   \
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/vtab/clevr_dist/baseline-61.89-20.pth.tar   
	# --model-ema --model-ema-decay 0.9  \
