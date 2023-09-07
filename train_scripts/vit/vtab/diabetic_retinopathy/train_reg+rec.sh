
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	pruning/train_reg+rec.py ${VTAB_PATH}/diabetic_retinopathy  --dataset diabetic_retinopathy --num-classes 5 --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 300 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-7 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/diabetic_retinopathy/pruning_rec_loss \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 1e-2  \
    --rec 0.7   \
    --model-path /data/hjy/SSF/vit_base_patch16_224_in21k/vtab/diabetic_retinopathy/baseline-75.84-21.pth.tar   
	# --model-ema --model-ema-decay 0.9  \
