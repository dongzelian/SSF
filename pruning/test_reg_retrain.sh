
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=$2  --master_port=$3  \
	train_reg_retrain.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 100 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-7 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/pruning_retrain \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 1e-4\
	# --model-ema --model-ema-decay 0.9  \
