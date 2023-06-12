
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=0  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=12347  \
	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
    --batch-size 256 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-3 --min-lr 1e-7 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.95  \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/linear_probe \
	--amp --tuning-mode linear_probe --pretrained  \
	# --model-ema --model-ema-decay 0.99992  \ # 2023.06.08 11:50
