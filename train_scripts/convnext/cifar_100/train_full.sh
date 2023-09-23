
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=33518 \
	train.py ${CIFAR100_PATH} --dataset torch/cifar100 --num-classes 100 --model convnext_base_in22k \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 5e-7 --warmup-epochs 10  \
    --lr 5e-5 --min-lr 5e-8 \
    --drop-path 0.2 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output ${OUTPUT_PATH}/convnext_base_in22k/cifar_100/full \
	--amp  --pretrained  \