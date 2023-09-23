
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=0,1,2,3, python  -m torch.distributed.launch --nproc_per_node=4  --master_port=33518 \
	train.py ${CIFAR100_PATH} --dataset torch/cifar100 --num-classes 100 --model as_base_patch4_window7_224 \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output ${OUTPUT_PATH}/as_base_patch4_window7_224/cifar_100/ssf \
	--amp  --tuning-mode ssf --pretrained \