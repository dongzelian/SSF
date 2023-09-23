CUDA_VISIBLE_DEVICES=0,1,2,3, python  -m torch.distributed.launch --nproc_per_node=4  --master_port=27524 \
	train.py /path/to/cifar100 --dataset torch/cifar100 --num-classes 100 --model convnext_base_in22k \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output  output/convnext_base_in22k/cifar_100/ssf \
	--amp --tuning-mode ssf  --pretrained  \