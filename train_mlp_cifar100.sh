CUDA_VISIBLE_DEVICES=0,1  python  -m torch.distributed.launch --nproc_per_node=2  --master_port=12346  \
	train_mlp.py /media/disk1/CIFAR100/ --dataset torch/cifar100 --num-classes-down 100 --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 200 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  /media/disk1/wyh/SSF/vit_base_patch16_224_in21k/cifar100_mlp/ \
	--amp --tuning-mode tail_mlp --pretrained --no-save --seed -1  \
	# --model-ema --model-ema-decay 0.99992  \
	# --output  /media/disk1/wyh/SSF/vit_base_patch16_224_in21k/cifar100_mlp/ \
	# --amp --tuning-mode tail_mlp --pretrained --no-save --seed -1  \