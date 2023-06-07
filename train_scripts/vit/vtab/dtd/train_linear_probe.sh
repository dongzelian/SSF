CUDA_VISIBLE_DEVICES=$1  python  -m torch.distributed.launch --nproc_per_node=2  --master_port=$2  \
	train.py /media/disk1/VTAB/vtab-1k/dtd  --dataset dtd --num-classes 47  --no-aug --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  /media/disk1/wyh/SSF/vit_base_patch16_224_in21k/vtab/dtd/linear_probe \
	--amp --tuning-mode linear_probe --pretrained --no-save --seed -1  \
	--mixup 0 --cutmix 0 --smoothing 0