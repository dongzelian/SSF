
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,  python  -m torch.distributed.launch --nproc_per_node=8  --master_port=33518 \
	train.py /path/to/imagenet_1k  --dataset imagenet --num-classes 1000 --model convnext_base_in22k \
    --batch-size 32 --epochs 30 \
	--opt adamw --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 5  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output ${OUTPUT_PATH}/convnext_base_in22k/imagenet_1k/ssf \
	--amp --tuning-mode ssf --pretrained  
