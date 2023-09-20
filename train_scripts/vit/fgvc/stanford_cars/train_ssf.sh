source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=2,3  python  -m torch.distributed.launch --nproc_per_node=2  --master_port=12349  \
	train.py ${FGVC_PATH}/cars --dataset stanford_cars --num-classes 196 --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.9998  \
	--output ${OUTPUT_PATH}/vit_base_patch16_224_in21k/fgvc/stanford_cars/ssf \
	--amp --tuning-mode ssf --pretrained  