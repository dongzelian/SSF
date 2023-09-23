source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	train.py ${FGVC_PATH} --dataset nabirds --num-classes 555  --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-4 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--model-ema --model-ema-decay 0.9998  \
	--output ${OUTPUT_PATH}/vit_base_patch16_224_in21k/nabirds/ssf \
	--amp --tuning-mode ssf --pretrained  