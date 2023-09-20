source DATA_PATH.sh

export DATASET="dtd"
export MODEL_PATH="/data/hjy/SSF/ckpts/vtab/retrained/dtd-68.19.pth.tar"
export LORA_RANK=4

CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	pruning/retrain_lora.py ${VTAB_PATH}/${DATASET} --dataset ${DATASET} --num-classes 47 --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 200 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/${DATASET}/pruning_retrain_lora \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--mixup 0 --cutmix 0 --smoothing 0  \
    --model-path ${MODEL_PATH} \
    --lora-rank ${LORA_RANK}
