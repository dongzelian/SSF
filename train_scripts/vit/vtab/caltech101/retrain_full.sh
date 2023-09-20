source DATA_PATH.sh

export DATASET="caltech101"
export MODEL_PATH="/data/hjy/SSF/ckpts/vtab/pruned/caltech101-82.84.pth.tar"

CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	pruning/retrain_full.py ${VTAB_PATH}/${DATASET} --dataset ${DATASET} --num-classes 102 --model vit_base_patch16_224_in21k  \
    --batch-size 128 --epochs 100 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-7 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/${DATASET}/pruning_retrain_full \
	--amp --tuning-mode ssf --pretrained --seed 1  \
    --model-path ${MODEL_PATH}
