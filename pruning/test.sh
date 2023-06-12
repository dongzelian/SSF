# 流程测试用

# CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
# 	train_regular.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
#     --batch-size 128 --epochs 1 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 0  \
#     --lr 5e-3 --min-lr 2e-7 --sched step\
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/pruning \
# 	--amp  --tuning-mode ssf --pretrained --no-save --seed -1  \
# 	# --amp  --tuning-mode ssf --pretrained --no-save --seed -1  \

# # 训练用 
# # 2023年6月8日23:42 
# reg=2e-5
# # 2023年6月9日 2023年6月9日10:17

source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
	train_regular.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
    --batch-size 64 --epochs 100 \
	--opt adamw  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-7 \
    --drop-path 0 --img-size 224 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/pruning_no_ema \
	--amp --tuning-mode ssf --pretrained --seed 1  \
	--reg 1e-4\
	# --model-ema --model-ema-decay 0.99992  \

# # # 2023年6月9日10:12:05
# CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
# 	train_regular.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 64 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-7 \
#     --drop-path 0 --img-size 224 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/pruning_no_ema \
# 	--amp --tuning-mode ssf --pretrained --seed 1  \
# 	--reg 1e-4\
# 	# --model-ema --model-ema-decay 0.99992  \

# # 2023年6月8日23:44:21
# reg=2e-5
# CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
# 	train_regular.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 64 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-7 \
#     --drop-path 0 --img-size 224 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/pruning_no_ema \
# 	--amp --tuning-mode ssf --pretrained --seed 1  \
# 	--model-ema --model-ema-decay 0.99992  \

# 测试用

# CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
# 	train_regular.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 64 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/pruning \
# 	--amp --tuning-mode ssf --pretrained --no-save --seed -1  \
# 	--resume ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/pruning/20230608-204025-vit_base_patch16_224_in21k-224/model_best.pth.tar \
# 	--evaluate \


# #  记录稀疏后，插入参数的分布变化
# CUDA_VISIBLE_DEVICES=$1  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=$2  \
# 	train_regular.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 128 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/pruning \
# 	--amp --tuning-mode ssf --pretrained --seed -1  \