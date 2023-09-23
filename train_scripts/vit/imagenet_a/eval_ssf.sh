CUDA_VISIBLE_DEVICES=0,  python validate_ood.py \
    /path/to/imagenet-a  \
    --num-classes 1000 \
    --model vit_base_patch16_224_in21k \
    --batch-size 64 \
    --no-test-pool 
    --imagenet_a \
	--results-file  output/vit_base_patch16_224_in21k/imagenet_a/ssf \
    --tuning-mode ssf \
    --checkpoint /path/to/vit_base_patch16_224_in21k/imagenet_1k/ssf/model_best.pth.tar


