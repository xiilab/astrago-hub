# python train.py \
#     --data-path /workspace/COCO \
#     --dataset coco \
#     --model deeplabv3_resnet50 \
#     --imgsz 640 \
#     --batch-size 4 \
#     --epochs 100 \
#     --lr 0.02 \
#     --aux-loss \
#     --weights-backbone ResNet50_Weights.IMAGENET1K_V1


model_list="fcn_resnet50 fcn_resnet101 deeplabv3_resnet50 deeplabv3_resnet101 deeplabv3_mobilenet_v3_large lraspp_mobilenet_v3_large"
imgsz_list="320 480 640 1280"
batch_list="1 2 4 8 16 32"

for model in $model_list
do
    for imgsz in $imgsz_list
    do
        for batch in $batch_list
        do
            python train.py \
                --data-path /workspace/COCO \
                --dataset coco \
                --model $model \
                --imgsz $imgsz \
                --batch-size $batch \
                --epochs 100 \
                --lr 0.02 \
                --aux-loss 
        done
    done
done
