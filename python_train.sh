# imgsz_list="640 1280"
# batch_list="1 2 4 8 16"


# for imgsz in $imgsz_list
# do
#     for batch in $batch_list
#     do
#         python train.py \
#             --batch $batch \
#             --epoch 100 \
#             --imgsz $imgsz \
#             --lr 0.001 \
#             --pretrained True
#     done
# done


python train.py --batch 16  --epoch 100 --imgsz 1280 --lr 0.001 --pretrained True