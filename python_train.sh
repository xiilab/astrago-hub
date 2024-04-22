# batch_list="8"
# imgsz_list="480"

# for imgsz in $imgsz_list
# do
#     for batch in $batch_list
#     do
#         python train.py \
#         --epochs 100 \
#         --batch-size $batch \
#         --imgsz $imgsz \
#         --validation 50.0 \
#         --checkpoints ../UNet/checkpoints
#     done
# done


python train.py \
    --epochs 3 \
    --batch-size 1 \
    --imgsz 320 \
    --validation 50.0 \