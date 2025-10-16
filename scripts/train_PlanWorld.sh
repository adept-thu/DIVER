## stage1
# bash ./tools/dist_train.sh \
#    projects/configs/sparsedrive_small_stage1_roboAD.py \
#    1 \
#    --deterministic

## stage2
# bash ./tools/dist_train.sh \
#    projects/configs/sparsedrive_small_stage2_roboAD.py \
#    8 \
#    --deterministic
CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh \
   projects/configs/PlanWorld_small_stage2.py \
   1 \
   --deterministic