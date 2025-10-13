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

bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage2_roboAD_6s.py \
   8 \
   --deterministic