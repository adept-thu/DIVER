## stage1
# bash ./tools/dist_train.sh \
#    projects/configs/sparsedrive_small_trainval_1_10_stage1_test.py \
#    1 \
#    --deterministic

# stage2
bash ./adzoo/sparsedrive/tools/dist_train.sh \
   ./adzoo/sparsedrive/configs/sparsedrive_small_stage2.py \
   8 \
   --deterministic

