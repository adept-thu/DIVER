## stage1
# bash ./tools/dist_train.sh \
#    projects/configs/sparsedrive_small_trainval_1_10_stage1_test.py \
#    1 \
#    --deterministic

# stage2
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 bash ./tools/dist_train.sh \
   projects/configs/DIVER_small_stage2.py \
   7 \
   --deterministic

