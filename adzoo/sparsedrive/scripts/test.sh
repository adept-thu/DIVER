CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\ 
   bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    work_dirs/sparsedrive_small_stage2/iter_5860.pth \
    8 \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl