#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
#TEAM_AGENT=./team_agent/vad_b2d_agent.py
TEAM_AGENT=leaderboard/team_agent/sparsedrive_b2d_agent.py
#TEAM_CONFIG=/data/songziying/workspace/Bench2Drive/Bench2DriveZoo/adzoo/vad/configs/VAD/MomAD_base_e2e_b2d.py+/data/songziying/workspace/Bench2Drive/Bench2DriveZoo/work_dirs/MomAD_base_e2e_b2d/epoch_5.pth
TEAM_CONFIG=/data/songziying/workspace/SparseDriveb2d/adzoo/sparsedrive/configs/momad_small_b2d_stage2_multiplan.py+/data/songziying/workspace/SparseDriveb2d/ckpt/iter_9782.pth
# /data/songziying/workspace/Bench2Drive/Bench2DriveZoo/ckpts/vad_b2d_base.pth
BASE_CHECKPOINT_ENDPOINT=eval
SAVE_PATH=./eval_v1/
PLANNER_TYPE=only_traj

GPU_RANK=3
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
