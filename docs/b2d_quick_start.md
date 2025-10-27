# Quick Start

### Set up a new virtual environment
```bash
conda create -n b2d_zoo python=3.8
conda activate b2d_zoo
```

### Install dependency packpages
```bash
pip install ninja packaging
cd close_loop/Bench2Drive/Bench2DriveZoo
mkdir ckpts
# Download resnet50-19c8e357.pth or
# Download r101_dcn_fcos3d_pretrain.pth
cd ..
pip install -v -e .
```

### install carla
```bash
cd close_loop/Bench2Drive/Bench2DriveZoo
mkdir carla
cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
export CARLA_ROOT=YOUR_CARLA_PATH

## Important!!! Otherwise, the python environment can not find carla package
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.8/site-packages/carla.pth # python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME
```

### Prepare the data
Download the dataset from [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) and follow Bench2Drive to organize data.
```bash
cd Bench2DriveZoo/mmcv/datasets
python prepare_B2D.py --workers 16   # workers used to prepare data
```

####  train MomAD 
```
./adzoo/vad/dist_train.sh ./adzoo/vad/configs/VAD/MomAD_base_e2e_b2d.py  1
# or
./adzoo/sparsedrive/dist_train.sh ./adzoo/sparsedrive/configs/sparsedrive_small_b2d_stage2.py 1
```

####  test MomAD （Open_loop in Bench2Drive）
```
./adzoo/vad/dist_test.sh ./adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py 1
```
####  test MomAD (Close_loop in Carla)
```
bash leaderboard/scripts/run_evaluation_multi_vad.sh
```
### Visualization
```
sh scripts/visualize.sh
```


