# Quick Start

### downlop DIVER code for NuScenes
```bash
git clone -b nusc --single-branch https://github.com/adept-thu/DIVER.git
```

### Set up a new virtual environment
```bash
conda create -n diver_nus_env python=3.8 -y
conda activate diver_nus_env
```

### Install dependency packpages
```bash
diver_path="path/to/diver"
cd ${diver_path}

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirement.txt
```
:fire: Tips
1. It is necessary to install the `mmcv-full` version of `mmcv`. It is recommended to use `cuda113` and `torch1.11.0`, and to install `mmcv` version `1.7.2`. Use the following command: 
   
   ```
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. Remember to install `flash-attn` offline. The download address for `flash_attn-0.2.2+cu113torch1.11.0-cp38-cp38-linux_x86_64.whl` is [here](https://github.com/Dao-AILab/flash-attention/releases). Move the file to the `diver` folder and execute the following command to install:

   ```
   pip install flash_attn-0.2.2+cu113torch1.11.0-cp38-cp38-linux_x86_64.whl
   ```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.
```bash
cd ${diver_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```

### Generate anchors by K-means
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```


### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Commence training and testing

####  train diver 
```
bash ./tools/dist_train.sh \
   projects/configs/DIVER_small_stage2.py \
   8 \
```

####  test diver
diver_nus.pth:[diver_nus weights]([https://huggingface.co/ZI-YING/DIVER_NuScenes/blob/main/iter_11720.pth])
```
bash ./tools/dist_test.sh \
    projects/configs/diver_small_stage2_roboAD.py \
    work_dirs/DIVER_small_stage2/diver_nus.pth\
    1 \
    --deterministic \
    --eval bbox
```

### Visualization
```
sh scripts/visualize.sh
```

