# [DIVER] Breaking Imitation Bottlenecks: Reinforced Diffusion Powers Diverse Trajectory Generation

<div align="justify">  

This is the official repository of [**DIVER**](https://arxiv.org/abs/2503.03125). 



</div>

<div align="center">
  <img src="open_loop/vis_5(3).png" />
</div>

## Abstract
<div align="justify">
  Existing end-to-end autonomous driving (E2E-AD) methods predominantly rely on single expert demonstrations through imitation learning, often leading to conservative and homogeneous driving behaviors that struggle to generalize to complex real-world scenarios. In this work, we propose \textbf{DIVER}, a novel E2E-AD framework that combines diffusion-based multi-mode trajectory generation with reinforcement learning to produce diverse, safe, and goal-directed trajectories. First, the model conditions on map elements and surrounding agents to generate multiple reference trajectories from each ground-truth reference trajectory that overcome the inherent limitations of single-mode imitation. Second, we treat the diffusion process as a stochastic policy and employ Group Relative Policy Optimization (GPRO) objectives to guide the diffusion process. By optimizing trajectory-level rewards for both diversity and safety, GRPO directly mitigates mode collapse and enhances collision avoidance, encouraging exploration beyond expert demonstrations and ensuring physically plausible plans. Furthermore, to address the limitations of L2-based open-loop metrics in capturing trajectory diversity, we propose a novel trajectory diversity metric to evaluate the diversity of multi-mode predictions. Extensive experiments on the closed-loop NAVSIM and Bench2Drive benchmarks, as well as the open-loop nuScenes dataset, demonstrate that DIVER significantly improves trajectory diversity, effectively addressing the mode collapse problem inherent in imitation learning.
</div>
<div align="justify">



:fire: Contributions:
* **DIVER Concept.**  We propose the \textbf{DIVER}, an novel multi-mode E2E-AD framework that uses reinforcement learning to guide diffusion models in generating diverse and feasible driving behaviors.

* **Diffusion Model.** We introduce the \textbf{Policy-Aware Diffusion Generator (PADG)}, which incorporates map elements and agent interactions as conditional inputs, enabling the generation of multi-mode trajectory that capture diverse driving styles.

* **Reinforcement Learning.** We leverage reinforcement learning to guide the diffusion model with diversity and safety rewards, addressing the limitations of imitation learning.

* **Diversity Metric.** We propose a novel \textbf{Diversity Metric} to evaluate multi-mode trajectory generation, providing a more principled way to assess the diversity and effectiveness of generated trajectories compared to existing metrics.

* **Performance Evaluation.** Extensive evaluations on the Bench2Drive, NAVSIM, NuScenes demonstrate that DIVER significantly improves the diversity, safety, and feasibility of generated trajectories over state-of-the-art methods.
* </div>




## Method
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="main.png" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">The overall architecture of DIVER. As a multi-mode trajectory E2E-AD framework, DIVER first encodes multi-view images into feature maps to extract scene representations through a perception module. It then predicts the motion of surrounding agents and performs planning via a conditional diffusion model guided by reinforcement learning to generate diverse multi-intention trajectories. Our approach effectively addresses the inherent mode collapse in imitation learning, enabling the generation of safe and diverse behaviors for complex driving scenarios.</div>
</center>


## Results in paper

### Open-loop mertics

- Planning results on [nuScenes](https://github.com/nutonomy/nuscenes-devkit).
- DIVER 3s stage2: [ckpt](https://huggingface.co/ZI-YING/DIVER_nuScenes_stage2_planing_3s)

| Method |  L2 (m) 1s  | L2 (m) 2s | L2 (m) 3s | L2 (m) Avg | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | Col. (%) Avg | 
| :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UniAD | 0.07| 0.14| 0.24| 0.15 |0.03| 0.05| 0.16| 0.08 |
SparseDrive |0.05| 0.11| 0.23| 0.13| **0.01**| 0.05| 0.18| 0.08|
**DIVER (Ours)**   | **0.10**| **0.19**| **0.34**| **0.21**| **0.01**| **0.05**| **0.15**| **0.07**|

- Planning results on the Turning-nuScenes validation dataset [Turning-nuScenes ](nuscenes_infos_val_hrad_planing_scene.pkl). 

| Method |L2 (m) 1s  | L2 (m) 2s | L2 (m) 3s  | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | TPC (m) 1s | TPC (m) 2s | TPC (m) 3s |
| :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|SparseDrive| 0.09| 0.18| 0.36| 0.21| 0.04| 0.17| 0.98| 0.40|
|DiffusionDrive| 0.11| 0.21| 0.37| 0.23| **0.03**| 0.14| 0.85| 0.34|
|MomAD |0.09| 0.17 |0.34| 0.20 |**0.03** |0.13| 0.79| 0.32|
|**DIVER (Ours)** |**0.17**| **0.29**| **0.47**| **0.31**| **0.03**| **0.11**| **0.67**| **0.27**|


### Close-loop mertics (**weight** and **pkl**)

- Open-loop and Closed-loop Results of E2E-AD Methods in Bench2Drive (V0.0.3)} under base training set. `mmt' denotes the extension of VAD on Multi-modal Trajectory. * denotes our re-implementation. The metircs DIVER used follows [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- The **weight(stage-1)**, **data pkl** and**kmenas**  of DIVER in Bench2Drive:[**DIVER**](https://pan.baidu.com/s/1qBVdpXUohfveU8au9ShAyg?pwd=u36f)
<table border="1">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="1">Open-loop Metric</th>
      <th colspan="4">Closed-loop Metric</th>
    </tr>
    <tr>
      <th>Avg. L2 ↓</th>
      <th>DS ↑</th>
      <th>SR(%) ↑</th>
      <th>Effi ↑</th>
      <th>Comf ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VAD</td>
      <td>0.91</td>
      <td>42.35</td>
      <td>15.00</td>
      <td>157.94</td>
      <td>46.01</td>
    </tr>
    <tr>
      <td>VAD mmt*</td>
      <td>0.89</td>
      <td>42.87</td>
      <td>15.91</td>
      <td>158.12</td>
      <td>47.22</td>
    </tr>
    <tr>
      <td>Our DIVER (Euclidean)</td>
      <td>0.84</td>
      <td>46.12</td>
      <td>17.45</td>
      <td>173.35</td>
      <td>50.98</td>
    </tr>
    <tr>
      <td>Our DIVER</td>
      <td>0.85</td>
      <td>45.35</td>
      <td>17.44</td>
      <td>162.09</td>
      <td>49.34</td>
    </tr>
    <tr>
      <td>SparcDrive*</td>
      <td>0.87</td>
      <td>44.54</td>
      <td>16.71</td>
      <td>170.21</td>
      <td>48.63</td>
    </tr>
    <tr>
      <td>Our DIVER (Euclidean)</td>
      <td>0.84</td>
      <td>46.12</td>
      <td>17.45</td>
      <td>173.35</td>
      <td>50.98</td>
    </tr>
    <tr>
      <td>Our DIVER</td>
      <td>0.82</td>
      <td>47.91</td>
      <td>18.11</td>
      <td>174.91</td>
      <td>51.20</td>
    </tr>
  </tbody>
</table>

### Close_loop Vis

<p align="left">
  <img src = "./close_loop/video_show.gif" width="60%">
</p>

### Robustness evaluation

- Robustness analysis on [nuScenes-C](https://github.com/thu-ml/3D_Corruptions_AD)

<table border="1">
  <thead>
    <tr>
      <th rowspan="2">Setting</th>
      <th rowspan="2">Method</th>
      <th colspan="2">Detection</th>
      <th colspan="1">Tracking</th>
      <th colspan="1">Mapping</th>
      <th colspan="1">Motion</th>
      <th colspan="3">Planning</th>
    </tr>
    <tr>
      <th>mAP ↑</th>
      <th>NDS ↑</th>
      <th>AMOTA ↑</th>
      <th>mAP ↑</th>
      <th>mADE ↓</th>
      <th>L2 ↓</th>
      <th>Col. ↓</th>
      <th>TPC ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Clean</td>
      <td>SparseDrive</td>
      <td>0.418</td>
      <td>0.525</td>
      <td>0.386</td>
      <td>55.1</td>
      <td>0.62</td>
      <td>0.61</td>
      <td>0.08</td>
      <td>0.57</td>
    </tr>
    <tr>
      <td>Clean</td>
      <td>Our DIVER</td>
      <td>0.423</td>
      <td>0.531</td>
      <td>0.391</td>
      <td>55.9</td>
      <td>0.61</td>
      <td>0.60</td>
      <td>0.09</td>
      <td>0.54</td>
    </tr>
    <tr>
      <td>Snow</td>
      <td>SparseDrive</td>
      <td>0.091</td>
      <td>0.111</td>
      <td>0.102</td>
      <td>16.0</td>
      <td>0.98</td>
      <td>0.88</td>
      <td>0.32</td>
      <td>0.82</td>
    </tr>
    <tr>
      <td>Snow</td>
      <td>Our DIVER</td>
      <td>0.154</td>
      <td>0.173</td>
      <td>0.166</td>
      <td>20.9</td>
      <td>0.76</td>
      <td>0.73</td>
      <td>0.16</td>
      <td>0.68</td>
    </tr>
    <tr>
      <td>Fog</td>
      <td>SparseDrive</td>
      <td>0.141</td>
      <td>0.159</td>
      <td>0.154</td>
      <td>18.8</td>
      <td>0.91</td>
      <td>0.86</td>
      <td>0.41</td>
      <td>0.80</td>
    </tr>
    <tr>
      <td>Fog</td>
      <td>Our DIVER</td>
      <td>0.197</td>
      <td>0.197</td>
      <td>0.206</td>
      <td>24.9</td>
      <td>0.73</td>
      <td>0.71</td>
      <td>0.18</td>
      <td>0.67</td>
    </tr>
    <tr>
      <td>Rain</td>
      <td>SparseDrive</td>
      <td>0.128</td>
      <td>0.140</td>
      <td>0.193</td>
      <td>19.4</td>
      <td>0.97</td>
      <td>0.93</td>
      <td>0.46</td>
      <td>0.92</td>
    </tr>
    <tr>
      <td>Rain</td>
      <td>Our DIVER</td>
      <td>0.207</td>
      <td>0.213</td>
      <td>0.266</td>
      <td>25.2</td>
      <td>0.76</td>
      <td>0.71</td>
      <td>0.21</td>
      <td>0.71</td>
    </tr>
  </tbody>
</table>


## Trajectory Prediction Consistency (TPC) metric
To evaluate the planning stability of DIVER, we propose a new [Trajectory Prediction Consistency (TPC) metric](/open_loop/projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval_roboAD_6s.py) to measure consistency between predicted and historical trajectories.

## How to generate a 6s nuScenes trajectory dataset？
```
python tools/data_converter/nuscenes_converter_6s.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0
```
## Quick Start
[Quick Start for Open_loop](open_loop/docs/quick_start.md)

[Quick start for Close_loop](close_loop/quick_start.md)

## Citation
If you find DIVER is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```
@article{song2025DIVER,
      title={Don't Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving}, 
      author={Ziying Song and Caiyan Jia and Lin Liu and Hongyu Pan and Yongchang Zhang and Junming Wang and Xingyu Zhang and Shaoqing Xu and Lei Yang and Yadan Luo},
      year={2025},
      eprint={2503.03125},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.03125}, 
}
```
## Acknowledgement
- [SparseDrive](https://github.com/swc-17/SparseDrive)
- [UniAD](https://github.com/OpenDriveLab/UniAD) 
- [VAD](https://github.com/hustvl/VAD)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)

