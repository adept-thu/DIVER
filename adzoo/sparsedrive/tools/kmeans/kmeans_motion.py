import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

CLASSES = [
    'car',
    'van',
    'truck',
    'bicycle',
    'traffic_sign',
    'traffic_cone',
    'traffic_light',
    'pedestrian',
    'others',
]

NameMapping = {
    #=================vehicle=================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.audi.etron": 'car',
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": 'car',
    # bus
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    #=========================================

    #=================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',
    
    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    #=========================================

    #===================Construction===========
    "static.prop.warningconstruction" : 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",

    #===================Construction===========
    "static.prop.constructioncone": 'traffic_cone',

    #=================pedestrian==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0003": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0010": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0015": 'pedestrian',
    "walker.pedestrian.0016": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0021": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0027": 'pedestrian',
    "walker.pedestrian.0030": 'pedestrian',
    "walker.pedestrian.0031": 'pedestrian',
    "walker.pedestrian.0032": 'pedestrian',
    "walker.pedestrian.0034": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0042": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',

    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}


def get_fut_agent(idx, sample_rate, frames, data_infos):
        adj_idx_list = range(idx,idx+(frames+1)*sample_rate,sample_rate)
        cur_frame = data_infos[idx]
        cur_boxes = cur_frame['gt_boxes'].copy()
        box_ids = cur_frame['gt_ids']

        future_track = np.zeros((len(box_ids),frames+1,2))
        future_mask = np.zeros((len(box_ids),frames+1))
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            box_id = box_ids[i]
            cur_box2lidar = world2lidar_lidar_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2,3]
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx < 0 or adj_idx >= len(data_infos):
                    break
                adj_frame = data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame ['folder']:
                    break
                if len(np.where(adj_frame['gt_ids']==box_id)[0])==0:
                    break
                assert len(np.where(adj_frame['gt_ids']==box_id)[0]) == 1 , np.where(adj_frame['gt_ids']==box_id)[0]
                adj_idx = np.where(adj_frame['gt_ids']==box_id)[0][0]
                adj_box2lidar = world2lidar_lidar_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2,3]
                if j > 0:
                    last_xy = future_track[i,j-1,:]
                    distance = np.linalg.norm(last_xy - adj_xy)
                    if distance > 10:
                        break
                future_track[i,j,:] = adj_xy
                future_mask[i,j] = 1

        future_track_offset = future_track[:,1:,:] - future_track[:,:-1,:]
        future_mask_offset = future_mask[:,1:]
        future_track_offset[future_mask_offset==0] = 0

        return future_track_offset.astype(np.float32), future_mask_offset.astype(np.float32)


def lidar2agent(trajs_offset, boxes):
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float32)
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)
    trajs = trajs_offset.cumsum(axis=1)
    yaws = - boxes[:, 6]
    rot_sin = np.sin(yaws)
    rot_cos = np.cos(yaws)
    rot_mat_T = np.stack(
        [
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos]),
        ]
    )
    trajs_new = np.einsum('aij,jka->aik', trajs, rot_mat_T)
    trajs_new = trajs_new[:, 1:]
    return trajs_new

K = 6
DIS_THRESH = 65

fp = 'data/infos_sparsedrive/b2d_infos_train.pkl'
data = mmcv.load(fp)
#import pdb;pdb.set_trace()
data_infos = data #sorted(data, key=lambda e: e["timestamp"])

intention = dict()

for i in range(len(CLASSES)):
    intention[i] = []
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    boxes = info['gt_boxes']
    names = info['gt_names']
    for name_i in range(len(names)):
        names[name_i] = NameMapping[names[name_i]]
    #import pdb;pdb.set_trace()
    gt_agent_fut_trajs, gt_agent_fut_masks = get_fut_agent(idx, 5, 6, data_infos)
    fut_masks = gt_agent_fut_masks
    trajs = gt_agent_fut_trajs
    x = gt_agent_fut_trajs[:,0,0]
    y = gt_agent_fut_trajs[:,0,1]
    result = np.sqrt(x**2 + y**2)
    #import pdb;pdb.set_trace()
    velos = result
    labels = []
    for cat in names:
        if cat in CLASSES:
            labels.append(CLASSES.index(cat))
        else:
            labels.append(-1)
    labels = np.array(labels)
    if len(boxes) == 0:
        continue
    #import pdb;pdb.set_trace()    
    for i in range(len(CLASSES)):
        #import pdb;pdb.set_trace()
        cls_mask = (labels == i)
        box_cls = boxes[cls_mask]
        fut_masks_cls = fut_masks[cls_mask]
        trajs_cls = trajs[cls_mask]
        velos_cls = velos[cls_mask]

        distance = np.linalg.norm(box_cls[:, :2], axis=1)
        mask = np.logical_and(
            fut_masks_cls.sum(axis=1) == 6,
            distance < DIS_THRESH,
        )
        trajs_cls = trajs_cls[mask]
        box_cls = box_cls[mask]
        velos_cls = velos_cls[mask]

        trajs_agent = lidar2agent(trajs_cls, box_cls)
        if trajs_agent.shape[0] == 0:
            continue
        intention[i].append(trajs_agent)
#import pdb;pdb.set_trace()
clusters = []
for i in range(len(CLASSES)):
    intention_cls = np.concatenate(intention[i], axis=0).reshape(-1, 12)
    if intention_cls.shape[0] < K:
        continue
    cluster = KMeans(n_clusters=K).fit(intention_cls).cluster_centers_
    cluster = cluster.reshape(-1, 6, 2)
    clusters.append(cluster)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])
    plt.savefig(f'vis/kmeans/motion_intention_{CLASSES[i]}_{K}', bbox_inches='tight')
    plt.close()

clusters = np.stack(clusters, axis=0)
np.save(f'data/kmeans/kmeans_motion_{K}.npy', clusters)