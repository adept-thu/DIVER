import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

K = 6

fp = './data/infos/b2d_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(data)
navi_trajs = [[], [], [], [], [], []]


def get_ego_trajs(idx,sample_rate,past_frames,future_frames,data_infos):
        #import pdb;pdb.set_trace()
        # idx = 128386
        adj_idx_list = range(idx-past_frames*sample_rate,idx+(future_frames+1)*sample_rate,sample_rate)
        cur_frame = data_infos[idx]
        full_adj_track = np.zeros((past_frames+future_frames+1,2))
        full_adj_adj_mask = np.zeros(past_frames+future_frames+1)
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx <0 or adj_idx>=len(data_infos):
                print("True")
                break
            adj_frame = data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame ['folder']:
                break
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2,3]
            full_adj_track[j,0:2] = xy
            full_adj_adj_mask[j] = 1
        offset_track = full_adj_track[1:] - full_adj_track[:-1]
        for j in range(past_frames-1,-1,-1):
            if full_adj_adj_mask[j] == 0:
                offset_track[j] = offset_track[j+1]
        for j in range(past_frames,past_frames+future_frames,1):

            if full_adj_adj_mask[j+1] == 0 :
                offset_track[j] = 0
        #command = self.command2hot(cur_frame['command_near'])
        return offset_track[past_frames:].copy()


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

sum_turn = 0
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    #import pdb;pdb.set_trace()
    plan_traj = get_ego_trajs(idx, 5, 6, 6, data_infos)
    plan_traj = plan_traj.cumsum(axis=-2)
    #plan_mask = info['gt_ego_fut_masks']
    #import pdb;pdb.set_trace()
    cmd = info['command_near']#.astype(np.int32)
    steer = info['steer']
    #if cmd == 1 or cmd == 2 or cmd == 5 or cmd == 6:
    #    print(cmd)
    #    sum_turn = sum_turn + 1
    #cmd = cmd.argmax(axis=-1)
    #if not plan_mask.sum() == 6:
    #    continue
    navi_trajs[cmd-1].append(plan_traj)
#import pdb;pdb.set_trace()
clusters = []
#import pdb;pdb.set_trace()
for trajs in navi_trajs:
    trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
    cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
    cluster = cluster.reshape(-1, 6, 2)
    clusters.append(cluster)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])
plt.savefig(f'vis/kmeans/plan_{K}', bbox_inches='tight')
plt.close()

clusters = np.stack(clusters, axis=0)
np.save(f'data/kmeans/kmeans_plan_{K}.npy', clusters)