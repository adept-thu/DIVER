import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import mmcv

K = 6

fp = './data/infos/b2d_infos_val.pkl'
data = mmcv.load(fp)
data_infos = list(data)
navi_trajs = [[], [], [], [], [], []]

def compute_adaptive_eps(data, min_samples=5, percentile=53):
    """自动计算DBSCAN的eps参数"""
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    k_distances = distances[:, -1]
    return np.percentile(k_distances, percentile)

def cluster_trajectories(traj_array, min_samples=5, percentile=53):
    """
    对轨迹数据进行自适应聚类，返回按簇大小排序的索引列表
    
    参数：
    traj_array: 形状为[N,6,2]的轨迹数组
    min_samples: DBSCAN的最小样本数参数
    percentile: 用于自动计算eps的百分位数
    
    返回：
    按簇大小降序排列的索引列表，每个元素为一个簇的索引数组
    """
    # 数据预处理：展平轨迹数据
    n_trajectories = traj_array.shape[0]
    flattened_data = traj_array.reshape(n_trajectories, -1)
    
    # 自动计算eps参数
    eps = compute_adaptive_eps(flattened_data, min_samples, percentile)
    
    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(flattened_data)
    
    # 统计簇信息并排序（包含噪声簇）
    unique_labels, counts = np.unique(labels, return_counts=True)
    clusters_info = sorted(zip(unique_labels, counts), 
                         key=lambda x: x[1], reverse=True)
    
    # 创建簇索引字典
    cluster_indices = {}
    for idx, label in enumerate(labels):
        cluster_indices.setdefault(label, []).append(idx)
    
    # 生成排序后的簇索引列表
    sorted_clusters = [cluster_indices[label] for label, _ in clusters_info]
    
    return sorted_clusters



def cluster_trajectories1(traj_array, min_samples=5, percentile=65):
    """
    对轨迹数据进行自适应聚类，返回按簇大小排序的索引列表
    
    参数：
    traj_array: 形状为[N,6,2]的轨迹数组
    min_samples: DBSCAN的最小样本数参数
    percentile: 用于自动计算eps的百分位数
    
    返回：
    按簇大小降序排列的索引列表，每个元素为一个簇的索引数组
    """
    # 数据预处理：展平轨迹数据
    n_trajectories = traj_array.shape[0]
    flattened_data = traj_array.reshape(n_trajectories, -1)
    
    # 自动计算eps参数
    eps = compute_adaptive_eps(flattened_data, min_samples, percentile)
    
    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(flattened_data)
    
    # 统计簇信息并排序（包含噪声簇）
    unique_labels, counts = np.unique(labels, return_counts=True)
    clusters_info = sorted(zip(unique_labels, counts), 
                         key=lambda x: x[1], reverse=True)
    
    # 创建簇索引字典
    cluster_indices = {}
    for idx, label in enumerate(labels):
        cluster_indices.setdefault(label, []).append(idx)
    
    # 生成排序后的簇索引列表
    sorted_clusters = [cluster_indices[label] for label, _ in clusters_info]
    
    return sorted_clusters, labels

def plot_sample_trajectories(traj_array, sorted_clusters, labels):
    """
    从每个簇中随机抽取一个样本并绘制其轨迹
    
    参数：
    traj_array: 形状为[N,6,2]的轨迹数组
    sorted_clusters: 按簇大小排序的索引列表
    labels: 每个样本的簇标签
    """
    plt.figure(figsize=(16, 12),dpi=600)
    
    # 为每个簇分配一个颜色
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for cluster_label, color in zip(unique_labels, colors):
        if cluster_label == -1 or cluster_label == 0 or cluster_label == 1:
            continue  # 跳过噪声簇
        
        # 获取当前簇的所有索引
        cluster_indices = [idx for idx, label in enumerate(labels) if label == cluster_label]
        
        # 随机抽取一个样本
        sample_idx = np.random.choice(cluster_indices)
        sample_traj = traj_array[sample_idx]
        
        # 绘制轨迹
        x = sample_traj[:, 0]
        y = sample_traj[:, 1]
        plt.plot(x, y, 
                markersize=10, 
                linewidth=2.5,
                marker='o', 
                color=color
        )#, label=f'Cluster {cluster_label}')
    
    plt.title("Sampled Trajectories from Each Cluster")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    #plt.grid(True)
    #plt.xticks(np.arange(-8, 6.5, 0.3918))  # x轴从0到10，步长0.5
    #plt.yticks(np.arange(-1, 36, 1))  # y轴从-1到1，步长0.2
    plt.savefig("/data/songziying/workspace/SparseDriveb2d/vis1.png", bbox_inches='tight', dpi=600)
    #plt.show()


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
fs_trajs = []
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    #import pdb;pdb.set_trace()
    plan_traj = get_ego_trajs(idx, 5, 6, 6, data_infos)
    plan_traj = plan_traj.cumsum(axis=-2)
    #plan_mask = info['gt_ego_fut_masks']
    #import pdb;pdb.set_trace()
    cmd = info['command_near']#.astype(np.int32)
    steer = info['steer']
    fs_trajs.append(plan_traj)
#import pdb;pdb.set_trace()
fs_trajs = np.stack(fs_trajs)
clustered_indices = cluster_trajectories(fs_trajs)
sorted_clusters, labels = cluster_trajectories1(fs_trajs)
plot_sample_trajectories(fs_trajs, sorted_clusters, labels)
import pdb;pdb.set_trace()
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