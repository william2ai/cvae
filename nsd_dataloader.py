import os
import torch
from torch.utils.data import Dataset
import time
from tqdm.auto import tqdm
import concurrent
import numpy as np

class NSDDataset(Dataset):
    def __init__(self, npy_root, mask, session_list, load_in_memory=True):
        """
        npy_root: str, 存储 .npy 文件路径
        mask: np.ndarray 3D bool 数组
        session_list: list of int
        load_in_memory: 是否一次性加载到内存
        """
        self.npy_root = npy_root
        self.mask = mask.astype(bool)
        self.session_list = session_list
        self.load_in_memory = load_in_memory

        # 只保留 mask 内体素数量
        self.mask_idx = np.where(self.mask.flatten())[0]   #记录了mask=true的voxel索引位置
        self.n_voxels = len(self.mask_idx)

        self.data_fmri = []
        self.data_clip = []
        self.session_lengths = []

        for sess in session_list :
            fmri_path = os.path.join(npy_root, f'session{sess:02d}_fmri_norm.npy')
            clip_path = os.path.join(npy_root, f'session{sess:02d}_clip_stim.npy')

            if load_in_memory:
                fmri_data = np.load(fmri_path).astype(np.float32)  # (T, X, Y, Z)
                # 只保留 mask 内的体素
                fmri_data_masked = fmri_data.reshape(fmri_data.shape[0], -1)[:, self.mask_idx]
                clip_data = np.load(clip_path).astype(np.float32)
            else:
                # 使用 mmap 也可以按 mask 索引
                fmri_data = np.load(fmri_path, mmap_mode='r')
                fmri_data_masked = fmri_data.reshape(fmri_data.shape[0], -1)[:, self.mask_idx]
                clip_data = np.load(clip_path, mmap_mode='r')

            self.data_fmri.append(fmri_data_masked)
            self.data_clip.append(clip_data)
            self.session_lengths.append(fmri_data_masked.shape[0])

        self.cumulative_lengths = np.cumsum([0] + self.session_lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        session_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        idx_in_session = idx - self.cumulative_lengths[session_idx]

        fmri_data = self.data_fmri[session_idx]
        clip_data = self.data_clip[session_idx]

        fmri_vec = fmri_data[idx_in_session]  # shape: [n_voxels_in_mask]
        average_beta = fmri_vec.mean(dtype=np.float32)

        stim_clip = clip_data[idx_in_session]

        # 转 torch tensor
        avg_beta_tensor = torch.tensor(average_beta, dtype=torch.float32)
        stim_clip_tensor = torch.from_numpy(stim_clip).float()

        return avg_beta_tensor, stim_clip_tensor

class NSDVoxelDataset(Dataset):
    def __init__(self, npy_root, mask, session_list, load_in_memory=True):
        self.mask = mask.astype(bool)
        self.mask_idx = np.where(self.mask.flatten())[0]
        self.voxel_positions = torch.from_numpy(np.array(np.where(self.mask)).T).long()
        self.n_voxels = len(self.mask_idx)

        self.data_fmri = []
        self.data_clip = []
        self.session_lengths = []

        for sess in session_list:
            fmri_path = os.path.join(npy_root, f'session{sess:02d}_fmri_norm.npy')
            clip_path = os.path.join(npy_root, f'session{sess:02d}_clip_stim.npy')

            fmri_data = np.load(fmri_path, mmap_mode=None if load_in_memory else 'r')
            fmri_data_masked = fmri_data.reshape(fmri_data.shape[0], -1)[:, self.mask_idx]

            clip_data = np.load(clip_path, mmap_mode=None if load_in_memory else 'r')

            self.data_fmri.append(torch.from_numpy(fmri_data_masked.astype(np.float32)))
            self.data_clip.append(torch.from_numpy(clip_data.astype(np.float32)))
            self.session_lengths.append(fmri_data_masked.shape[0])

        self.index_map = [(s, t, v)
                          for s, T in enumerate(self.session_lengths)
                          for t in range(T)
                          for v in range(self.n_voxels)]

        print(f"NSDVoxelDataset: {len(self.index_map)} samples "
              f"(sessions={len(session_list)}, voxels={self.n_voxels})")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        s, t, v = self.index_map[idx]
        beta = self.data_fmri[s][t, v]
        stim = self.data_clip[s][t]
        pos  = self.voxel_positions[v]
        return pos, beta, stim


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # mask = np.zeros((81, 104, 83))
    # mask[40, 70, 40] = 1
    # nsd_dataset = NSDDataset(
    #     npy_root='/cephfs/shared/nsd/session_npy',
    #     mask=mask,
    #     session_list=[1,2,3,4],
    # )
    # dataloader = DataLoader(nsd_dataset, batch_size=4, shuffle=False, num_workers=0)

    # for i, (masked_fmri, beta, stim_clip) in tqdm(enumerate(dataloader)):
    #     # print(f"Batch {i}:")
    #     # print(f"  masked_fmri shape: {masked_fmri.shape}")
    #     print(f"  beta shape: {beta}")
    #     # print(f"  stim_img shape: {stim_clip.shape}")
    #     break
    pass