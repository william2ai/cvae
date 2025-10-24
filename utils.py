import numpy as np
import os
import torch
from torch.utils.data import Dataset
import numpy as np
def create_voxel_mask(shape, center, radius):
    """
    创建一个 0/1 的掩膜，包含中心 voxel 周围 radius 的所有 voxel。
    shape: 原始 fMRI 数据的形状，例如 (81, 104, 83)
    center: 三维中心坐标 [x, y, z]
    radius: 半径（1 表示取中心 ±1，即 3x3x3 的体素块）
    """
    mask = np.zeros(shape, dtype=np.uint8)
    x0, y0, z0 = center
    x1, y1, z1 = [max(0, x0 - radius), max(0, y0 - radius), max(0, z0 - radius)]
    x2, y2, z2 = [min(shape[0], x0 + radius + 1),
                  min(shape[1], y0 + radius + 1),
                  min(shape[2], z0 + radius + 1)]
    mask[x1:x2, y1:y2, z1:z2] = 1
    return mask

class NSDDataset_PIC(Dataset):
    def __init__(self, npy_root, mask, session_list):
        """
        npy_root: str, 存储 .npy 文件的路径
        mask: np.ndarray 三维bool或0/1数组，形状与fMRI体积一致 (X,Y,Z)
        session_list: list of int, 需要加载的session编号
        transform: 对stimulus图片的transform，默认None
        """
        self.npy_root = npy_root
        self.mask = mask.astype(bool)  # 确保是bool类型，方便乘法和索引
        self.session_list = session_list

        self.session_lengths = []
        for sess in session_list:
            fmri_path = os.path.join(npy_root, f'session{sess:02d}_fmri.npy')
            data = np.load(fmri_path, mmap_mode='r')   #只读取一小块需要的数据，避免读入全部数据到内存中
            self.session_lengths.append(data.shape[0])

        self.cumulative_lengths = np.cumsum([0] + self.session_lengths)     #np.cumsum([0, 100, 120, 80])  →  [0, 100, 220, 300]

    def __len__(self):
        return self.cumulative_lengths[-1]    # total fMRI data length

    def __getitem__(self, idx):
        session_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1 
        #np.searchsorted([0, 100, 220, 300], 101, side='right') --> 2 (会在一个有序数组里找到value应该插入的位置)
        #根据全局样本索引找到对应的 session 和在该 session 中的样本位置
        idx_in_session = idx - self.cumulative_lengths[session_idx]
        sess = self.session_list[session_idx]

        fmri_path = os.path.join(self.npy_root, f'session{sess:02d}_fmri_norm.npy')
        # stim_path = os.path.join(self.npy_root, f'session{sess:02d}_stim.npy')
        stim_clip_path = os.path.join(self.npy_root, f'session{sess:02d}_stim.npy')

        fmri_data = np.load(fmri_path, mmap_mode='r')
        # stim_data = np.load(stim_path, mmap_mode='r')
        stim_clip_data = np.load(stim_clip_path, mmap_mode='r')

        fmri_vol = fmri_data[idx_in_session]          # (X,Y,Z)
        masked_fmri = fmri_vol * self.mask            # (X,Y,Z)

        average_beta = fmri_vol[self.mask].mean(dtype=np.float32)   #当前 fMRI 体积中感兴趣脑区的平均活动强度

        # stim_img = stim_data[idx_in_session]    
        # stim_img = torch.from_numpy(stim_img.copy()).float().squeeze(0) / 255.0
        stim_clip = stim_clip_data[idx_in_session]
        stim_clip = torch.from_numpy(stim_clip.copy()).float()
        masked_fmri_tensor = torch.from_numpy(masked_fmri.copy()).float()
        average_beta_tensor = torch.tensor(average_beta, dtype=torch.float32)
        return masked_fmri_tensor, average_beta_tensor, stim_clip