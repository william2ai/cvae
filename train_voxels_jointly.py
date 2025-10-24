import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm.auto import tqdm
import nibabel as nib
import numpy as np
import argparse
import os
import json
from typing import List, Tuple

from model_voxels import CVAEWithRouter
from nsd_dataloader import NSDVoxelDataset

def create_roi_mask_multi(roi_specs: List[Tuple[str, int]]) -> np.ndarray:
    """
    roi_specs: [("faces", 1), ("bodies", 2), ...]
    将多个 floc-{roi}.nii.gz 的指定 label 合并（并集）。
    """
    combined = None
    for roi_name, roi_label in roi_specs:
        roi_file = f'./ROI/floc-{roi_name}.nii.gz'
        roi = nib.load(roi_file).get_fdata()
        mask = (roi == roi_label).astype(np.int32)
        combined = mask if combined is None else (combined | mask)
    return combined

def compute_activation(cfg, beta, eps=1e-4):
    return cfg.temp * torch.sign(beta) / (beta.abs() + eps)

def train_one_epoch(model, dataloader, optimizer, cfg):
    model.train()
    epoch_loss, epoch_recon, epoch_kl = 0., 0., 0.
    for i, (pos, beta, img_clip) in tqdm(enumerate(dataloader), desc='Training'):
        pos, beta, img_clip = pos.to(cfg.device, dtype=torch.float32), beta.to(cfg.device), img_clip.to(cfg.device)
        activation = compute_activation(cfg, beta)
        total_loss, recon_loss, kl_loss, recon, mu, logvar, p_k = model.forward(
            x=img_clip, pos_idx=pos,
            activation=activation, recon_weight=cfg.recon_weight,
            KLD_weight=cfg.KLD_weight, router_l2_weight=cfg.l2_weight
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()
        if i % 2000 == 0 and i != 0:
            print(f'step {i} loss:{epoch_loss / i:.4f} recon:{epoch_recon / i:.4f} kl:{epoch_kl / i:.4f}')
    n_batches = len(dataloader)
    return epoch_loss / n_batches, epoch_recon / n_batches, epoch_kl / n_batches

@torch.no_grad()
def validate(model, dataloader, cfg):
    model.eval()
    val_loss, val_recon, val_kl = 0., 0., 0.
    for pos, beta, img_clip in tqdm(dataloader, desc='Validation'):
        pos, beta, img_clip = pos.to(cfg.device, dtype=torch.float32), beta.to(cfg.device), img_clip.to(cfg.device)
        activation = compute_activation(cfg, beta)
        total_loss, recon_loss, kl_loss, recon, mu, logvar, p_k = model(
            x=img_clip, pos_idx=pos,
            activation=activation, recon_weight=cfg.recon_weight,
            KLD_weight=cfg.KLD_weight, router_l2_weight=cfg.l2_weight
        )
        val_loss += total_loss.item()
        val_recon += recon_loss.item()
        val_kl += kl_loss.item()
    n_batches = len(dataloader)
    return val_loss / n_batches, val_recon / n_batches, val_kl / n_batches

def main(cfg, combo_name, roi_specs):
    os.makedirs("checkpoints/VOXELs-joint-ckpts", exist_ok=True)
    print(f"=== Training voxels for {combo_name} ===")

    # 合并多个 ROI 掩码
    mask = create_roi_mask_multi(roi_specs)

    train_dataset = NSDVoxelDataset('/cephfs/shared/nsd/session_npy', mask, session_list=list(range(1, 36)))
    test_dataset  = NSDVoxelDataset('/cephfs/shared/nsd/session_npy', mask, session_list=list(range(36, 41)))
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CVAEWithRouter(input_dim=1024, pos_dim=3, pos_emb_dim=cfg.pos_emb_dim,
                           latent_dim=cfg.z_dim, hidden_dim=512, K=cfg.K).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch, eta_min=1e-6)
    best_val_loss = float('inf')

    for epoch in range(cfg.epoch):
        train_loss, train_recon, train_kl = train_one_epoch(model, train_loader, optimizer, cfg)
        val_loss, val_recon, val_kl = validate(model, test_loader, cfg)
        # scheduler.step()
        print(f"Epoch {epoch+1}/{cfg.epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_file = (f'checkpoints/VOXELs-joint-ckpts/{combo_name}_voxels'
                         f'_K{cfg.K}_zdim{cfg.z_dim}_posemb{cfg.pos_emb_dim}'
                         f'_RW{cfg.recon_weight}_KLW{cfg.KLD_weight}_L2W{cfg.l2_weight}.pt')
            torch.save(model.state_dict(), ckpt_file)
            print(f'Saved best model: {ckpt_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 新增：一次传多个 ROI，小脑区 index 从1开始。示例：faces:1 bodies:2
    parser.add_argument("--regions", type=str, nargs='+', required=False,
                        help="多个脑区参数，格式 roi:label，例如 faces:1 bodies:2")
    # 兼容旧参数（若只传单个 ROI）
    parser.add_argument("--roi", type=str, help="单个大脑区名称: faces/words/places/bodies")
    parser.add_argument("--roi_label", type=int, help="单个小脑区 index，从1开始")
    args = parser.parse_args()

    with open("config_roi.json", "r") as f:
        ROI_JSON = json.load(f)

    # 解析 regions
    roi_specs: List[Tuple[str, int]] = []
    if args.regions:
        for token in args.regions:
            try:
                roi_name, label_str = token.split(":")
                roi_name = roi_name.strip()
                roi_label = int(label_str)
            except Exception:
                raise ValueError(f"--regions 参数格式错误：{token}（应为 roi:label，例如 faces:1）")
            roi_specs.append((roi_name, roi_label))
    else:
        # 回退到旧用法
        if not (args.roi and args.roi_label):
            raise ValueError("请使用 --regions faces:1 bodies:2 ...，或使用旧参数 --roi 与 --roi_label。")
        roi_specs.append((args.roi, args.roi_label))

    # 生成用于日志/ckpt 的组合名称，如 "FFA-1+EBA"
    abbrevs = []
    for roi_name, roi_label in roi_specs:
        if roi_name not in ROI_JSON:
            raise ValueError(f"ROI '{roi_name}' 不在 config_roi.json 中。")
        candidates = [r for r in ROI_JSON[roi_name] if r.get("index") == roi_label]
        if len(candidates) != 1:
            raise ValueError(f"无法找到 {roi_name} 的 index {roi_label} 小脑区")
        abbrevs.append(candidates[0].get("abbreviation", f"{roi_name}-{roi_label}"))
    combo_name = "+".join(abbrevs)

    @dataclass
    class Config:
        batch_size: int = 128
        epoch: int = 5   # 3
        lr: float = 1e-5
        pos_emb_dim: int = 64
        z_dim: int = 256
        recon_weight: float = 1.0
        KLD_weight: float = 1.0
        l2_weight: float = 400.0
        temp: float = 1.0
        K: int = 4
        device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    cfg = Config()
    main(cfg, combo_name, roi_specs)
