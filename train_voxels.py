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

from model_voxels import CVAEWithRouter
from nsd_dataloader import NSDVoxelDataset

def create_roi_mask(roi_file, roi_label=1):
    roi = nib.load(roi_file).get_fdata()
    mask = (roi == roi_label).astype(int)
    return mask

def compute_activation(cfg, beta, eps=1e-4):
    return cfg.temp * torch.sign(beta) / (beta.abs()  + eps)

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
            print(f'step {i} loss:{epoch_loss / i} recon loss:{epoch_recon / i} kl loss:{epoch_kl / i}')
    n_batches = len(dataloader)
    return epoch_loss / n_batches, epoch_recon / n_batches, epoch_kl / n_batches

@torch.no_grad()
def validate(model, dataloader, cfg):
    model.eval()
    val_loss, val_recon, val_kl =0., 0., 0.
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

def main(cfg, subregion_name):
    os.makedirs("checkpoints/VOXELs-ckpts", exist_ok=True)
    print(f"=== Training {cfg.roi} voxels / {subregion_name} ===")
    mask = create_roi_mask(f'./ROI/floc-{cfg.roi}.nii.gz', roi_label=cfg.roi_label)
    train_dataset = NSDVoxelDataset('/cephfs/shared/nsd/session_npy', mask, session_list=list(range(1, 36)))
    test_dataset = NSDVoxelDataset('/cephfs/shared/nsd/session_npy', mask, session_list=list(range(36, 41)))
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CVAEWithRouter(input_dim=1024, pos_dim=3, pos_emb_dim=cfg.pos_emb_dim, latent_dim=cfg.z_dim, hidden_dim=512, K=cfg.K).to(cfg.device)
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
            ckpt_file = f'checkpoints/VOXELs-ckpts/{subregion_name}_voxels_K{cfg.K}_zdim{cfg.z_dim}_posemb{cfg.pos_emb_dim}_RW{cfg.recon_weight}_KLW{cfg.KLD_weight}_L2W{cfg.l2_weight}.pt'
            torch.save(model.state_dict(), ckpt_file)
            print(f'Saved best model: {ckpt_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", type=str, required=True, help="大脑区名称: faces/words/places/bodies")
    parser.add_argument("--roi_label", type=int, required=True, help="小脑区 index，从1开始")
    args = parser.parse_args()

    with open("config_roi.json", "r") as f:
        ROI_JSON = json.load(f)
    subregion_list = [r for r in ROI_JSON[args.roi] if r["index"] == args.roi_label]
    if len(subregion_list) != 1:
        raise ValueError(f"无法找到 {args.roi} 的 index {args.roi_label} 小脑区")
    subregion_name = subregion_list[0]["abbreviation"]

    @dataclass
    class Config:
        batch_size: int = 128
        epoch: int = 1 # 3
        lr: float = 1e-5 # 1e-5
        pos_emb_dim: int = 64
        z_dim: int = 256
        recon_weight: float = 1.0
        KLD_weight: float = 1.0
        l2_weight: float = 400.0
        temp: float = 1.0
        K: int = 4
        roi: str = args.roi
        roi_label: int = args.roi_label
        device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    cfg = Config()
    main(cfg, subregion_name)