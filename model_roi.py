import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=256, hidden_dim=512):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h1 = F.gelu(self.fc_dec1(z))
        h = F.gelu(self.fc_dec2(h1))
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class Router(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        h1 = F.gelu(self.fc1(x))
        logits = self.fc2(h1)
        probs = F.softmax(logits, dim=-1)
        return probs


class VAEWithRouter(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=256, hidden_dim=512, K=4):
        super().__init__()
        self.vae = VAE(input_dim, latent_dim, hidden_dim)
        self.router = Router(input_dim, hidden_dim, K)
        self.K = K
        self.latent_dim = latent_dim

    def forward(self, x, activation, recon_weight=1.0, KLD_weight=1.0, router_l2_weight=4.0):
        B = x.size(0)
        device = x.device
        latent_dim = self.latent_dim

        # --- VAE 前向 ---
        recon_x, mu, logvar = self.vae(x)

        # --- Router 前向 ---
        p_k = self.router(x)  # (B, K)

        # --- KL loss ---
        # prior_mu 按新规则 e[i] 构造
        prior_mu = torch.zeros(B, self.K, latent_dim, device=device)
        activation_sign = torch.sign(activation).view(B, 1)  # [B,1]

        for k in range(self.K):
            prior_mu[:, k, :] = activation_sign * (1.0 + k)

        prior_var = activation.abs().view(B,1,1).expand(B, self.K, latent_dim) + 1e-5

        var = torch.exp(logvar).unsqueeze(1)       # (B,1,latent_dim)
        mu_expand = mu.unsqueeze(1)                # (B,1,latent_dim)
        logvar_expand = logvar.unsqueeze(1)        # (B,1,latent_dim)

        kl_dim = 0.5 * (torch.log(prior_var) - logvar_expand + (var + (mu_expand - prior_mu)**2)/prior_var - 1)
        kl_per_class = kl_dim.sum(dim=2)  # sum over latent_dim -> (B,K)

        # --- Router 概率加权 KL ---
        kl_loss = (kl_per_class * p_k).sum(dim=1).mean()

        # --- Reconstruction loss ---
        recon_loss = torch.sum((recon_x - x)**2, dim=1).mean()

        # --- Router softmax L2 正则 ---
        router_l2 = (p_k**2).sum(dim=1).mean()  # L2 loss over softmax output

        total_loss = recon_weight * recon_loss + KLD_weight * kl_loss + router_l2_weight * router_l2

        return total_loss, recon_loss, kl_loss, recon_x, mu, logvar, p_k