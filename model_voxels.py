import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=1024, pos_dim=3, pos_emb_dim=64, latent_dim=256, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.pos_emb_dim = pos_emb_dim
        self.latent_dim = latent_dim
        self.pos_dim = pos_dim

        self.pos_emb = nn.Sequential(
            nn.Linear(pos_dim, 4*pos_dim),
            nn.GELU(),
            nn.Linear(4*pos_dim, pos_emb_dim),
            nn.GELU()
        )

        self.enc_fc = nn.Sequential(
            nn.Linear(input_dim + pos_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim + pos_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.dec_out = nn.Linear(hidden_dim, input_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def encode(self, clip, pos_idx):
        pos_emb = self.pos_emb(pos_idx)
        h = torch.cat([clip, pos_emb], dim=1)
        h = self.enc_fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, pos_idx):
        pos_emb = self.pos_emb(pos_idx)
        h = torch.cat([z, pos_emb], dim=1)
        h = self.dec_fc(h)
        out = self.dec_out(h)
        return out
    
    def forward(self, clip, pos_idx):
        mu, logvar = self.encode(clip, pos_idx)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, pos_idx)
        return recon, mu, logvar

# import torch
# import torch.nn as nn

# class ConditionalVAE(nn.Module):
#     def __init__(self, input_dim=1024, pos_dim=3, pos_emb_dim=75, latent_dim=256, hidden_dim=512, num_positions=[81, 104, 83]):
#         super().__init__()
#         # ... (other dimensions remain the same) ...
#         self.pos_emb_dim = pos_emb_dim
        
#         # Create three separate embedding layers, one for each position dimension (x, y, z)
#         # We assume each coordinate can go from 0 up to num_positions-1.
#         # The final embedding dimension will be 3 * individual_emb_dim.
#         # Let's make individual_emb_dim divisible and clear.
#         individual_emb_dim = pos_emb_dim // pos_dim
#         if pos_emb_dim % pos_dim != 0:
#             raise ValueError("pos_emb_dim must be divisible by pos_dim")

#         self.pos_emb_x = nn.Embedding(num_positions[0], individual_emb_dim)
#         self.pos_emb_y = nn.Embedding(num_positions[1], individual_emb_dim)
#         self.pos_emb_z = nn.Embedding(num_positions[2], individual_emb_dim)

#         # The rest of your model architecture remains the same
#         # The input to enc_fc is still input_dim + pos_emb_dim
#         self.enc_fc = nn.Sequential(
#             nn.Linear(input_dim + pos_emb_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU()
#         )
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

#         self.dec_fc = nn.Sequential(
#             nn.Linear(latent_dim + pos_emb_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU()
#         )
#         self.dec_out = nn.Linear(hidden_dim, input_dim)
#         self._init_weights()

#     def _init_weights(self):
#         # ... (your init_weights function is fine) ...
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Embedding):
#                 nn.init.normal_(m.weight, mean=0.0, std=0.02) # A common std for embeddings

#     def get_pos_embedding(self, pos_idx):
#         # IMPORTANT: Input pos_idx must be a LongTensor (integers)
#         # It should have the shape [B, 3]
#         if pos_idx.dtype != torch.long:
#             pos_idx = pos_idx.long()
            
#         x_emb = self.pos_emb_x(pos_idx[:, 0]) # Shape: [B, individual_emb_dim]
#         y_emb = self.pos_emb_y(pos_idx[:, 1]) # Shape: [B, individual_emb_dim]
#         z_emb = self.pos_emb_z(pos_idx[:, 2]) # Shape: [B, individual_emb_dim]
        
#         # Concatenate to get the final position embedding
#         pos_emb = torch.cat([x_emb, y_emb, z_emb], dim=1) # Shape: [B, pos_emb_dim]
#         return pos_emb

#     def encode(self, clip, pos_idx):
#         pos_emb = self.get_pos_embedding(pos_idx) # Use the new embedding method
#         h = torch.cat([clip, pos_emb], dim=1)
#         h = self.enc_fc(h)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         return mu, logvar
    
#     def decode(self, z, pos_idx):
#         pos_emb = self.get_pos_embedding(pos_idx) # Use the new embedding method
#         h = torch.cat([z, pos_emb], dim=1)
#         h = self.dec_fc(h)
#         out = self.dec_out(h)
#         return out
    
#     # ... (rest of the class remains the same) ...
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def forward(self, clip, pos_idx):
#         mu, logvar = self.encode(clip, pos_idx)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decode(z, pos_idx)
#         return recon, mu, logvar
    
class Router(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, K=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, K)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        logits = self.fc2(h)
        probs = F.softmax(logits, dim=-1)
        return probs
    
class CVAEWithRouter(nn.Module):
    def __init__(self, input_dim=1024, pos_dim=3, pos_emb_dim=64, latent_dim=256, hidden_dim=512, K=4):
        super().__init__()
        self.cvae = ConditionalVAE(input_dim=input_dim, pos_dim=pos_dim, pos_emb_dim=pos_emb_dim, 
                                   latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.router = Router(input_dim=input_dim, hidden_dim=hidden_dim, K=K)
        self.K = K
        self.latent_dim = latent_dim

    def forward(self, x, pos_idx, activation, recon_weight=1.0, KLD_weight=1.0, router_l2_weight=4.0):
        B = x.size(0)
        device = x.device
        latent_dim = self.latent_dim

        # cvae forward
        recon_x, mu, logvar = self.cvae(x, pos_idx)
        # router forward
        p_k = self.router(x)

        # KL loss
        prior_mu = torch.zeros(B, self.K, latent_dim, device=device)
        activation_sign = torch.sign(activation).view(B, 1) # [B, 1]

        for k in range(self.K):
            prior_mu[:, k, :] = activation_sign * (1.0 + k)

        prior_var = activation.abs().view(B, 1, 1).expand(B, self.K, latent_dim) 
        prior_var = torch.clamp(prior_var, min = 1e-5)

        var = torch.exp(logvar).unsqueeze(1)
        mu_expand = mu.unsqueeze(1)
        logvar_expand = logvar.unsqueeze(1)

        kl_dim = 0.5 * ( torch.log(prior_var) -logvar_expand + (var + (mu_expand - prior_mu)**2)/prior_var - 1 )
        kl_per_class = kl_dim.sum(dim=2) # [B, K]

        kl_loss = (kl_per_class * p_k).sum(dim=1).mean()

        recon_loss = torch.sum((recon_x - x)**2, dim=1).mean()

        router_l2 = (p_k**2).sum(dim=1).mean()

        total_loss = recon_weight * recon_loss + KLD_weight * kl_loss + router_l2_weight * router_l2

        return total_loss, recon_loss, kl_loss, recon_x, mu, logvar, p_k
