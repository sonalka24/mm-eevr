import torch
import torch.nn as nn
from torch.distributions import kl_divergence

from lib.torch_utils import mvn_diag, param_groups


# ============================================================
# Modality as torch Module
# ============================================================
class Modality(nn.Module):
    def __init__(self,name,encoder,decoder):
        super().__init__()
        self.name    = name                                         # m1 / m2
        self.encoder = encoder                                      # Encoder or TextEncoder
        self.decoder = decoder                                      # Decoder

    @property
    def numel(self):
        return self.encoder.input_dim                               # reconstruction dimension

    @property
    def latent_dim(self):
        return self.encoder.latent_dim                              # latent dimension

    def params(self):
        return self.encoder.params() + self.decoder.params()         # encoder + decoder params

    def qz_x(self,x):
        return self.encoder(x)                                      # q(z|x), target

    def px_z(self,z):
        return self.decoder(z)                                      # p(x|z)

    def forward(self,x):
        return self.qz_x(x)


# ============================================================
# MVAE
# m1 = EDA/PPG
# m2 = PPG/TEXT
# TEXT only in m2
# ============================================================
class MVAE(nn.Module):
    def __init__(self,pz,m1_modality,m2_modality,beta_kl=0.001,beta_recon=0.01,
                 fusion_method="moe",moe_weighting="equal",bert_lr=1e-6):
        super().__init__()

        self.pz            = pz                                      # prior distribution constructor
        self.beta_kl       = beta_kl                                 # KL weight
        self.beta_recon    = beta_recon                              # reconstruction weight
        self.fusion_method = fusion_method                           # moe / poe
        self.moe_weighting = moe_weighting                           # equal / dimension
        self.bert_lr       = bert_lr                                 # BERT lr if text encoder trains BERT

        self.m1 = m1_modality                                       # EDA/PPG
        self.m2 = m2_modality                                       # PPG/TEXT

        if self.fusion_method not in ["moe","poe"]:
            raise ValueError("fusion_method must be either 'moe' or 'poe'.")

        if self.moe_weighting not in ["equal","dimension"]:
            raise ValueError("moe_weighting must be either 'equal' or 'dimension'.")

        if self.m1.latent_dim != self.m2.latent_dim:
            raise ValueError("Both modalities must have same latent_dim.")

        self.latent_dim = self.m1.latent_dim                         # shared latent dimension

    def get_effective_alpha(self,device):
        if self.moe_weighting=="equal":
            alpha = torch.tensor([0.5,0.5],device=device)            # equal MoE weight

        if self.moe_weighting=="dimension":
            m1_numel = float(self.m1.numel)
            m2_numel = float(self.m2.numel)
            total    = m1_numel + m2_numel
            alpha    = torch.tensor([m1_numel/total,m2_numel/total],device=device)

        alpha = alpha.reshape(2,1,1)                                 # broadcast over batch and latent dim

        return alpha

    def fuse_moe(self,qz_m1,qz_m2):
        mu_stack  = torch.stack([qz_m1.mean,qz_m2.mean],dim=0)       # 2,batch,latent
        var_stack = torch.stack([qz_m1.variance,qz_m2.variance],dim=0)

        var_stack = torch.clamp(var_stack,min=1e-6)                  # numerical safety
        alpha     = self.get_effective_alpha(mu_stack.device)        # modality weights

        fused_mu = torch.sum(alpha*mu_stack,dim=0)                   # weighted mean

        second_moment = torch.sum(
            alpha*(var_stack + torch.square(mu_stack)),
            dim=0,
        )                                                           # E[z^2]

        fused_var = second_moment - torch.square(fused_mu)           # Var[z]
        fused_var = torch.clamp(fused_var,min=1e-6)

        qz_fused = mvn_diag(
            loc=fused_mu,
            scale_diag=torch.sqrt(fused_var) + 1e-6,
        )

        return qz_fused

    def fuse_poe(self,qz_m1,qz_m2):
        mu_stack  = torch.stack([qz_m1.mean,qz_m2.mean],dim=0)       # expert means
        var_stack = torch.stack([qz_m1.variance,qz_m2.variance],dim=0)

        var_stack       = torch.clamp(var_stack,min=1e-6)
        precision_stack = 1.0/var_stack                              # expert precision

        prior_mu        = torch.zeros_like(mu_stack[0])              # prior mean
        prior_var       = torch.ones_like(var_stack[0])              # prior variance
        prior_precision = 1.0/prior_var                              # prior precision

        fused_precision = prior_precision + torch.sum(precision_stack,dim=0)
        fused_var       = 1.0/fused_precision

        fused_mu = fused_var*(
            prior_precision*prior_mu +
            torch.sum(precision_stack*mu_stack,dim=0)
        )

        qz_fused = mvn_diag(
            loc=fused_mu,
            scale_diag=torch.sqrt(fused_var) + 1e-6,
        )

        return qz_fused

    def fuse(self,qz_m1,qz_m2):
        if self.fusion_method=="moe":
            return self.fuse_moe(qz_m1,qz_m2)                       # MoE fusion

        if self.fusion_method=="poe":
            return self.fuse_poe(qz_m1,qz_m2)                       # PoE fusion

    def encode_m1(self,x):
        qz_x,target = self.m1.qz_x(x)                                # physiological input

        return qz_x,target

    def encode_m2(self,x):
        qz_x,target = self.m2.qz_x(x)                                # PPG or TEXT

        return qz_x,target

    def get_posteriors(self,inputs):
        x_m1 = inputs["m1"]
        x_m2 = inputs["m2"]

        qz_m1,target_m1 = self.encode_m1(x_m1)
        qz_m2,target_m2 = self.encode_m2(x_m2)

        qz_fused = self.fuse(qz_m1,qz_m2)                            # fused posterior

        target_dict = {
            "m1":target_m1,
            "m2":target_m2,
        }

        return qz_fused,[qz_m1,qz_m2],target_dict

    def forward(self,inputs,L=1):
        qz_fused,qz_list,target_dict = self.get_posteriors(inputs)

        device = qz_fused.mean.device
        pz = self.pz(
            loc=torch.zeros(self.latent_dim,device=device),
            scale_diag=torch.ones(self.latent_dim,device=device),
        )                                                           # p(z)

        recon = 0.0

        for l in range(L):
            z = qz_fused.rsample()                                   # reparameterized latent sample

            px_m1 = self.m1.px_z(z)
            px_m2 = self.m2.px_z(z)

            recon_m1 = px_m1.log_prob(target_dict["m1"]) / float(self.m1.numel)  # dim-normalized
            recon_m2 = px_m2.log_prob(target_dict["m2"]) / float(self.m2.numel)  # dim-normalized

            recon = recon + (recon_m1 + recon_m2)/2.0                # average modalities

        recon_mean = recon/float(L)

        kl   = kl_divergence(qz_fused,pz)                            # KL(q||p)
        elbo = self.beta_recon*recon_mean - self.beta_kl*kl          # ELBO

        self.recon_loss = -torch.mean(recon_mean)
        self.kl_loss    = torch.mean(kl)
        self.vae_loss   = -torch.mean(elbo)
        self.loss       = self.vae_loss

        return qz_fused

    def param_groups(self):
        all_params = self.m1.params() + self.m2.params()             # all trainable params

        if hasattr(self.m2.encoder,"bert_params"):
            bert_params = self.m2.encoder.bert_params()              # only TextEncoder has this
        else:
            bert_params = []                                         # normal Encoder has no BERT

        return param_groups(all_params,bert_params,self.bert_lr)     # BERT effective lr = bert_lr

    def train_step(self,inputs,optimizer,L=1):
        self.train()
        self.zero_grad(set_to_none=True)

        self.forward(inputs,L=L)
        self.loss.backward()
        optimizer.step()

        return self.loss.item()

    def get_alpha_values(self):
        if self.fusion_method=="moe":
            if self.moe_weighting=="equal":
                return {"m1":0.5,"m2":0.5}

            if self.moe_weighting=="dimension":
                m1_numel = float(self.m1.numel)
                m2_numel = float(self.m2.numel)
                total    = m1_numel + m2_numel

                return {"m1":m1_numel/total,"m2":m2_numel/total}

        if self.fusion_method=="poe":
            return {"m1":"precision-weighted","m2":"precision-weighted"}
