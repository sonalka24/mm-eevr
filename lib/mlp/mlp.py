import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution

from lib.torch_utils import dense, param_groups, to_tensor


# ============================================================
# Multimodal MLP baseline - intermediate fusion
# m1 = EDA/PPG
# m2 = PPG/TEXT
# TEXT only in m2
#
# Same Modality objects as MVAE (encoder + decoder), but no VAE stage:
# each encoder gives a deterministic 50-d feature (posterior mean),
# features are concatenated here and classified, trained end-to-end with BCE.
# Decoders are not used by this model.
# ============================================================
class MLP(nn.Module):
    def __init__(self,m1_modality,m2_modality,bert_lr=1e-6):
        super().__init__()

        self.m1      = m1_modality                                  # EDA/PPG
        self.m2      = m2_modality                                  # PPG/TEXT
        self.bert_lr = bert_lr                                      # BERT lr if text encoder trains BERT

        self.fused_dim  = self.m1.latent_dim + self.m2.latent_dim     # concatenated feature dimension
        self.classifier = dense(self.fused_dim,1)                    # binary classifier head

    def encode(self,modality,x):
        out,_ = modality.qz_x(x)                                     # Encoder / TextEncoder output

        if isinstance(out,Distribution):
            return out.mean                                         # deterministic feature, same as MVAE classifier

        return out

    def fuse(self,h_m1,h_m2):
        return torch.cat([h_m1,h_m2],dim=-1)                        # batch, latent_m1 + latent_m2

    def forward(self,inputs,y=None):
        h_m1 = self.encode(self.m1,inputs["m1"])
        h_m2 = self.encode(self.m2,inputs["m2"])

        h      = self.fuse(h_m1,h_m2)                                # intermediate fusion
        logits = self.classifier(h)

        if y is not None:
            y = y.float().reshape(-1,1)

            self.cls_loss = F.binary_cross_entropy_with_logits(logits,y)
            self.loss     = self.cls_loss

        return logits

    def param_groups(self):
        classifier_params = list(self.classifier.parameters())       # classifier head params
        encoder_params    = self.m1.encoder.params() + self.m2.encoder.params()  # encoder params only

        if hasattr(self.m2.encoder,"bert_params"):
            bert_params = self.m2.encoder.bert_params()              # only TextEncoder has this
        else:
            bert_params = []                                         # no BERT for PPG

        return param_groups(classifier_params + encoder_params,bert_params,self.bert_lr)

    def train_step(self,inputs,y,optimizer):
        self.train()
        self.zero_grad(set_to_none=True)

        self.forward(inputs,y=y)
        self.loss.backward()
        optimizer.step()

        return self.loss.item()

    @torch.no_grad()
    def predict_logits(self,inputs):
        self.eval()
        device = self.classifier.weight.device
        logits = self.forward(to_tensor(inputs,device))
        return logits.squeeze(-1).cpu()

    def predict_proba(self,inputs):
        logits = self.predict_logits(inputs)
        probs  = torch.sigmoid(logits)
        return probs

    def predict_class(self,inputs):
        probs = self.predict_proba(inputs)
        preds = (probs >= 0.5).int()
        return preds
