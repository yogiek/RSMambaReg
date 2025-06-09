import torch
import torch.nn as nn
import torch.nn.functional as F

class ElasticWeightConsolidation:
    def __init__(self, model, lambda_ewc=1000, device="cuda"):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher_matrix = {}
        self.saved_params = {}

    def compute_fisher_matrix(self, dataloader):
        self.model.eval()
        fisher_matrix = {n: torch.zeros_like(p) for n, p in self.params.items()}
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_matrix[n] += (p.grad ** 2).detach()

        # Averaging Fisher Matrix
        for n in fisher_matrix:
            fisher_matrix[n] /= len(dataloader)

        self.fisher_matrix = fisher_matrix

    def save_params(self):
        for n, p in self.params.items():
            self.saved_params[n] = p.clone().detach()

    def compute_ewc_loss(self):
        ewc_loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_matrix:
                fisher = self.fisher_matrix[n]
                saved_param = self.saved_params[n]
                ewc_loss += (fisher * (p - saved_param) ** 2).sum()
        return self.lambda_ewc * ewc_loss
    
class CustomEWCLoss(nn.Module):
    def __init__(self, base_loss, ewc):
        super(CustomEWCLoss, self).__init__()
        self.base_loss = base_loss
        self.ewc = ewc

    def forward(self, logits, labels):
        base_loss = self.base_loss(logits, labels)
        ewc_loss = self.ewc.compute_ewc_loss()
        total_loss = base_loss + ewc_loss
        return total_loss