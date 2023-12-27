import torch
from torch import nn
import torch.nn.functional as F


class NegativeSampler(nn.Module):
    def __init__(self, num_classes, hidden, r, radius):
        super().__init__()

        self.cov = nn.Parameter(torch.zeros(num_classes, hidden, hidden), requires_grad=False)
        self.r = r
        self.radius = radius
        self.num_classes = num_classes
        self.k = 3
        self.steps = 10
        self.lr = 0.1  # 하이퍼 파라미터??????

    def forward(self, zs, classifiers=None):
        device = zs[0].device

        # first, sample e
        es = {i: None for i in range(self.num_classes)}
        for i in range(self.num_classes):
            num_samples, hidden_size = zs[i].size()
            diag = (self.radius / 2) * torch.diag(self.cov[i])
            e = torch.randn(num_samples, hidden_size).to(device) * torch.sqrt(diag)
            es[i] = e.requires_grad_(True)

        # update e by gradient ascent
        if classifiers:
            for _ in range(self.steps):
                # classifiers.train()
                for i, classifier in enumerate(classifiers):
                    # classifier.train()

                    logits = classifier(es[i]).squeeze(1)
                    # print(es[i].requires_grad)
                    # print(logits.requires_grad)
                    # assert 1 == 0

                    labels = torch.zeros(logits.size()).to(device)

                    # for n, p in classifier.named_parameters():
                    #     print(f"{n} {p.requires_grad}")

                    # print(logits)
                    # print(logits.shape)
                    # print(logits.requires_grad)
                    # print(labels)
                    # print(labels.shape)
                    # print(labels.requires_grad)
                    # assert 1 == 0

                    loss = F.binary_cross_entropy_with_logits(logits, labels)

                    loss.backward()

                    with torch.no_grad():
                        es[i] += self.lr * es[i].grad
                    es[i].grad.zero_()

                    # classifier.eval()

        # calculate alpha, then update e
        for i in range(self.num_classes):
            norm = torch.norm(es[i] - zs[i])
            if self.radius <= norm <= self.r * self.radius:
                alpha = 1
            elif self.r * self.radius <= norm:
                alpha = self.r * self.radius / norm
            elif norm <= self.radius:
                alpha = self.radius / norm

            norm_e = torch.norm(es[i])
            alpha = torch.clamp(self.radius / norm_e, max=1)

            es[i] = (alpha / norm_e) * es[i]

        for i in range(self.num_classes):
            zs[i] = zs[i] + es[i]

        return zs

    def update_cov(self, model, train_dataloader):
        zs = {i: [] for i in range(self.num_classes)}

        device = model.device

        for batch in train_dataloader:
            model_input = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            pooled_output, outputs = model(model_input)

            for i in range(self.num_classes):
                zs[i].append(pooled_output[batch["labels"] == i])

        for i in range(self.num_classes):
            zs[i] = torch.cat(zs[i])
            zs[i] = zs[i] - zs[i].mean(dim=0)
            self.cov[i] = torch.mm(zs[i].t(), zs[i]) / zs[i].size(0)
