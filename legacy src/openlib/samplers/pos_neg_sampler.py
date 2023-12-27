import torch
import torch.nn as nn
import numpy as np


class PosNegSampler(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.K = 7500
        # self.K = 7500  # args.quesize  -> hyperparmeter (데이터마다 다르게 함)

        self.T = 0.5
        # self.T = 0.5  # args.temperature  -> hyperparameter
        self.top_k = 25

        self.label_queue = torch.randint(0, num_classes + 1, [self.K])
        # self.feature_queue = torch.randn(self.K, self.encoder_k.dense.out_features)
        # hidden_size = self.encoder_k.dense.out_features
        self.feature_queue = torch.randn(self.K, hidden_size)
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    def forward(self, liner_q, label_q):
        """
        input
            linear_q
            label_q
        """
        device = liner_q.device

        label_queue = self.label_queue.clone().detach()  # K
        feature_queue = self.feature_queue.clone().detach()  # K * hidden_size

        # 1. expand label_queue and feature_queue to batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat(
            [batch_size, 1, 1]
        )  # batch_size * K * hidden_size

        # 2.caluate sim
        cos_sim = torch.einsum("nc,nkc->nk", [liner_q, tmp_feature_queue.to(device)])

        # 3. get index of postive and neigative
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue.to(device), tmp_label)
        neg_mask_index = ~pos_mask_index

        # 4.another option
        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5.topk
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = cos_sim.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0 : self.top_k]  # self.topk = 25
        pos_sample = pos_sample_top_k
        pos_sample = pos_sample.contiguous().view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_topk = min(pos_min, self.top_k)
        neg_sample = neg_sample.repeat([1, neg_topk])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def _dequeue_and_enqueue(self, keys, label):
        """
        using in model.forward
        """

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[:batch_size]
            label = label[:batch_size]

        # replace the keys at ptr (dequeue ans enqueue)
        self.feature_queue[ptr : ptr + batch_size, :] = keys
        self.label_queue[ptr : ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr
