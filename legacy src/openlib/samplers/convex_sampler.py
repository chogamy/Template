"""
This is augment sampler
"""
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np


class ConvexSampler(nn.Module):
    def __init__(
        self,
        unseen_label_id,
        multiple_convex,
        multiple_convex_eval,
        batch_size,
        oos_num,
    ):
        super(ConvexSampler, self).__init__()

        self.multiple_convex = multiple_convex
        self.multiple_convex_eval = multiple_convex_eval
        self.unseen_label_id = unseen_label_id
        self.batch_size = batch_size
        # batch_size 를...gpu 개수에 따라서 조절할 필요가 있을 듯.
        self.oos_num = oos_num

        # self.num_convex = self.batch_size * self.multiple_convex
        # self.num_convex_eval = self.batch_size * self.multiple_convex_eval

    def forward(self, z, label_ids):
        device = label_ids.device

        # if n_label_ids is None:
        batch_size = z.size(0)

        num_convex = batch_size * self.multiple_convex

        pos_ids = label_ids != self.unseen_label_id
        neg_ids = label_ids == self.unseen_label_id

        b = label_ids[pos_ids].size(0)
        if b < 10:
            return z, label_ids

        convex_list = []

        patience = 0
        while len(convex_list) < num_convex:
            patience += 1
            cdt = np.random.choice(b, 2, replace=False)

            if label_ids[cdt[0]] != label_ids[cdt[1]]:  # 무슨 의미 인거지?
                s = np.random.uniform(0, 1, 1)
                convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])

            if patience > num_convex * 10:
                return z, label_ids
                break

        # convex_samples = torch.cat(convex_list, dim=0).view(self.num_convex, -1)
        convex_samples = torch.cat(convex_list, dim=0).view(num_convex, -1)

        z = torch.cat((z, convex_samples), dim=0)
        label_ids = torch.cat(
            # (label_ids, torch.tensor([self.unseen_label_id] * self.num_convex).to(device)),
            (
                label_ids,
                torch.tensor([self.unseen_label_id] * num_convex).to(device),
            ),
            dim=0,
        )

        return z, label_ids

        # else:
        #     # z = z.last_hidden_state.mean(dim=1)

        #     z = z.last_hidden_state

        #     b, l, e = z.shape

        #     batch_size = z.size(0)

        #     num_convex = batch_size * self.multiple_convex

        #     pos_ids = label_ids[:, -1] != 1

        #     b = label_ids[pos_ids].size(0)

        #     if b < 10:
        #         return z, label_ids, n_label_ids

        #     convex_list = []

        #     patience = 0
        #     while len(convex_list) < num_convex:
        #         patience += 1
        #         cdt = np.random.choice(b, 2, replace=False)  # 이 부분이 말이 되는지?

        #         if not torch.equal(label_ids[cdt[0]], label_ids[cdt[1]]):  # 무슨 의미 인거지?
        #             s = np.random.uniform(0, 1, 1)
        #             convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])

        #         if patience > num_convex * 10:
        #             return z, label_ids, n_label_ids

        #     # convex_samples = torch.cat(convex_list, dim=0).view(self.num_convex, -1)
        #     convex_samples = torch.cat(convex_list, dim=0).view(num_convex, l, e)

        #     sample_b = convex_samples.size(0)
        #     label_size = label_ids.size(1)
        #     sample_labels = torch.zeros(
        #         (sample_b, label_size), dtype=label_ids.dtype
        #     ).to(device)
        #     sample_labels[:, -1] = 1
        #     label_ids = torch.cat((label_ids, sample_labels), dim=0)

        #     z = torch.cat((z, convex_samples), dim=0)

        #     sample_n_label_ids = torch.ones(sample_b, dtype=n_label_ids.dtype).to(
        #         device
        #     )
        #     n_label_ids = torch.cat((n_label_ids, sample_n_label_ids), dim=0)

        #     return z, label_ids, n_label_ids
