"""
This is augment sampler
"""
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np


class MultiConvexSampler(nn.Module):
    def __init__(self, unseen_label_id, max_len=150):
        super(MultiConvexSampler, self).__init__()


        self.max_len = max_len
        

    def forward(self, input_ids, attn_mask, z, label_ids, n_label_ids):
        device = label_ids.device

        z = z.last_hidden_state

        if z.shape[0] < 10:
            return z, label_ids, n_label_ids
    
        # 여기서 sampling
        convex_list = []
        for b in range(z.shape[0]):
            cdt = np.random.choice(z.shape[0], 2, replace=False)


            patience = 0
            while not torch.equal(label_ids[cdt[0]], label_ids[cdt[1]]):
                cdt = np.random.choice(z.shape[0], 2, replace=False)
                patience += 1
                if patience > 1000:
                    break
            
            input_id1 = input_ids[cdt[0]]
            input_id2 = input_ids[cdt[1]]

            pad_start_idx1 = (input_id1 == 0).nonzero(as_tuple=True)[0][0]
            pad_start_idx2 = (input_id2 == 0).nonzero(as_tuple=True)[0][0]
            
            length = (pad_start_idx1 + pad_start_idx2) / 2
            length = int(length)

            s = np.random.uniform(0, 1, 1)
            convex_sample = s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]]
            convex_list.append(convex_sample[:length, :])
            

            
        # 여기서 concat
        for b in range(z.shape[0]):
            input_id = input_ids[b]
            pad_start_idx = (input_id == 0).nonzero(as_tuple=True)[0][0]

            attn_start_idx = (attn_mask[b] == 0).nonzero(as_tuple=True)[0][0]

            convex_sample = convex_list[b]
            length = convex_sample.shape[0]

            randomness = np.random.choice([0,1,2])

            z_ = z[b, :pad_start_idx, :].squeeze(0)
            padding = z[b, pad_start_idx:, :].squeeze(0)

            # concat
            if randomness == 0 or n_label_ids[b] == 3:
                z__ = z[b]
            elif randomness == 1:
                '''
                원래 벡터 왼쪽에
                '''
                z__ = torch.cat((convex_sample, z_, padding), dim=0)
                label_ids[b][-1] = 1
                n_label_ids[b] += 1

                # size
                attn_mask[b][attn_start_idx:attn_start_idx+length] = 1
                
                
            elif randomness == 2:
                '''
                원래 벡터 오른쪽에 
                '''
                z__ = torch.cat((z_, convex_sample, padding), dim=0)
                label_ids[b][-1] = 1
                n_label_ids[b] += 1

                attn_mask[b][attn_start_idx:attn_start_idx+length] = 1
                
                    
            z[b] = z__[:self.max_len, :]
        
        return z, attn_mask, label_ids, n_label_ids
