import torch
import random
from typing import Any, Dict, Union

def generate_positive_sample(negative_data, label: torch.Tensor):
    # print("generate_po")
    positive_num = 3 # args.positive_num
    positive_sample = []
    # print("\nnegative in generate", negative_data[0])
    for index in range(label.shape[0]):
        input_label = int(label[index])
        # print("input_label", negative_data[input_label])
        positive_sample.extend(random.sample(negative_data[input_label], positive_num))
        # print("po", len(positive_sample))

    return list_item_to_tensor(positive_sample)

def _prepare_inputs(device, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
                    
    return inputs


def list_item_to_tensor(inputs_list):
    # print("listitem")
    batch_list = {}
    for key, value in inputs_list[0].items():
        batch_list[key] = []
    for inputs in inputs_list:
        for key, value in inputs.items():
            batch_list[key].append(value)
            # print("len", len(value))
    # print("batch_list", batch_list.keys())
    batch_tensor = {}
    for key, value in batch_list.items():
        # print("value", value)
        # batch_tensor[key] = torch.tensor(value)
        batch_tensor[key] = torch.stack(value)
    return batch_tensor