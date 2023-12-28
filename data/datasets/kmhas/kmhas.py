import os
from ordered_set import OrderedSet

from datasets import load_dataset

"""
데이터셋 관련한 전처리,
vocab 구성
"""


def _load_dataset():
    """
    must return
    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 78977
        })
        validation: Dataset({
            features: ['text', 'label'],
            num_rows: 8776
        })
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 21939
        })
    })
    """
    return load_dataset("jeanlee/kmhas_korean_hate_speech")


if __name__ == "__main__":
    dataset = _load_dataset()

    path = os.path.dirname(os.path.abspath(__file__))

    vocab_path = os.path.join(path, "vocab.yaml")

    unique_ids = OrderedSet()

    id_to_label = {}
    label_to_id = {}

    with open(vocab_path, "w") as f:
        f.write("")
