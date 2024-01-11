import os
import yaml

from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
from transformers import AutoTokenizer

"""
데이터셋 관련한 전처리,
vocab 구성
"""

# 0: Origin(출신차별) hate speech based on place of origin or identity;
# 1: Physical(외모차별) hate speech based on physical appearance (e.g. body, face) or disability;
# 2: Politics(정치성향차별) hate speech based on political stance;
# 3: Profanity(혐오욕설) hate speech in the form of swearing, cursing, cussing, obscene words, or expletives; or an unspecified hate speech category;
# 4: Age(연령차별) hate speech based on age;
# 5: Gender(성차별) hate speech based on gender or sexual orientation (e.g. woman, homosexual);
# 6: Race(인종차별) hate speech based on ethnicity;
# 7: Religion(종교차별) hate speech based on religion;
# 8: Not Hate Speech(해당사항없음).

IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
LABELS = [
    "Origin",
    "Physical",
    "Politics",
    "Profanity",
    "Age",
    "Gender",
    "Race",
    "Religion",
    "Not Hate Speech",
]


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
    dataset = load_dataset("jeanlee/kmhas_korean_hate_speech")
    dataset["val"] = dataset.pop("validation")

    # cache 고민

    return dataset


def get_dataconfig(args):
    id_to_label = {k: v for k, v in zip(IDS, LABELS)}
    label_to_id = {k: v for k, v in zip(LABELS, IDS)}
    tokenizer = AutoTokenizer.from_pretrained(args.enc)

    dataconfig = {
        "id_to_label": id_to_label,
        "label_to_id": label_to_id,
        "tokenizer": tokenizer,
    }
    return dataconfig


def preprocess(example, tokenizer, args):
    """
    Task 별로 다른 전처리 필요
    """
    # if args.task == "e1c1":
    model_input = tokenizer(
        example["text"],
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
    )

    model_input["n_label"] = [len(label) for label in example["label"]]
    mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    one_hot_labels = mlb.fit_transform(example["label"])
    model_input["label"] = one_hot_labels.tolist()

    return model_input


if __name__ == "__main__":
    dataset = _load_dataset()

    path = os.path.dirname(os.path.abspath(__file__))

    vocab_path = os.path.join(path, "vocab.yaml")

    id_to_label = {k: v for k, v in zip(IDS, LABELS)}
    label_to_id = {k: v for k, v in zip(LABELS, IDS)}

    with open(vocab_path, "w") as f:
        yaml.dump({"id_to_label": id_to_label, "label_to_id": label_to_id}, f)
