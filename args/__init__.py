import os
import yaml
import argparse


TASK = ["e1c1"]
OPTIMIZER = ["adamw", "adam"]
# 밑에 애들 이름은 고민좀 하자
LRSCHEDULER = ["cosine", "constant"]


def build_trainerargs(args):
    path = os.path.join("args", "trainer")
    trainer = os.path.join(path, args.trainer)
    with open(f"{trainer}.yaml") as f:
        trainer = yaml.load(f, Loader=yaml.FullLoader)

    args.trainer = trainer

    return args


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=None,
        required=True,
        type=str,
        help="mode",
        choices=["train", "infer", "analysis"],
    )
    parser.add_argument(
        "--model",
        default=None,
        required=True,
        type=str,
        help="task.model",
        choices=["e1c1.E1C1"],
    )
    parser.add_argument("--task", default=None, type=str, help="task")
    parser.add_argument("--data", default=None, required=True, type=str, help="data")
    parser.add_argument(
        "--wrapper",
        default=None,
        type=str,
        help="wrapper",
        choices=["trainer.custom", "transformers", "lightning.pytorch"],
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--model_path", default=None, type=str, help="save_dir")

    parser.add_argument(
        "--trainer", default=None, required=True, type=str, help="Trainer"
    )
    # callback은 ,로 파싱할 수 있게
    parser.add_argument(
        "--callbacks", default=None, required=True, type=str, help="Callbacks"
    )
    parser.add_argument(
        "--optimizers", default=None, required=True, type=str, help="Optimizers"
    )
    parser.add_argument(
        "--lrscheduler", default=None, required=True, type=str, help="LRScheduler"
    )

    parser.add_argument("--max_length", default=512, type=int, help="max_length")

    # Encoder args, 여러개일 경우 ,로 구분
    parser.add_argument("--enc", default=None, type=str, help="Encoder")

    # Decoder args, 여러개일 경우 ,로 구분
    parser.add_argument("--dec", default=None, type=str, help="Decoder")

    args = parser.parse_args()

    # TODO: 시드를 여기서 고정해도 되는지 확인
    def set_seed(args):
        if args.wrapper == "PL":
            from lightning.pytorch import seed_everything

            seed_everything(args.seed, workers=True)

        if args.wrapper == "HF":
            pass

        if args.wrapper == None:
            pass

    set_seed(args)

    args.task = args.model.rsplit(".", 1)[0]

    args = build_trainerargs(args)

    return args
