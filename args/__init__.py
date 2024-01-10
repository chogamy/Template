import argparse


TASK = ["e1c1"]
WRAPPER = ["trainer.custom", "transformers", "lightning.pytorch"]
OPTIMIZER = ["adamw", "adam"]
# 밑에 애들 이름은 고민좀 하자
LRSCHEDULER = ["cosine", "constant"]


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
    parser.add_argument("--wrapper", default=None, type=str, help="data")
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

    # def check_args(args):
    #     assert (
    #         args.mode in MODE
    #     ), f"Invalid mode\n Avail modes: {MODE}\n YOU:{args.mode}"
    #     assert (
    #         args.model in MODEL
    #     ), f"Invalid model\n Avail models: {MODEL}\n YOU:{args.model}"
    #     assert (
    #         args.task in TASK
    #     ), f"Invalid task\n Avail tasks: {TASK}\n YOU:{args.task}"
    #     assert (
    #         args.wrapper in WRAPPER
    #     ), f"Invalid wrapper\n Avail wrappers: {WRAPPER}\n YOU:{args.wrapper}"
    #     assert (
    #         args.optimizers in OPTIMIZER
    #     ), f"Invalid optimizer\n Avail optimizers: {OPTIMIZER}\n YOU:{args.optimizers}"
    #     assert (
    #         args.lrscheduler in LRSCHEDULER
    #     ), f"Invalid lrscheduler\n Avail optimizers: {LRSCHEDULER}\n YOU:{args.lrscheduler}"
    #     return args

    return args
