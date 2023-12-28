import argparse


MODE = ["train", "infer", "analysis"]
TASK = ["e1c1", "1e2c", "1e1d", "1e2d", "2e1c"]
WRAPPER = [None, "HF", "PL"]  # vllm?


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, required=True, type=str, help="mode")
    parser.add_argument("--task", default=None, required=True, type=str, help="task")
    parser.add_argument("--data", default=None, required=True, type=str, help="data")
    parser.add_argument("--wrapper", default=None, type=str, help="data")
    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--enc", default=None, type=str, help="Encoder")
    parser.add_argument("--enc1", default=None, type=str, help="Encoder 1")
    parser.add_argument("--enc2", default=None, type=str, help="Encoder 2")

    parser.add_argument("--dec", default=None, type=str, help="Decoder")
    parser.add_argument("--dec0", default=None, type=str, help="Decoder 1")
    parser.add_argument("--dec1", default=None, type=str, help="Decoder 2")

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

    def check_args(args):
        assert args.mode in MODE, f"Invalid mode, {MODE}"
        assert args.task in TASK, f"Invalid task, {TASK}"
        assert args.wrapper in WRAPPER, "Invalid wrapper"
        return args

    return check_args(args)
