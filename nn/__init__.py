from importlib import import_module


def get_model(args, data_config):
    task, model = args.model.split(".")

    task = import_module(f"nn.task_templates.{task}")

    if args.mode == "train":
        nn = getattr(task, model)(args, data_config)
    else:
        raise ValueError("not yet")

    # TODO: wrapper 어차피 PL밖에 없는데...

    if args.wrapper == "PL":
        from nn.wrapper_templates.pl import Wrapper

        nn = Wrapper(args, nn)

    elif args.wrapper == "HF":
        pass
    elif args.wrapper == None:
        pass

    return nn
