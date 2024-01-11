from importlib import import_module


def get_model(args, data_config):
    task, model = args.model.split(".")

    task = import_module(f"nn.task_templates.{task}")

    if args.mode == "train":
        nn = getattr(task, model)(args, data_config)
    else:
        raise ValueError("not yet")

    # TODO: wrapper 어차피 PL밖에 없는데 이렇게 해야할까?
    if args.wrapper == "lightning.pytorch":
        from nn.wrapper_templates.pl import Wrapper

        nn = Wrapper(args, nn)

    elif args.wrapper == "transformers":
        pass
    elif args.wrapper == "trainer.custom":
        pass
    else:
        raise ValueError("Invalid wrapper")

    return nn
