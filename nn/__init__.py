"""
wrapper를 여기서 만들어도 될듯?
"""


def get_model(args, data_config):
    def build_model_by_task(args):
        nn = None
        if args.task == "e1c1":
            from .task_templates.e1c1 import E1C1

            nn = E1C1(args, data_config)
            
        return nn

    nn = build_model_by_task(args)

    if args.wrapper == "PL":
        from nn.wrapper_templates.pl import Wrapper

        nn = Wrapper(args, nn)

    elif args.wrapper == "HF":
        pass
    elif args.wrapper == None:
        pass

    return nn
