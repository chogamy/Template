"""
wrapper를 여기서 만들어도 될듯?
"""


def get_model(args):
    def build_model_by_task(args):
        nn = None
        if args.task == "e1c1":
            from .task_templates.e1c1 import E1C1

            print(args)

            nn = E1C1(args)
            assert 1 == 0

        return nn

    nn = build_model_by_task(args)

    assert 1 == 0

    if args.wrapper == "PL":
        from wrapper_templates.pl import Wrapper

        nn = Wrapper(args)

    elif args.wrapper == "HF":
        pass
    elif args.wrapper == None:
        pass

    return nn
