from collections import defaultdict, abc
import wandb


class WandbLogging:
    def __init__(self, args):
        self.stats = defaultdict(list)
        wandb.init(project=args.wandb.project, group=args.wandb.group)
        wandb.run.name = args.wandb.name + "_" + wandb.run.name
        wandb.config.update(self.__flatten(args))

    def __flatten(self, d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, abc.MutableMapping):
                items.extend(self.__flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    def update(self, **kwargs):
        wandb.log(dict(kwargs))

    def log(self, **kwargs):
        wandb.log(kwargs)
