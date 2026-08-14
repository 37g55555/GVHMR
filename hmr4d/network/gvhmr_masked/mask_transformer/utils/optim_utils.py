import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, ChainedScheduler


def get_scheduler(name):
    return getattr(optim.lr_scheduler, name)


def parse_optimizer(cfg, model):
    return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def parse_scheduler(cfg, optimizer):
    interval = cfg.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if cfg.name == "SequentialLR":
        scheduler = {
            "scheduler": SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in cfg.schedulers
                ],
                milestones=cfg.milestones,
            ),
            "interval": interval,
        }
    elif cfg.name == "ChainedScheduler":
        scheduler = {
            "scheduler": ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in cfg.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(cfg.name)(optimizer, **cfg.args),
            "interval": interval,
        }
    return scheduler
