import torch.optim
import torch.nn as nn


def build_optimizer(model, optim_cfg):
    assert 'type' in optim_cfg
    _optim_cfg = optim_cfg.copy()
    lr = _optim_cfg.get('lr')

    optim_type = _optim_cfg.pop('type')
    optim = getattr(torch.optim, optim_type)
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    pg_conv0, pg_conv1, pg_conv2 = [], [], []
    for k, v in model.named_modules():
        if "affinity_conv" in k:

            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg_conv2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg_conv0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg_conv1.append(v.weight)  # apply decay
        else:
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

    optimizer = optim(pg1, **_optim_cfg)
    optimizer.add_param_group({'params': pg2})
    optimizer.add_param_group({'params': pg0})
    optimizer.add_param_group({'params': pg_conv2, 'lr': lr * 500})
    optimizer.add_param_group({'params': pg_conv0, 'lr': lr * 500})
    optimizer.add_param_group({'params': pg_conv1, 'lr': lr * 500})
    del  pg0, pg1, pg2, pg_conv0, pg_conv1, pg_conv2
    return optimizer
    #return optim([{'params': filter(lambda p: p.requires_grad, model.parameters())},
    #              ], **_optim_cfg)
