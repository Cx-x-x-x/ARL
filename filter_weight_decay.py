import torch.nn as nn


def group_weight(model):
    group_decay = []
    group_no_decay = []
    for name, param in model.named_parameters():
        # if ('conv' and 'weight') in name:  # = conv or weight in name why?????????
        if 'conv' in name and 'weight' in name:
            group_decay.append(param)
            # print('decay:', name)
        else:
            group_no_decay.append(param)
            # print('no_decay:', name)

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


# def group_weight(model):
#     group_decay = []
#     group_no_decay = []
#     for m in model.modules():  # model.modules()是遍历所有层，不是遍历所有参数
#         if isinstance(m, nn.Linear):
#             group_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, nn.Conv2d):
#             group_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, nn.BatchNorm2d):
#             if m.weight is not None:
#                 group_no_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         # elif isinstance(m, nn.Parameter):
#         #     if m.factor is not None:
#         #         group_no_decay.append(m.factor)
#
#     assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
#     groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
#     return groups


# if isinstance(param, nn.Linear):
#     group_decay.append(param.weight)
#     if param.bias is not None:
#         group_no_decay.append(param.bias)
# elif isinstance(param, nn.Conv2d):
#     group_decay.append(param.weight)
#     if param.bias is not None:
#         group_no_decay.append(param.bias)
# elif isinstance(param, nn.BatchNorm2d):
#     if param.weight is not None:
#         group_no_decay.append(param.weight)
#     if param.bias is not None:
#         group_no_decay.append(param.bias)
# elif isinstance(param, nn.Parameter) and 'factor' in name:
#     if param.data is not None:
#         group_no_decay.append(param.data)