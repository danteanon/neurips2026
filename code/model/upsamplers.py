import torch


def anyup_upsampler():
    anyup = torch.hub.load('wimmerth/anyup', 'anyup').cuda()
    return anyup
