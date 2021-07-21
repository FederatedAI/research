"""
differential privacy utility functions
author: hyq
"""
import torch


def add_dp_v1(input_tensor, clip_value=5.0, variance=1.0, device=torch.device('cpu')):
    """
    cpu version
    :param input_tensor:
    :param clip_value:
    :param variance:
    :param device:
    :return:
    """
    input_copy = input_tensor.detach().clone()

    with torch.no_grad():
        up_norm_factor = max(torch.max(torch.norm(input_copy, dim=1)).item()/clip_value, 1.0)
        input_noised = input_copy / up_norm_factor + torch.normal(0, variance, input_tensor.shape)
    input_with_dp = torch.autograd.Variable(input_noised, requires_grad=True).to(device)
    return input_with_dp


def add_dp_v2(input_tensor, clip_value=5.0, variance=1.0, device=torch.device('cuda')):
    """
    gpu compatible version
    :param input_tensor: variable
    :param clip_value: clipping value of 2-norm
    :param variance: variance of gaussian noise
    :param device:
    :return: variable with dp applied
    """
    input_copy = input_tensor.detach().clone()

    with torch.no_grad():
        # clip 2-norm per sample
        norm_factor = torch.div(torch.max(torch.norm(input_copy, dim=1)), clip_value+1e-6).clamp(min=1.0)
        # add gaussian noise
        input_noised = torch.div(input_copy, norm_factor) + torch.normal(0, variance, input_tensor.shape, device=device)
    input_with_dp = torch.autograd.Variable(input_noised, requires_grad=True).to(device)
    return input_with_dp


def add_dp_v3(input_tensor, clip_value=5.0, variance=1.0):
    """
    gpu compatible version
    :param input_tensor: variable
    :param clip_value: clipping value of 2-norm
    :param variance: variance of gaussian noise
    :param device:
    :return: variable with dp applied
    """
    input_copy = input_tensor.detach().clone()

    with torch.no_grad():
        # clip 2-norm per sample
        norm_factor = torch.div(torch.max(torch.norm(input_copy, dim=1)), clip_value+1e-6).clamp(min=1.0)
        # add gaussian noise
        input_noised = torch.div(input_copy, norm_factor) + torch.normal(0, variance, input_tensor.shape).cuda()
    input_with_dp = torch.autograd.Variable(input_noised, requires_grad=True).cuda()
    return input_with_dp

def add_dp_to_list(input_tensor_list, clip_value=5.0, variance=1.0):
    """
    gpu compatible version
    :param input_tensor: variable
    :param clip_value: clipping value of 2-norm
    :param variance: variance of gaussian noise
    :param device:
    :return: variable with dp applied
    """
    output_list = []
    for input_tensor in input_tensor_list:
        if isinstance(input_tensor, tuple):
            input_copy = input_tensor[0].detach().clone()
        else:
            input_copy = input_tensor.detach().clone()
        with torch.no_grad():
            # clip 2-norm per sample
            norm_factor = torch.div(torch.max(torch.norm(input_copy, dim=1)), clip_value+1e-6).clamp(min=1.0)
            # add gaussian noise
            input_noised = torch.div(input_copy, norm_factor) + torch.normal(0, variance, input_copy.shape).cuda()
        input_with_dp = torch.autograd.Variable(input_noised, requires_grad=True).cuda()
        output_list.append(input_with_dp)
    return output_list