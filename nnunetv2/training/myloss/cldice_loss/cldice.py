import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import SoftSkeletonize

class soft_cldice(nn.Module):
    def __init__(self,iter_=10, smooth = 1.,
                 y_true_skel_input_channel=None,
                 y_pred_skel_input_channel=None,
                 exclude_background=False,
                 *args, **kwargs):
        super(soft_cldice, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=self.iter)

        self.exclude_background = exclude_background
        self.y_true_skel_input_channel = y_true_skel_input_channel
        self.y_pred_skel_input_channel = y_pred_skel_input_channel

    def compute_cldice(self, y_true, y_pred, skel_true, skel_pred):
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice
    def forward(self, y_true, y_pred):
        #Either choose to skeletonize y_true skeleton or to
        #provide y_true as skeleton as additional  input

        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]

        # skeletons can be stacked in the y_true and y_pred channels

        if self.y_true_skel_input_channel is not None:
            skel_true = y_true[:,self.y_true_skel_input_channel:self.y_true_skel_input_channel+1]
        else:
            skel_true = self.soft_skeletonize(y_true)

        if self.y_pred_skel_input_channel is not None:
            skel_pred = y_pred[:,self.y_pred_skel_input_channel:self.y_pred_skel_input_channel+1]
        else:
            skel_pred = self.soft_skeletonize(y_pred)

        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        #cl_dice = self.compute_cldice(y_true, y_pred, skel_true, skel_pred)
        return cl_dice

def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1., exclude_background=False):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        dice = soft_dice(y_true, y_pred)
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice
