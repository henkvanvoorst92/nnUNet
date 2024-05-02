from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(y_pred, y_true, skel_pred=None, skel_true=None):
    """[this function computes the cldice metric]

    Args:
        y_pred ([bool]): [predicted image]
        y_true ([bool]): [ground truth image]
        skel_pred : pred skeleton acquired via alternative way
        skel_true : true skeleton acquired in alternative way

    Returns:
        [float]: [cldice metric]
    """
    if skel_pred is None:
        if len(y_pred.shape) == 2:
            skel_pred = skeletonize(y_pred)
        elif len(y_pred.shape)==3:
            skel_pred = skeletonize_3d(y_pred)

    if skel_true is None:
        if len(y_true.shape) == 2:
            skel_true = skeletonize(y_true)
        elif len(y_true.shape)==3:
            skel_true = skeletonize_3d(y_true)

    tprec = cl_score(y_pred, skel_true)
    tsens = cl_score(y_true, skel_pred)

    return 2*tprec*tsens/(tprec+tsens)
