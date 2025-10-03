import numpy as np
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from nnunetv2.my_utils.utils import image_or_path_load

def compute_volume(mask: sitk.SimpleITK.Image or str):
    # mask is an sitk image
    # used to compute the volume in ml for foreground
    mask = image_or_path_load(mask)

    sp = mask.GetSpacing()
    vol_per_vox = sp[0] * sp[1] * sp[2]

    m = sitk.GetArrayFromImage(mask)
    voxels = m.sum()
    # volume in ml
    tot_volume = vol_per_vox * voxels / 1000
    return tot_volume

# def compute_binary_hausdorff(gt_bin: sitk.Image, pred_bin: sitk.Image):
#     # Return None if either mask empty (SimpleITK would return 0, which is misleading)
#     stats = sitk.StatisticsImageFilter()
#     stats.Execute(gt_bin)
#     gt_nonzero = stats.GetSum() > 0
#     stats.Execute(pred_bin)
#     pred_nonzero = stats.GetSum() > 0
#     if not (gt_nonzero and pred_nonzero):
#         return None
#     hd = sitk.HausdorffDistanceImageFilter()
#     hd.Execute(gt_bin, pred_bin)
#     return float(hd.GetHausdorffDistance()) # hd_filter.GetAverageHausdorffDistance()

def compute_binary_hausdorff(gt_bin: sitk.Image, pred_bin: sitk.Image):
    """
    Compute Hausdorff distance, 95th percentile Hausdorff distance (HD95),
    and average Hausdorff distance between two binary masks.

    Parameters:
        gt_bin (sitk.Image): Ground truth binary mask.
        pred_bin (sitk.Image): Predicted binary mask.

    Returns:
        dict: A dictionary containing 'Hausdorff', 'HD95', and 'AvgHausdorff' distances.
              Returns None if either mask is empty.
    """
    # Check if either mask is empty
    stats = sitk.StatisticsImageFilter()
    stats.Execute(gt_bin)
    gt_nonzero = stats.GetSum() > 0
    stats.Execute(pred_bin)
    pred_nonzero = stats.GetSum() > 0
    if not (gt_nonzero and pred_nonzero):
        return None

    # Compute Hausdorff distance
    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(gt_bin, pred_bin)
    hausdorff = float(hd_filter.GetHausdorffDistance())
    avg_hausdorff = float(hd_filter.GetAverageHausdorffDistance())

    # Compute HD95
    distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_bin, squaredDistance=False, useImageSpacing=True))
    gt_surface = sitk.LabelContour(gt_bin)
    distances = sitk.GetArrayFromImage(distance_map * sitk.Cast(gt_surface, sitk.sitkFloat32)).flatten()
    hd95 = float(np.percentile(distances[distances > 0], 95)) if distances.size > 0 else None

    return {
        'Hausdorff': hausdorff,
        'HD95': hd95,
        'AHD': avg_hausdorff
    }

def np_dice(y_true,y_pred,add=1e-6):
	return (2*(y_true*y_pred).sum()+add)/(y_true.sum()+y_pred.sum()+add)

def compare_masks(pred_mask: sitk.Image | str, gt_mask: sitk.Image | str, compute_hausdorff=False) -> dict:
    """
    Compare a predicted and ground truth binary mask using Dice, Hausdorff, TPR, FPR, PPV, and NPV.

    Parameters:
        pred_mask (sitk.Image): Predicted binary mask. or str (path to mask)
        gt_mask (sitk.Image): Ground truth binary mask. or str (path to mask)
        compute_hausdorff (bool): Whether to compute Hausdorff distance (slower).

    Returns:
        dict: Metric name → value
    """

    pred_mask = image_or_path_load(pred_mask)
    gt_mask = image_or_path_load(gt_mask)

    # Ensure binary masks (0 or 1)
    pred_mask_bin = sitk.Cast(pred_mask > 0, sitk.sitkInt16)
    gt_mask_bin = sitk.Cast(gt_mask > 0, sitk.sitkInt16)

    pred_vol = compute_volume(pred_mask_bin)
    gt_vol = compute_volume(gt_mask_bin)

    # Hausdorff distance (only if requested)
    hausdorff = None
    if compute_hausdorff:
        hd_filter = sitk.HausdorffDistanceImageFilter()
        hd_filter.Execute(gt_mask_bin, pred_mask_bin)
        hausdorff = hd_filter.GetHausdorffDistance()

    # Convert masks to NumPy arrays
    y_true = sitk.GetArrayFromImage(gt_mask_bin).flatten()
    y_pred = sitk.GetArrayFromImage(pred_mask_bin).flatten()

    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Derived metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity / Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Dice coefficient
    num = 2*tp
    denom = 2*tp + fp + fn
    denom = 1e6 if denom == 0 else denom
    dice = num / denom

    results = {
        'Dice': dice,
        'TPR': tpr,
        'FPR': fpr,
        'PPV': ppv,
        'NPV': npv,
        'pred_volume_ml': pred_vol,
        'gt_volume_ml': gt_vol,
        'pred-gt_VD': pred_vol - gt_vol,
        'AVD': abs(pred_vol - gt_vol)
    }

    if compute_hausdorff:
        results['Hausdorff'] = hausdorff

    return results

def compare_multiclass_masks(
    pred_mask: sitk.Image | str,
    gt_mask: sitk.Image | str,
    roi_mask: sitk.Image | str = None, #ROI to adjust pred and gt (specific area for analyses)
    compute_hausdorff: bool = False,
    include_background: bool = False,
) -> dict:
    """
    Compare predicted vs. ground-truth *multiclass* masks.

    Returns:
        {
          'classes': [labels...],
          'per_class': {cls: {...metrics...}},
          'macro_avg': {...},
          'micro_avg': {...},
        }
    """
    pred = image_or_path_load(pred_mask)
    gt = image_or_path_load(gt_mask)
    roi = image_or_path_load(roi_mask) if roi_mask is not None else None

    # Align geometry if needed (simple check); assume already aligned in most pipelines
    if (pred.GetSize() != gt.GetSize()) or (pred.GetSpacing() != gt.GetSpacing()):
        raise ValueError("pred_mask and gt_mask must have same size and spacing.")

    # Unique labels present in either mask
    pred_arr = sitk.GetArrayFromImage(pred)
    gt_arr = sitk.GetArrayFromImage(gt)
    if roi is not None:
        roi_arr = sitk.GetArrayFromImage(roi).astype(pred_arr.dtype)
        pred_arr = pred_arr * roi_arr
        gt_arr = gt_arr * roi_arr

    labels = np.unique(np.concatenate([np.unique(pred_arr), np.unique(gt_arr)]))
    if not include_background:
        labels = labels[labels != 0]

    # Build one-vs-rest stacks for confusion matrices
    # shape: (n_voxels, ), then binarize per class
    y_true_stacked = []
    y_pred_stacked = []
    used_labels = []
    for c in labels:
        y_true_stacked.append((gt_arr == c).astype(np.uint8).ravel())
        y_pred_stacked.append((pred_arr == c).astype(np.uint8).ravel())
        used_labels.append(int(c))

    if len(used_labels) == 0:
        return {
            'classes': [],
            'per_class': {},
            'macro_avg': {},
            'micro_avg': {}
        }

    Y_true = np.vstack(y_true_stacked).T  # (N, C)
    Y_pred = np.vstack(y_pred_stacked).T  # (N, C)

    # Class-wise confusion matrices: [[tn, fp], [fn, tp]] per class
    cms = multilabel_confusion_matrix(Y_true, Y_pred)  # shape (C, 2, 2)

    per_class = {}
    # Sums for micro-average
    TP_sum = FP_sum = FN_sum = TN_sum = 0

    for idx, c in enumerate(used_labels):
        tn, fp, fn, tp = cms[idx].ravel()
        TP_sum += tp; FP_sum += fp; FN_sum += fn; TN_sum += tn

        # Derived metrics (protect zero divisions)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        # Volumes (mL) for this class
        gt_bin = sitk.Cast(sitk.GetImageFromArray((gt_arr == c).astype(np.uint8)), sitk.sitkUInt8)
        gt_bin.CopyInformation(gt)
        pred_bin = sitk.Cast(sitk.GetImageFromArray((pred_arr == c).astype(np.uint8)), sitk.sitkUInt8)
        pred_bin.CopyInformation(pred)

        gt_vol = compute_volume(gt_bin)
        pred_vol = compute_volume(pred_bin)

        metrics = {
            'Dice': float(dice),
            'TPR': float(tpr),
            'FPR': float(fpr),
            'PPV': float(ppv),
            'NPV': float(npv),
            'pred_volume_ml': float(pred_vol),
            'gt_volume_ml': float(gt_vol),
            'pred-gt_VD': float(pred_vol - gt_vol),
            'AVD': abs(pred_vol - gt_vol)
        }

        if compute_hausdorff:
            hd = compute_binary_hausdorff(gt_bin, pred_bin)
            if hd is not None:
                for m, number in hd.items():
                    metrics[m] = number

        per_class[int(c)] = metrics

    # Macro average (unweighted mean across classes)
    def _macro(key):
        vals = [per_class[c][key] for c in per_class.keys()]
        return float(np.mean(vals)) if len(vals) else 0.0

    macro_avg = {
        'Dice': _macro('Dice'),
        'TPR': _macro('TPR'),
        'FPR': _macro('FPR'),
        'PPV': _macro('PPV'),
        'NPV': _macro('NPV'),
    }
    if compute_hausdorff:
        # Exclude Nones from averaging
        for hd_measure in ['Hausdorff', 'HD95', 'AHD']:
            vals = [v[hd_measure] for v in per_class.values() if v.get(hd_measure) is not None]
            macro_avg[hd_measure] = float(np.mean(vals)) if len(vals) else None

    # Micro average (pool TP/FP/FN/TN over classes)
    dice_micro = (2 * TP_sum) / (2 * TP_sum + FP_sum + FN_sum) if (2 * TP_sum + FP_sum + FN_sum) > 0 else 0.0
    tpr_micro = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0.0
    fpr_micro = FP_sum / (FP_sum + TN_sum) if (FP_sum + TN_sum) > 0 else 0.0
    ppv_micro = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0.0
    npv_micro = TN_sum / (TN_sum + FN_sum) if (TN_sum + FN_sum) > 0 else 0.0

    micro_avg = {
        'Dice': float(dice_micro),
        'TPR': float(tpr_micro),
        'FPR': float(fpr_micro),
        'PPV': float(ppv_micro),
        'NPV': float(npv_micro),
    }

    return {
        'classes': used_labels,
        'per_class': per_class,
        'macro_avg': macro_avg,
        'micro_avg': micro_avg,
    }

