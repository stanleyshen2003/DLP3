import numpy as np
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()
    intersection = np.sum(pred_mask * gt_mask)
    area_pred = np.sum(pred_mask)
    area_gt = np.sum(gt_mask)
    return 2 * intersection / (area_pred + area_gt)
