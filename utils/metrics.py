import torch


def confusion_matrix(preds, labels, num_classes):
    hist = (
        torch.bincount(
            num_classes * labels + preds,
            minlength=num_classes ** 2,
        )
        .reshape(num_classes, num_classes)
        .float()
    )
    return hist


def compute_IoU_from_cmatrix(hist, ignore_index=None):
    """Computes the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        m_IoU, fw_IoU, and matrix IoU
    """
    if ignore_index is not None:
        hist[ignore_index] = 0.0
    intersection = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - intersection
    IoU = intersection.float() / union.float()
    IoU[union == 0] = 1.0
    if ignore_index is not None:
        IoU = torch.cat((IoU[:ignore_index], IoU[ignore_index+1:]))
    m_IoU = torch.mean(IoU).item()
    fw_IoU = (
        torch.sum(intersection) / (2 * torch.sum(hist) - torch.sum(intersection))
    ).item()
    return m_IoU, fw_IoU, IoU


def compute_IoU(preds, labels, num_classes, ignore_index=None):
    """Computes the Intersection over Union (IoU)."""
    hist = confusion_matrix(preds, labels, num_classes)
    return compute_IoU_from_cmatrix(hist, ignore_index)
