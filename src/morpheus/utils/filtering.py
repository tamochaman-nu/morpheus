import torch.nn.functional as F


def median_filter(frame, kernel_size=3):
    """
    Applies a median filter to the depth map.

    Args:
        frame: A Frame object containing:
            - depth_b1hw: torch.Tensor of shape (B, 1, H, W)
        kernel_size: Size of the median filter kernel (default is 3)

    Returns:
        filtered_depth: torch.Tensor of shape (B, 1, H, W), the filtered depth map.
    """
    p = frame.depth_b1hw  # (B, 1, H, W)
    padding = kernel_size // 2

    # Pad the depth map to handle borders
    p_padded = F.pad(p, (padding, padding, padding, padding), mode="reflect")

    # Extract sliding local blocks (unfold)
    unfolded = p_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # (B, 1, H, W, k, k)

    # Reshape to (B, 1, H, W, k*k)
    unfolded = unfolded.contiguous().view(*unfolded.size()[:4], -1)

    # Compute the median along the last dimension (kernel window)
    median = unfolded.median(dim=-1)[0]  # (B, 1, H, W)

    return median
