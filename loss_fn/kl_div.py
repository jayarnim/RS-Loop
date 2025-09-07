from torch.distributions import kl_divergence


def kl_div(posterior, prior, padding=None):
    kl_tensor = kl_divergence(posterior, prior)

    if padding is not None:
        padding = padding.to(kl_tensor.device)
        num_valid = padding.numel() - padding.sum()
        kl_tensor_masked = kl_tensor.masked_fill(padding, 0.0)
        kl_sum = kl_tensor_masked.sum()
        return kl_sum / (num_valid + 1e-8)

    else:
        return kl_tensor.mean()