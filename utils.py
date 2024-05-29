def denormalize_targets(targets, original_means, original_stds):
    return targets * original_stds + original_means
