import numpy as np
import pandas as pd

def block_shuffle_1d(x, block_size, rng):
    """
    x: shape (T,)
    Returns x with contiguous blocks permuted.
    """
    T = len(x)
    n_blocks = int(np.ceil(T / block_size))
    pad_len = n_blocks * block_size - T
    # pad to full blocks
    if pad_len > 0:
        x_pad = np.pad(x, (0, pad_len), mode="edge")
    else:
        x_pad = x
    blocks = x_pad.reshape(n_blocks, block_size)  # (B, block_size)
    perm = rng.permutation(n_blocks)
    x_shuf = blocks[perm].reshape(-1)[:T]
    return x_shuf

def permutation_importance_global_sequence(
    model,
    X_val,                  # shape: (N, T, F)
    y_val,                  # shape: (N, T) or model-compatible target
    feature_names=None,
    n_repeats=5,
    batch_size=128,
    sample_weight=None,     # optional mask/weights, shape usually (N, T)
    shuffle_mode="sequence",# "sequence" or "per_timestep"
    random_state=42,
    block_size=32,
):
    """
    Global permutation importance for sequence models (GRU/LSTM), no retraining.
    Importance = mean(permuted_loss - baseline_loss) across repeats.
    """

    rng = np.random.default_rng(random_state)

    # 1) Baseline validation loss (trained model, no retraining)
    baseline = model.evaluate(
        X_val, y_val,
        sample_weight=sample_weight,
        batch_size=batch_size,
        verbose=0,
        return_dict=True
    )
    baseline_loss = baseline["loss"]

    n_features = X_val.shape[2]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    results = []

    # 2) Permute one feature at a time, recompute loss
    for j in range(n_features):
        deltas = []
        print(f"Permutation feature {feature_names[j]}")
        for _ in range(n_repeats):
            Xp = X_val.copy()

            if shuffle_mode == "sequence":
                # Shuffle entire trajectory of feature j across samples:
                # preserves within-trial temporal structure for that feature.
                perm_idx = rng.permutation(X_val.shape[0])
                Xp[:, :, j] = X_val[perm_idx, :, j]

            elif shuffle_mode == "per_timestep":
                # Shuffle feature j across samples independently at each time t.
                for t in range(X_val.shape[1]):
                    perm_idx = rng.permutation(X_val.shape[0])
                    Xp[:, t, j] = X_val[perm_idx, t, j]
            elif shuffle_mode == "within_sample":
                # independently shuffle timesteps within each sample for feature j
                for n in range(X_val.shape[0]):
                    Xp[n, :, j] = rng.permutation(X_val[n, :, j])
            elif shuffle_mode == "within_sample_block":
                # Shuffle blocks of timesteps within each sample for feature j
                for n in range(X_val.shape[0]):
                    Xp[n, :, j] = block_shuffle_1d(
                        X_val[n, :, j],
                        block_size=block_size,   # tune this
                        rng=rng
                    )                    
            else:
                raise ValueError("shuffle_mode must be 'sequence' or 'per_timestep'")

            perm_eval = model.evaluate(
                Xp, y_val,
                sample_weight=sample_weight,
                batch_size=batch_size,
                verbose=0,
                return_dict=True
            )
            perm_loss = perm_eval["loss"]
            print(f"Permutation loss: {perm_loss}")
            deltas.append(perm_loss - baseline_loss)  # degradation

        results.append({
            "feature": feature_names[j],
            "importance_mean": float(np.mean(deltas)),
            "importance_std": float(np.std(deltas)),
            "baseline_loss": float(baseline_loss),
            "permuted_loss_mean": float(baseline_loss + np.mean(deltas)),
        })

    ranking = pd.DataFrame(results).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return ranking