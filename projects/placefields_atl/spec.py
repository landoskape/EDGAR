"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [X, Y]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(X, param1=default, ...) and param_est_v1(X, Y)
- model_v2(X, param1=default, ...) and param_est_v2(X, Y)
- params are positional keyword args with defaults

LOSS FUNCTION:
- loss_fn(Y_pred, Y_true) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(X, Y, model_list, params_list)
"""
import numpy as np
from typing import Tuple
from src.data_structures import Inputs, Outputs
from matplotlib import pyplot as plt

from vrAnalysis.database import get_database
from vrAnalysis.helpers import edge2center, reliability_loo
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import get_placefield, get_frame_behavior
sessiondb = get_database("vrSessions")

# ========================
# 1. DATA
# ========================

def load_and_process_data(
    data_path: str,
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    random_seed: int = 42,
) -> Tuple[Inputs, Outputs]:
    """
    Load and preprocess data and return canonical Inputs/Outputs.
    """
    spks_type = "oasis"
    session = sessiondb.iter_sessions(imaging=True, session_params=dict(spks_type=spks_type))[40] # 40 is a "good" session

    env_stats = session.env_stats
    best_env = max(env_stats, key=env_stats.get)

    # Get frame_behavior of session
    frame_behavior = get_frame_behavior(session)

    # Filter frame behavior
    speed_threshold = 1.0
    idx_fast = frame_behavior.speed > speed_threshold
    idx_valid_frames = frame_behavior.valid_frames()
    idx_keep = idx_valid_frames & idx_fast
    frame_behavior = frame_behavior.filter(idx_keep)
    idx_to_spks = np.where(idx_keep)[0]

    num_bins = 100
    dist_edges = np.linspace(0, session.env_length[0], num_bins + 1)

    # Choose neurons based on good placefield properties
    spks = session.spks[:, session.idx_rois]
    reliability_threshold = 0.3
    fraction_active_threshold = 0.05
    _all_trials = get_placefield(spks, frame_behavior, dist_edges, average=False, idx_to_spks=idx_to_spks, use_fast_sampling=True, session=session).filter_by_environment(best_env)
    _pf_data = np.transpose(_all_trials.placefield, (2, 0, 1))
    _reliable = reliability_loo(_pf_data)
    _fraction_active = FractionActive.compute(
        _pf_data,
        activity_axis=2,
        fraction_axis=1,
        activity_method="rms",
        fraction_method="participation",
    )
    _idx_reliable = _reliable > reliability_threshold
    _idx_fraction_active = _fraction_active > fraction_active_threshold
    idx_keep = _idx_reliable & _idx_fraction_active
    spks = spks[:, idx_keep]
        
    # Get placefield data
    placefield = get_placefield(spks, frame_behavior, dist_edges, average=False, idx_to_spks=idx_to_spks, use_fast_sampling=True, session=session)
    placefield = placefield.filter_by_environment(best_env)
    placefield_data = placefield.placefield.transpose(2, 1, 0) # (num_neurons, num_positions, num_trials)

    # Normalize by std in placefield data
    _std_neuron = np.std(placefield_data, axis=(1, 2), keepdims=True)
    _std_neuron[_std_neuron == 0] = 1
    placefield_data /= _std_neuron

    # placefield.placefield is (num_trials, num_positions, num_neurons)
    dist_centers = edge2center(dist_edges)[None, :, None]
    
    # For some reason X needs to match shape of Y
    num_samples = placefield_data.shape[0]
    num_trials = placefield_data.shape[2]
    X = np.repeat(np.repeat(dist_centers, num_samples, axis=0), num_trials, axis=2)
    Y = placefield_data # (num_neurons, num_positions, num_trials)

    # We actually only have one feature, position, so reshape to be (num_samples, 1, num_trials * num_positions)
    X = X.reshape(num_samples, 1, num_trials * X.shape[1])
    Y = Y.reshape(num_samples, 1, num_trials * Y.shape[1])
    return X, Y


def train_test_split(
    X: Inputs,
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train sample indices and train trial indices.
    """
    spks_type = "oasis"
    session = sessiondb.iter_sessions(imaging=True, session_params=dict(spks_type=spks_type))[40] # 40 is a "good" session

    env_stats = session.env_stats
    best_env = max(env_stats, key=env_stats.get)

    # Get frame_behavior of session
    frame_behavior = get_frame_behavior(session)

    # Filter frame behavior
    speed_threshold = 1.0
    idx_fast = frame_behavior.speed > speed_threshold
    idx_valid_frames = frame_behavior.valid_frames()
    idx_keep = idx_valid_frames & idx_fast
    frame_behavior = frame_behavior.filter(idx_keep)
    idx_to_spks = np.where(idx_keep)[0]

    num_bins = 100
    dist_edges = np.linspace(0, session.env_length[0], num_bins + 1)

    # Choose neurons based on good placefield properties
    spks = session.spks[:, session.idx_rois]
    reliability_threshold = 0.3
    fraction_active_threshold = 0.05
    _all_trials = get_placefield(spks, frame_behavior, dist_edges, average=False, idx_to_spks=idx_to_spks, use_fast_sampling=True, session=session).filter_by_environment(best_env)
    _pf_data = np.transpose(_all_trials.placefield, (2, 0, 1))
    _reliable = reliability_loo(_pf_data)
    _fraction_active = FractionActive.compute(
        _pf_data,
        activity_axis=2,
        fraction_axis=1,
        activity_method="rms",
        fraction_method="participation",
    )
    _idx_reliable = _reliable > reliability_threshold
    _idx_fraction_active = _fraction_active > fraction_active_threshold
    idx_keep = _idx_reliable & _idx_fraction_active
    spks = spks[:, idx_keep]
        
    # Get placefield data
    placefield = get_placefield(spks, frame_behavior, dist_edges, average=False, idx_to_spks=idx_to_spks, use_fast_sampling=True, session=session)
    placefield = placefield.filter_by_environment(best_env)
    placefield_data = placefield.placefield.transpose(2, 1, 0) # (num_neurons, num_positions, num_trials)
    
    # For some reason X needs to match shape of Y
    num_trials = placefield_data.shape[2]
    T = np.repeat(np.arange(num_trials)[None, :], len(dist_edges) - 1, axis=0)
    T = T.reshape(-1)

    # Cross validate obs by trials
    n_samples, _, n_obs = X.shape
    assert n_obs == T.shape[0], "Inferred trial array not match observations in X"
    assert n_samples >= 2, "Need at least 2 samples for model optimization/eval"
    assert n_obs >= 2, "Need at least 2 observations for parameter optimization/eval"

    rng = np.random.default_rng(random_seed)
    train_samples = rng.choice(np.arange(n_samples), n_samples // 2, replace=False)
    train_trials = rng.choice(np.arange(num_trials), num_trials // 2, replace=False)
    train_obs = np.isin(T, train_trials)
    return train_samples, train_obs


# ========================
# 2. SEED MODELS
# ========================

def model_v1(X, baseline=0.0, amplitude=1.0, mu=50.0, sigma=10.0):
    """
    Gaussian placefield model.

    Independent variable:
    X : shape (n_features=1, n_obs) where n_obs = num_positions * num_trials.
        X[0] gives the flat position array (positions repeat across trials).

    Parameters
    ----------
    baseline : float
        Baseline firing rate.
    amplitude : float
        Peak above baseline.
    mu : float
        Preferred position (cm).
    sigma : float
        Tuning width (cm, std of Gaussian).

    Returns
    -------
    np.ndarray, shape (n_obs,)
    """
    pos = X[0]  # (n_obs,)
    baseline = np.clip(baseline, 0, None)
    amplitude = np.clip(amplitude, 0, None)
    sigma = np.clip(sigma, 1e-3, None)

    return baseline + amplitude * np.exp(-0.5 * ((pos - mu) / sigma) ** 2)


def _per_pos_mean(X, Y):
    """Average Y over trials for each unique position.

    Parameters
    ----------
    X : array, shape (n_features=1, n_obs)
    Y : array, shape (1, n_obs) or (n_obs,)

    Returns
    -------
    unique_pos : np.ndarray, shape (num_positions,)
    mean_rate  : np.ndarray, shape (num_positions,)
    """
    pos = np.asarray(X[0])
    rates = np.asarray(Y[0] if np.ndim(Y) == 2 else Y)
    unique_pos, inv = np.unique(pos, return_inverse=True)
    counts = np.bincount(inv)
    mean_rate = np.bincount(inv, weights=rates.astype(float)) / counts
    return unique_pos, mean_rate


def param_est_v1(X, Y):
    """
    Heuristic parameter estimator for the Gaussian placefield model.

    Returns
    -------
    np.ndarray
        [baseline, amplitude, mu, sigma]
    """
    pos, mean_rate = _per_pos_mean(X, Y)
    bin_width = float((pos[-1] - pos[0]) / len(pos))

    baseline = float(np.min(mean_rate))
    amplitude = float(np.max(mean_rate) - baseline)

    if amplitude < 1e-6:
        return np.array([baseline, amplitude,
                         float(np.mean(pos)), float((pos[-1] - pos[0]) / 4)])

    mu = float(pos[np.argmax(mean_rate)])

    # FWHM -> sigma
    half_max = baseline + amplitude / 2.0
    fwhm = float(np.sum(mean_rate >= half_max)) * bin_width
    sigma = max(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), bin_width)

    return np.array([baseline, amplitude, mu, sigma])


def model_v2(X, baseline=0.0, amplitude=1.0, center=50.0, width=20.0, sharpness=1.0):
    """
    Smooth square-wave (logistic-edge) placefield model.

    Independent variable:
    X : shape (n_features=1, n_obs) where n_obs = num_positions * num_trials.
        X[0] gives the flat position array (positions repeat across trials).

    Parameters
    ----------
    baseline : float
        Baseline firing rate.
    amplitude : float
        Firing rate above baseline inside the field.
    center : float
        Center of the place field (cm).
    width : float
        Full width of the place field (cm).
    sharpness : float
        Steepness of the logistic edges (1/cm).

    Returns
    -------
    np.ndarray, shape (n_obs,)
    """
    pos = X[0]  # (n_obs,)
    baseline = np.clip(baseline, 0, None)
    amplitude = np.clip(amplitude, 0, None)
    width = np.clip(width, 1e-3, None)
    sharpness = np.clip(sharpness, 0.1, None)

    left_edge = center - width / 2.0
    right_edge = center + width / 2.0

    # Two opposing sigmoids form a smooth top-hat
    left_rise = 1.0 / (1.0 + np.exp(-sharpness * (pos - left_edge)))
    right_fall = 1.0 / (1.0 + np.exp(-sharpness * (right_edge - pos)))

    return baseline + amplitude * left_rise * right_fall


def param_est_v2(X, Y):
    """
    Heuristic parameter estimator for the smooth square-wave placefield model.

    Returns
    -------
    np.ndarray
        [baseline, amplitude, center, width, sharpness]
    """
    pos, mean_rate = _per_pos_mean(X, Y)
    bin_width = float((pos[-1] - pos[0]) / len(pos))

    baseline = float(np.min(mean_rate))
    amplitude = float(np.max(mean_rate) - baseline)

    if amplitude < 1e-6:
        return np.array([baseline, amplitude,
                         float(np.mean(pos)), float((pos[-1] - pos[0]) / 4),
                         1.0])

    half_max = baseline + amplitude / 2.0
    active_pos = pos[mean_rate >= half_max]

    if len(active_pos) == 0:
        center = float(pos[np.argmax(mean_rate)])
        width = bin_width
    else:
        center = float(np.mean(active_pos))
        width = float(active_pos[-1] - active_pos[0]) + bin_width

    # For a logistic: peak gradient ≈ sharpness * amplitude / 4
    max_grad = float(np.max(np.abs(np.gradient(mean_rate, pos))))
    sharpness = max(4.0 * max_grad / amplitude, 0.1)

    return np.array([baseline, amplitude, center, width, sharpness])


# ========================
# 3. LOSS
# ========================

def loss_fn(Y_pred, Y_true):
    return 10 * (Y_true - Y_pred) ** 2


# ========================
# 4. DIAGNOSTICS
# ========================

def plot_model_fits(
    X,
    Y,
    programs_list,
    X_eval=None,
    save_path="",
    labels=("model_v1", "model_v2"),
):
    """
    Plot mean observed placefield and model predictions for 9 random neurons.

    Parameters
    ----------
    X : array-like or Inputs, shape (n_samples, num_positions, num_trials)
    Y : array-like or Outputs, shape (n_samples, num_positions, num_trials)
    programs_list : list[dict]  keys: 'model', 'params', 'losses'
    X_eval : ignored (position grid is taken directly from X)
    save_path : str
    """
    if save_path == "":
        raise ValueError("Please provide a save path for the plot")

    x_arr = _to_array3d(X)
    y_arr = _to_array3d(Y)
    n_samples = x_arr.shape[0]

    if len(programs_list) == 1:
        colours = ["tab:red"]
    elif len(programs_list) == 2:
        colours = ["tab:green", "tab:red"]
    else:
        colours = ["tab:orange", "tab:green", "tab:red"]

    n_show = min(9, n_samples)
    idx = np.random.default_rng().choice(n_samples, size=n_show, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.reshape(3, 3)

    for i in range(9):
        ax = axes[i // 3, i % 3]
        if i >= n_show:
            ax.axis("off")
            continue

        s = idx[i]
        pos, mean_rate = _per_pos_mean(x_arr[s], y_arr[s])

        ax.plot(pos, mean_rate, color="deepskyblue", linewidth=3,
                label="Mean observed", alpha=0.9)

        for j, program in enumerate(programs_list):
            model = program["model"]
            params = program["params"][s]
            y_pred = np.asarray(model(x_arr[s], *params))
            if y_pred.ndim == 2:
                y_pred = y_pred[0]  # (n_obs,)
            _, y_pred_mean = _per_pos_mean(x_arr[s], y_pred)

            label = labels[j] if labels is not None and j < len(labels) else f"Model {j+1}"
            if "losses" in program:
                label += f" (loss={program['losses'][s]:.3f})"
            ax.plot(pos, y_pred_mean,
                    color=colours[j % len(colours)], linewidth=2.5,
                    label=label, alpha=0.85)

        ax.set_title(f"Neuron {s}")
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Firing rate")
        ax.legend(fontsize=10)

    mean_loss_parts = []
    for j, program in enumerate(programs_list):
        if "losses" in program and np.size(program["losses"]) > 0:
            mean_loss_parts.append(f"Model {j+1}: {np.mean(program['losses']):.3f}")
        else:
            mean_loss_parts.append(f"Model {j+1}: n/a")
    summary = "  |  ".join(mean_loss_parts) if mean_loss_parts else "n/a"
    plt.suptitle(f"Placefield Model Fits\n{summary}", fontsize=22)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)


# ========================
# 4. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================

def _to_array3d(obj) -> np.ndarray:
    """
    Convert Inputs/Outputs/ndarray-like objects to a 3D ndarray.
    """
    if hasattr(obj, "to_tensor"):
        return np.asarray(obj.to_tensor())
    arr = np.asarray(obj)
    if arr.ndim == 2:
        return arr[:, np.newaxis, :]
    return arr


def compute_binned_means(theta, y, n_bins=20, domain=(-1.0, 1.0)):
    """
    Compute binned means of y over theta for visualization.

    Returns
    -------
    x_eval : np.ndarray
        Bin centers.
    y_mean : np.ndarray
        Mean y per bin.
    """
    edges = np.linspace(domain[0], domain[1], n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    idx = np.digitize(theta, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    sums = np.bincount(idx, weights=y, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    mean = sums / (counts + 1e-8)
    return centres, mean


def compute_binned_means_on_eval(theta, y, x_eval):
    """
    Compute binned means of y at a provided evaluation grid.
    """
    x_eval = np.asarray(x_eval).reshape(-1)
    if x_eval.size == 0:
        return x_eval
    if x_eval.size == 1:
        return np.array([float(np.mean(y))])

    edges = np.empty(x_eval.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x_eval[:-1] + x_eval[1:])
    edges[0] = x_eval[0] - 0.5 * (x_eval[1] - x_eval[0])
    edges[-1] = x_eval[-1] + 0.5 * (x_eval[-1] - x_eval[-2])

    idx = np.digitize(theta, edges) - 1
    y_mean = np.full(x_eval.size, np.nan, dtype=float)
    for i in range(x_eval.size):
        vals = y[idx == i]
        if vals.size > 0:
            y_mean[i] = float(np.mean(vals))

    valid = np.isfinite(y_mean)
    if np.any(valid):
        y_mean = np.interp(
            x_eval,
            x_eval[valid],
            y_mean[valid],
            left=float(y_mean[valid][0]),
            right=float(y_mean[valid][-1]),
        )
    else:
        y_mean = np.zeros_like(x_eval, dtype=float)
    return y_mean