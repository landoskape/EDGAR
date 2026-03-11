"""Rewind: interactive inspection of a finished EDGAR run.

Usage (notebook)::

    import nest_asyncio; nest_asyncio.apply()
    from src.rewind import Rewind

    rw = Rewind().load_previous_run("program_databases/03-04/14-22-01")
    rw.list_models()

    rw.go_to("D18").estimate_params().optimize_params().evaluate_test().plot_fits()
    rw.print_state()
    rw.show_code("jax")
"""

import asyncio
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

from .monitoring.io import (
    assign_node_labels,
    load_generation_log,
    record_key,
)
from .utils import str_to_func
from .hypothesis_engine import (
    compute_initial_params,
    compute_default_params,
    objective,
)


def _make_node_id(iteration, island, batch):
    return f"{iteration}_{island}_{batch}"


def backfill_run_config(run_dir: str, config_path: str) -> None:
    """Write a ``run_config.json`` into an existing run directory.

    Use this for runs made before Rewind was added, assuming the spec and
    config have not changed since the run was produced.

    Args:
        run_dir: Path to the run output directory (must contain
            ``program_generation_log.jsonl``).
        config_path: Path to the project config file (e.g.
            ``"projects/synthetic_data/config.yaml"``).  Merged with
            ``projects/config_default.yaml`` the same way ``run.py`` does.
    """
    import yaml

    run_dir = os.path.abspath(run_dir)
    out_path = os.path.join(run_dir, "run_config.json")

    if os.path.isfile(out_path):
        print(f"[backfill] run_config.json already exists at {out_path} — skipping.")
        return

    if not os.path.isfile(os.path.join(run_dir, "program_generation_log.jsonl")):
        raise FileNotFoundError(
            f"No program_generation_log.jsonl found in {run_dir}. "
            "Is this a valid run directory?"
        )

    # Resolve config relative to the project root (same logic as run.py)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.abspath(
        os.path.join(project_root, config_path)
        if not os.path.isabs(config_path)
        else config_path
    )

    default_config_path = os.path.join(project_root, "projects", "config_default.yaml")
    with open(default_config_path) as f:
        config = yaml.safe_load(f) or {}
    with open(config_path) as f:
        experiment_config = yaml.safe_load(f) or {}

    # Deep-merge (override with experiment config)
    def _deep_merge(base, override):
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    config = _deep_merge(config, experiment_config)
    task_name = config.get("task", "")

    # Infer date/time stamps from the directory path (MM-DD/HH-MM-SS layout)
    parts = os.path.normpath(run_dir).split(os.sep)
    date_stamp = parts[-2] if len(parts) >= 2 else ""
    time_stamp = parts[-1]

    run_config_data = {
        "task_name": task_name,
        "spec_path": f"projects/{task_name}/spec.py" if task_name else "",
        "config": config,
        "date_stamp": date_stamp,
        "time_stamp": time_stamp,
    }

    with open(out_path, "w") as f:
        json.dump(run_config_data, f, indent=2, default=str)

    print(f"[backfill] Wrote run_config.json → {out_path}")
    print(f"           task_name : {task_name!r}")
    print(f"           config    : {config_path}")


class Rewind:
    """Interactive inspector for a finished EDGAR run."""

    def __init__(self):
        self._run_dir = None
        self._records = None        # raw JSONL records
        self._label_map = None      # label string -> record dict
        self._current_record = None
        self._current_label = None

        # from run_config.json
        self.config = None
        self.task_name = None

        # data
        self.X = None               # full inputs (n_samples, n_features, n_trials)
        self.Y = None               # full outputs
        self.X_split = None         # [train_sample_split, test_sample_split]
        self.Y_split = None         # each split is [train_trials_data, test_trials_data]
        self.loss_fn = None
        self.plot_fn = None
        self.X_eval = None

        # current model state
        self.model_np = None
        self.model_jax = None
        self.param_estimator = None
        self.model_code_np = None
        self.model_code_jax = None
        self.param_est_code = None

        self.initial_params = None
        self.optimized_params = None
        self.train_loss = None
        self.test_loss = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_previous_run(self, run_dir: str) -> "Rewind":
        """Load a finished run from *run_dir*.

        Reads run_config.json, re-imports the spec, reloads data, and assigns
        family-tree labels.  If run_config.json is missing (old run), only the
        JSONL records are loaded; data-dependent methods will raise.

        Args:
            run_dir: Path to the run output directory (contains
                ``program_generation_log.jsonl``).
        """
        run_dir = os.path.abspath(run_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        self._run_dir = run_dir

        # ---- load JSONL records ----
        log_path = os.path.join(run_dir, "program_generation_log.jsonl")
        if not os.path.isfile(log_path):
            raise FileNotFoundError(
                f"No program_generation_log.jsonl found in {run_dir}"
            )
        self._records = load_generation_log(log_path)
        self._build_label_map()

        # ---- load run_config.json ----
        config_path = os.path.join(run_dir, "run_config.json")
        if not os.path.isfile(config_path):
            print(
                f"[Rewind] run_config.json not found in {run_dir}.\n"
                "This run was made before Rewind was added.  "
                "list_models / go_to / show_code will work; "
                "estimate_params / optimize_params / plot_fits require data."
            )
            return self

        with open(config_path) as f:
            run_config = json.load(f)

        self.task_name = run_config.get("task_name", "")
        self.config = run_config.get("config", {})

        if not self.task_name:
            print("[Rewind] task_name missing from run_config.json — data not loaded.")
            return self

        self._load_spec_and_data()
        return self

    def _build_label_map(self):
        """Assign family-tree labels and build label -> record mapping."""
        # build synthetic seed records (not in JSONL, but needed for labelling)
        seed_records = [
            {
                "iteration_number": -1,
                "birth_island": -1,
                "batch_index": i,
                "is_seed": True,
            }
            for i in range(2)
        ]
        all_records = seed_records + self._records

        node_label_map = assign_node_labels(all_records)  # node_id -> label

        # invert: label -> record
        self._label_map = {}
        for rec in self._records:
            nid = _make_node_id(*record_key(rec))
            label = node_label_map.get(nid)
            if label:
                self._label_map[label] = rec

    def _load_spec_and_data(self):
        """Import the spec module and reconstruct X, Y, and the train/test split."""
        spec_module_path = f"projects.{self.task_name}.spec"
        try:
            spec = importlib.import_module(spec_module_path)
        except ModuleNotFoundError:
            # Ensure the project root is on sys.path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            try:
                spec = importlib.import_module(spec_module_path)
            except ModuleNotFoundError as e:
                print(f"[Rewind] Could not import {spec_module_path}: {e}")
                print("[Rewind] Data not loaded — code-only mode.")
                return

        data_processing_params = self.config.get("data_processing_params", {})
        experiment_params = self.config.get("experiment_params", {})
        random_seed = experiment_params.get("random_seed", 42)

        # Load raw data
        try:
            raw = spec.load_and_process_data(**data_processing_params)
        except Exception as e:
            print(f"[Rewind] load_and_process_data failed: {e}")
            print("[Rewind] Data not loaded — code-only mode.")
            return

        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            from .data_structures import ensure_inputs, ensure_outputs
            inputs_obj = ensure_inputs(raw[0])
            outputs_obj = ensure_outputs(raw[1])
        else:
            print("[Rewind] Unexpected return from load_and_process_data.")
            return

        self.X = np.asarray(inputs_obj.to_tensor())
        self.Y = np.asarray(outputs_obj.to_tensor())

        # Apply train/test split
        try:
            from .data_structures import Inputs
            train_samples_raw, train_trials_raw = spec.train_test_split(
                Inputs.from_array(self.X), random_seed
            )
        except Exception as e:
            print(f"[Rewind] train_test_split failed: {e}")
            print("[Rewind] Data split not applied.")
            return

        n_samples = self.X.shape[0]
        n_trials = self.X.shape[2]
        train_samples = np.asarray(train_samples_raw).reshape(-1).astype(np.int64)
        train_trials = np.asarray(train_trials_raw).reshape(-1).astype(np.int64)
        test_samples = np.setdiff1d(np.arange(n_samples, dtype=np.int64), train_samples)
        test_trials = np.setdiff1d(np.arange(n_trials, dtype=np.int64), train_trials)

        X_train_train = self.X[train_samples][:, :, train_trials]
        X_train_test = self.X[train_samples][:, :, test_trials]
        X_test_train = self.X[test_samples][:, :, train_trials]
        X_test_test = self.X[test_samples][:, :, test_trials]

        Y_train_train = self.Y[train_samples][:, :, train_trials]
        Y_train_test = self.Y[train_samples][:, :, test_trials]
        Y_test_train = self.Y[test_samples][:, :, train_trials]
        Y_test_test = self.Y[test_samples][:, :, test_trials]

        # X_split[0] = train samples with [train_trials, test_trials]
        # X_split[1] = test  samples with [train_trials, test_trials]
        # (matches the X[0] / X[1] layout used in hypothesis_engine.objective)
        from .data_structures import Inputs, Outputs
        self.X_split = [
            [Inputs.from_array(X_train_train), Inputs.from_array(X_train_test)],
            [Inputs.from_array(X_test_train),  Inputs.from_array(X_test_test)],
        ]
        self.Y_split = [
            [Outputs.from_array(Y_train_train), Outputs.from_array(Y_train_test)],
            [Outputs.from_array(Y_test_train),  Outputs.from_array(Y_test_test)],
        ]

        # Loss and plot functions
        self.loss_fn = getattr(spec, "loss_fn", None)
        raw_plot_fn = getattr(spec, "plot_model_fits", None)
        if callable(raw_plot_fn):
            self.plot_fn = raw_plot_fn

        # Build evaluation grid for the train-sample set
        try:
            from .utils import build_evaluation_points
            n_bins = experiment_params.get("n_bins", 100)
            self.X_eval = build_evaluation_points(
                inputs=self.X_split[0][0],
                x_min=experiment_params.get("x_min"),
                x_max=experiment_params.get("x_max"),
                n_bins=n_bins,
            )
        except Exception as e:
            print(f"[Rewind] Could not build X_eval: {e}")

        print(
            f"[Rewind] Loaded {self.X.shape[0]} samples × {self.X.shape[2]} trials "
            f"from '{self.task_name}'."
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def list_models(self) -> pd.DataFrame:
        """Return a DataFrame of all models sorted by test_loss (then train_loss).

        Columns: label | iteration | island | batch | init_loss | train_loss | test_loss | n_params
        """
        if self._label_map is None:
            raise RuntimeError("No run loaded.  Call load_previous_run() first.")

        rows = []
        for label, rec in self._label_map.items():
            rows.append({
                "label": label,
                "iteration": rec.get("iteration_number"),
                "island": rec.get("birth_island"),
                "batch": rec.get("batch_index"),
                "init_loss": rec.get("initial_loss"),
                "train_loss": rec.get("train_loss"),
                "test_loss": rec.get("test_loss"),
                "n_params": rec.get("n_params"),
            })

        df = pd.DataFrame(rows)
        # Sort: prefer test_loss; fall back to train_loss
        if df["test_loss"].notna().any():
            df = df.sort_values("test_loss", na_position="last")
        else:
            df = df.sort_values("train_loss", na_position="last")
        return df.reset_index(drop=True)

    def go_to(self, label: str) -> "Rewind":
        """Navigate to a model by its family-tree label (e.g. "D18", "A1").

        Execs model_code_jax, model_code_numpy, and param_est_code.
        Resets params and losses.
        """
        if self._label_map is None:
            raise RuntimeError("No run loaded.  Call load_previous_run() first.")

        rec = self._label_map.get(label)
        if rec is None:
            available = sorted(self._label_map.keys())
            raise KeyError(
                f"Label '{label}' not found.  Available: {available[:20]}"
                + ("..." if len(available) > 20 else "")
            )

        self._current_record = rec
        self._current_label = label

        # Store raw code strings
        self.model_code_jax = rec.get("model_code_jax")
        self.model_code_np = rec.get("model_code_numpy")
        self.param_est_code = rec.get("param_est_code")

        # Exec model code — prefer JAX, fall back to numpy
        model_name = self._infer_model_name(self.model_code_jax or self.model_code_np)
        self.model_jax = str_to_func(self.model_code_jax, model_name) if self.model_code_jax else None
        self.model_np = str_to_func(self.model_code_np, model_name) if self.model_code_np else None
        if self.model_jax is None and self.model_np is not None:
            print(f"[Rewind] No JAX code for {label}; using numpy model (optimize_params may fail).")

        pe_name = self._infer_model_name(self.param_est_code)
        self.param_estimator = str_to_func(self.param_est_code, pe_name) if self.param_est_code else None

        # Reset computed state
        self.initial_params = None
        self.optimized_params = None
        self.train_loss = None
        self.test_loss = None

        return self

    @staticmethod
    def _infer_model_name(code: str | None) -> str:
        """Extract the first 'def <name>' from a code string."""
        if not code:
            return "model"
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("def "):
                name = line[4:].split("(")[0].strip()
                if name:
                    return name
        return "model"

    # ------------------------------------------------------------------
    # Parameter estimation & optimization
    # ------------------------------------------------------------------

    def estimate_params(self, use_param_estimator: bool = True) -> "Rewind":
        """Compute initial parameters using the param estimator or defaults.

        Args:
            use_param_estimator: If True, use the stored param_estimator;
                otherwise use model default values.
        """
        self._require_model()
        self._require_data()

        model = self.model_jax or self.model_np
        x_train = np.asarray(self.X_split[0][0].to_tensor())
        y_train = np.asarray(self.Y_split[0][0].to_tensor())

        if use_param_estimator and self.param_estimator is not None:
            self.initial_params = compute_initial_params(
                self.param_estimator, model, x_train, y_train
            )
        else:
            defaults = compute_default_params(model)
            if defaults is not None:
                self.initial_params = defaults.repeat(x_train.shape[0], axis=0)
            else:
                raise RuntimeError("compute_default_params returned None.")

        return self

    def optimize_params(
        self,
        max_iter: int = 1000,
        lr: float = 3e-3,
        fit_params: bool = True,
    ) -> "Rewind":
        """Optimize model parameters on the training-sample data.

        Calls ``objective`` on ``X_split[0]`` / ``Y_split[0]`` (train samples).
        Stores ``initial_params``, ``optimized_params``, and ``train_loss``.

        Args:
            max_iter: Maximum gradient steps.
            lr: Adam learning rate.
            fit_params: If False, only estimate params without gradient descent.
        """
        self._require_model()
        self._require_data()

        model = self.model_jax or self.model_np
        x = self.X_split[0]  # [train_trials, test_trials] for train samples
        y = self.Y_split[0]

        initial_loss, initial_params, train_loss, optimized_params = objective(
            model,
            self.param_estimator,
            x=x,
            y=y,
            loss_fn=self.loss_fn,
            fit_params=fit_params,
            max_iter=max_iter,
            learning_rate=lr,
        )

        self.initial_params = initial_params
        self.optimized_params = optimized_params
        self.train_loss = float(train_loss)
        return self

    def evaluate_test(self) -> "Rewind":
        """Evaluate the model on the test-sample data.

        Calls ``objective`` on ``X_split[1]`` / ``Y_split[1]`` (test samples),
        fitting fresh parameters.  Stores ``test_loss``.
        """
        self._require_model()
        self._require_data()

        model = self.model_jax or self.model_np
        x = self.X_split[1]
        y = self.Y_split[1]

        _, _, test_loss, _ = objective(
            model,
            self.param_estimator,
            x=x,
            y=y,
            loss_fn=self.loss_fn,
            fit_params=True,
        )

        self.test_loss = float(test_loss)
        return self

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_fits(
        self,
        split: str = "test",
        show: bool = True,
        save_path: str | None = None,
    ) -> "Rewind":
        """Plot model fits using spec.plot_model_fits.

        Args:
            split: ``"train"`` or ``"test"`` — selects which samples to plot.
            show: Whether to display the plot.
            save: Whether to save the plot.
            save_path: Where to save the figure.  Defaults to
                ``{run_dir}/rewind_plots/{label}_{split}.png``.
        """
        self._require_model()
        self._require_data()

        if self.plot_fn is None:
            print("[Rewind] No plot_model_fits found in spec.  Cannot plot.")
            return self

        params = self.optimized_params if self.optimized_params is not None else self.initial_params
        if params is None:
            raise RuntimeError("No params available.  Call estimate_params() or optimize_params() first.")

        if save_path is None:
            plots_dir = os.path.join(self._run_dir, "rewind_plots")
            os.makedirs(plots_dir, exist_ok=True)
            label_str = self._current_label or "unknown"
            save_path = os.path.join(plots_dir, f"{label_str}_{split}.png")

        if split == "train":
            X_plot = np.asarray(self.X_split[0][0].to_tensor())
            Y_plot = np.asarray(self.Y_split[0][0].to_tensor())
        else:
            X_plot = np.asarray(self.X_split[1][1].to_tensor())
            Y_plot = np.asarray(self.Y_split[1][1].to_tensor())

        n_samples = X_plot.shape[0]
        # Build X_eval aligned to this sample count
        if self.X_eval is not None:
            from .utils import build_evaluation_points
            from .data_structures import Inputs
            x_eval = build_evaluation_points(
                inputs=Inputs.from_array(X_plot),
                n_bins=self.X_eval.shape[2],
            )
        else:
            x_eval = X_plot

        model = self.model_jax or self.model_np
        loss_col = self.train_loss if split == "train" else self.test_loss
        losses = np.full(n_samples, loss_col) if loss_col is not None else None

        programs_list = [{"model": model, "params": np.asarray(params[:n_samples]), "losses": losses}]

        try:
            self.plot_fn(
                X=X_plot,
                Y=Y_plot,
                programs_list=programs_list,
                X_eval=x_eval,
                save_path=save_path,
                labels=(self._current_label or "model",),
            )
            print(f"[Rewind] Saved plot to {save_path}")
        except Exception as e:
            print(f"[Rewind] plot_fits failed: {e}")

        # If show=True, load the png and display it inline (for notebook users)
        if show:
            try:
                from PIL import Image
                import matplotlib.pyplot as plt

                img = Image.open(save_path)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis("off")
                plt.show()
            except Exception as e:
                print(f"[Rewind] Could not display plot: {e}")

        return self

    # ------------------------------------------------------------------
    # Re-translation
    # ------------------------------------------------------------------

    def retranslate(self) -> "Rewind":
        """Re-translate model_code_numpy to JAX via translate_to_jax.

        Uses ``asyncio.run()`` with nest_asyncio for notebook compatibility.
        Updates ``self.model_jax`` and ``self.model_code_jax``.
        """
        if not self.model_code_np:
            raise RuntimeError("No numpy model code stored.  Call go_to() first.")

        from dotenv import load_dotenv
        from google import genai
        from .hypothesis_engine import translate_to_jax
        from .prompt_manager import PromptManager

        load_dotenv()
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        pm = PromptManager(config=self.config or {})

        async def _run():
            return await translate_to_jax(
                self.model_code_np, client, pm,
                llm_name=self.config.get("experiment_params", {}).get(
                    "tiny_lm_name", "gemini-2.0-flash-lite"
                ),
            )

        jax_code, jax_func = asyncio.run(_run())
        self.model_code_jax = jax_code
        self.model_jax = jax_func
        print("[Rewind] JAX translation complete.")
        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_state(self) -> "Rewind":
        """Pretty-print the current model label, losses, and parameter shape."""
        label = self._current_label or "(none)"
        print(f"Rewind — label: {label}")
        rec = self._current_record or {}
        print(f"  iteration: {rec.get('iteration_number')}  "
              f"island: {rec.get('birth_island')}  "
              f"batch: {rec.get('batch_index')}")
        print(f"  stored train_loss : {rec.get('train_loss')}")
        print(f"  stored test_loss  : {rec.get('test_loss')}")
        print(f"  rewind train_loss : {self.train_loss}")
        print(f"  rewind test_loss  : {self.test_loss}")
        if self.optimized_params is not None:
            print(f"  optimized_params  : shape {np.asarray(self.optimized_params).shape}")
        elif self.initial_params is not None:
            print(f"  initial_params    : shape {np.asarray(self.initial_params).shape}")
        else:
            print("  params            : (none)")
        return self

    def show_code(self, which: str = "jax") -> "Rewind":
        """Print model or param-estimator source code.

        Args:
            which: ``"jax"``, ``"numpy"``, or ``"param_est"``.
        """
        mapping = {
            "jax": self.model_code_jax,
            "numpy": self.model_code_np,
            "param_est": self.param_est_code,
        }
        code = mapping.get(which)
        if code is None:
            print(f"[Rewind] No code stored for which='{which}'.  "
                  "Call go_to() first, or choose 'jax'/'numpy'/'param_est'.")
        else:
            print(code)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_model(self):
        if self.model_jax is None and self.model_np is None:
            raise RuntimeError("No model loaded.  Call go_to('<label>') first.")

    def _require_data(self):
        if self.X_split is None:
            raise RuntimeError(
                "Data not loaded.  Ensure run_config.json exists and "
                "load_previous_run() succeeded in data mode."
            )
        if self.loss_fn is None:
            raise RuntimeError("loss_fn not loaded from spec.")

    # ------------------------------------------------------------------
    # Saved image display
    # ------------------------------------------------------------------

    def show_fit_image(self, label: str | None = None, split: str = "train") -> "Rewind":
        """Display the pre-rendered fit PNG stored in the family tree for *label*.

        These are the same images shown in the family_tree.html sidebar.
        If *label* is None, uses the currently selected model (``go_to`` label).

        Args:
            label: Family-tree label (e.g. ``"D18"``).  Defaults to the current label.
            split: ``"train"`` or ``"test"``.
        """
        import matplotlib.pyplot as plt

        if self._label_map is None:
            raise RuntimeError("No run loaded.  Call load_previous_run() first.")

        target_label = label or self._current_label
        if target_label is None:
            raise RuntimeError("No label given and no current model selected.  Call go_to() first.")

        rec = self._label_map.get(target_label)
        if rec is None:
            raise KeyError(f"Label '{target_label}' not found.")

        key = "train_fit_image_path" if split == "train" else "test_fit_image_path"
        path_str = rec.get(key)

        if not path_str:
            print(f"[Rewind] No {split} fit image path stored for '{target_label}'.")
            return self

        import os
        if not os.path.isfile(path_str):
            print(f"[Rewind] Image file not found: {path_str}")
            return self

        img = plt.imread(path_str)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img)
        ax.axis("off")
        loss = rec.get("train_loss")
        loss_str = f"{loss:.4f}" if loss is not None else "n/a"
        ax.set_title(
            f"{split.capitalize()} fit — {target_label}  "
            f"iter={rec.get('iteration_number')}  "
            f"island={rec.get('birth_island')}  "
            f"train_loss={loss_str}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.show()
        return self
