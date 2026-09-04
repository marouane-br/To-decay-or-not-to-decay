"""Run a fixed-configuration DAE experiment grid for IDS datasets.

The runner supports FF_DAE, DVAE, RES_DAE, and SPARSE_DAE variants. The main
inferential focus is decaying DAE schedules versus DAE CONST baselines.

Workflow: CLI -> dataset loading -> DAE model grid -> validation threshold
selection -> test evaluation -> artifact export.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import math
import random
import hashlib
import argparse
import json
import gc
import copy
import pickle
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, BatchNormalization, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)


# ============================================================
# 0) Reproducibility
# ============================================================
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_deterministic_init_seed(dataset_name, seed, ae_variant, base_seed=0):
    token = (
        f"Dataset={dataset_name}|"
        f"Seed={int(seed)}|"
        f"AEVariant={normalize_ae_variant(ae_variant)}|"
        f"BaseSeed={int(base_seed)}"
    )
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def set_config_random_seeds(config_seed):
    config_seed = int(config_seed)
    random.seed(config_seed)
    np.random.seed(config_seed)
    tf.random.set_seed(config_seed)
    os.environ["PYTHONHASHSEED"] = str(config_seed)


def _default_dataset_dir():
    return os.environ.get("DATASET_DIR", os.path.join(os.getcwd(), "dataset"))


def _resolve_training_phase(training):
    """Return a boolean/symbolic training flag for custom layers.

    Functional models call custom layers once during graph construction with
    ``training=None``. Returning the inference branch in that case bakes a
    no-op into the model graph, so scheduled corruption and DVAE sampling would
    never be active during ``fit``. When available, defer to Keras' learning
    phase so the graph can switch between training and inference at runtime.
    """
    if training is not None:
        return training

    learning_phase = getattr(tf.keras.backend, "learning_phase", None)
    if learning_phase is not None:
        return learning_phase()

    # Keras versions without a public learning-phase symbol should prefer the
    # safe inference path when a caller does not explicitly provide training.
    return False


def _resolve_dataset_path(configured_path, *fallbacks):
    if configured_path is None or str(configured_path).strip() == "":
        raise ValueError("Dataset path must be a non-empty string")

    configured_path = os.path.expanduser(str(configured_path))
    if os.path.exists(configured_path):
        return configured_path

    checked = [configured_path]
    for fb in fallbacks:
        if not fb:
            continue
        fb = os.path.expanduser(str(fb))
        checked.append(fb)
        if os.path.exists(fb):
            print(f"[DatasetPath] Using fallback path: {fb}")
            return fb

    raise FileNotFoundError(
        "Dataset file not found. Checked paths:\n- " + "\n- ".join(checked)
    )


# ============================================================
# 1) Metrics
# ============================================================
def compute_mcc(tp, tn, fp, fn):
    tp = float(int(tp))
    tn = float(int(tn))
    fp = float(int(fp))
    fn = float(int(fn))

    a = tp + fp
    b = tp + fn
    c = tn + fp
    d = tn + fn
    denom = a * b * c * d
    if not math.isfinite(denom) or denom <= 0.0:
        return 0.0
    num = (tp * tn) - (fp * fn)
    return num / math.sqrt(denom)


def _pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d


def _print_dist(name, y_series, normal_token="normal"):
    total = len(y_series)
    ys = pd.Series(y_series)
    n_norm = int(np.sum(ys.astype(str).str.lower().eq(str(normal_token).lower()).values))
    n_anom = total - n_norm
    print(
        f"[{name}] total={total:,} | normal={n_norm:,} ({_pct(n_norm,total):5.2f}%) "
        f"| anomaly={n_anom:,} ({_pct(n_anom,total):5.2f}%)"
    )


def _print_split_overview(dataset_name, n_total, n_train_window, n_ae_train, n_val, n_test):
    print(
        f"[{dataset_name}] split overview | "
        f"dataset_total={n_total:,} (100.00%) | "
        f"initial_train_window={n_train_window:,} ({_pct(n_train_window, n_total):5.2f}%) | "
        f"ae_training_benign_subset={n_ae_train:,} ({_pct(n_ae_train, n_total):5.2f}%) | "
        f"val_mixed={n_val:,} ({_pct(n_val, n_total):5.2f}%) | "
        f"test_mixed={n_test:,} ({_pct(n_test, n_total):5.2f}%)"
    )


def _print_train_window_stats(dataset_name, train_window_total, train_window_normals):
    print(
        f"[{dataset_name}] initial train window | "
        f"total={train_window_total:,} | "
        f"benign_retained_for_ae={train_window_normals:,} ({_pct(train_window_normals, train_window_total):5.2f}%)"
    )


def _print_split_plan(dataset_name, train_frac, val_frac, test_frac):
    print(
        f"[{dataset_name}] split plan | "
        f"initial_train_window={100.0 * float(train_frac):5.2f}% | "
        f"val_mixed={100.0 * float(val_frac):5.2f}% | "
        f"test_mixed={100.0 * float(test_frac):5.2f}%"
    )


def labels_to_binary_int(y_str):
    return np.array([1 if str(v).lower() == "anomaly" else 0 for v in y_str], dtype=int)


def compute_binary_metrics_from_preds(y_true_str, y_pred_str, scores=None):
    """Compute confusion-derived binary metrics and optional ROC/PR AUC values."""
    cm = confusion_matrix(y_true_str, y_pred_str, labels=["normal", "anomaly"])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
    mcc = compute_mcc(tp, tn, fp, fn)

    roc_auc = np.nan
    pr_auc = np.nan
    if scores is not None:
        y_true_bin = labels_to_binary_int(y_true_str)
        if len(np.unique(y_true_bin)) > 1:
            roc_auc = roc_auc_score(y_true_bin, scores)
            pr_auc = average_precision_score(y_true_bin, scores)

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "MCC": mcc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


# ============================================================
# 2) Noise layer
# ============================================================
class ScheduledNoise(Layer):
    ALLOWED_NOISE_TYPES = {"gaussian", "uniform", "masking"}

    def __init__(self, init_sigma=0.10, noise_type="gaussian", **kwargs):
        super().__init__(**kwargs)
        self.sigma_var = tf.Variable(init_sigma, dtype=tf.float32, trainable=False, name="sigma")
        self.noise_type = tf.Variable("gaussian", dtype=tf.string, trainable=False, name="noise_type")
        self.set_noise_type(noise_type)

    def call(self, inputs, training=None):
        sigma = tf.maximum(self.sigma_var, tf.constant(0.0, dtype=tf.float32))

        def _apply_noise():
            noise_type = self.noise_type

            def _gaussian():
                return inputs + tf.random.normal(tf.shape(inputs), 0.0, sigma)

            def _uniform():
                return inputs + tf.random.uniform(tf.shape(inputs), minval=-sigma, maxval=sigma)

            def _masking():
                mask_prob = tf.clip_by_value(sigma, 0.0, 1.0)
                keep_mask = tf.cast(
                    tf.random.uniform(tf.shape(inputs), minval=0.0, maxval=1.0) >= mask_prob,
                    inputs.dtype,
                )
                return inputs * keep_mask

            return tf.case(
                [
                    (tf.equal(noise_type, tf.constant("gaussian")), _gaussian),
                    (tf.equal(noise_type, tf.constant("uniform")), _uniform),
                ],
                default=_masking,
                exclusive=True,
            )

        training = _resolve_training_phase(training)
        if training is True:
            return _apply_noise()
        if training is False:
            return inputs
        return tf.cond(
            tf.cast(training, tf.bool),
            _apply_noise,
            lambda: inputs,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def set_sigma(self, new_sigma):
        self.sigma_var.assign(tf.cast(new_sigma, tf.float32))

    def set_noise_type(self, noise_type):
        noise_type = str(noise_type).strip().lower()
        if noise_type not in self.ALLOWED_NOISE_TYPES:
            raise ValueError(
                f"Unsupported noise type '{noise_type}'. "
                f"Allowed values: {sorted(self.ALLOWED_NOISE_TYPES)}"
            )
        self.noise_type.assign(noise_type)


# ============================================================
# 3) Optimizer
# ============================================================
def _make_optimizer(name: str, lr: float):
    name = (name or "adam").lower()
    if name == "adam":
        return Adam(learning_rate=lr)
    if name == "sgd":
        return SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return RMSprop(learning_rate=lr)
    raise ValueError(f"Unknown optimizer '{name}'")


# ============================================================
# 4) AE builders
# ============================================================
def build_denoising_ae(
    input_dim: int,
    latent_dim: int,
    enc_units,
    dec_units,
    activation="relu",
    output_activation="sigmoid",
    l2_reg: float = 0.0,
    dropout: float = 0.0,
    batch_norm: bool = False,
    noise_std: float = 0.1,
    noise_type: str = "gaussian",
    optimizer: str = "adam",
    lr: float = 1e-3,
    loss: str = "mse",
    compile_model: bool = True,
):
    reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    inp = Input(shape=(input_dim,), name="inp")

    noise_layer = ScheduledNoise(init_sigma=noise_std, noise_type=noise_type, name="sched_noise")
    # Let Keras provide the runtime training flag so noise is applied only during training passes.
    x = noise_layer(inp)

    for i, u in enumerate(enc_units):
        x = Dense(u, activation=activation, kernel_regularizer=reg, name=f"dae_enc_{u}_{i}")(x)
        if batch_norm:
            x = BatchNormalization(name=f"dae_enc_bn_{i}")(x)
        if dropout and dropout > 0:
            x = Dropout(dropout, name=f"dae_enc_do_{i}")(x)

    z = Dense(latent_dim, activation=activation, kernel_regularizer=reg, name="dae_latent")(x)

    x = z
    for i, u in enumerate(dec_units):
        x = Dense(u, activation=activation, kernel_regularizer=reg, name=f"dae_dec_{u}_{i}")(x)
        if batch_norm:
            x = BatchNormalization(name=f"dae_dec_bn_{i}")(x)
        if dropout and dropout > 0:
            x = Dropout(dropout, name=f"dae_dec_do_{i}")(x)

    outp = Dense(input_dim, activation=output_activation, name="dae_recon")(x)
    model = Model(inp, outp, name="DAE")
    model.noise_layer = noise_layer

    if compile_model:
        opt = _make_optimizer(optimizer, lr)
        model.compile(optimizer=opt, loss=loss)

    return model


def _residual_dense_block(x, units, activation, reg, batch_norm=False, dropout=0.0, block_name="res_block"):
    shortcut = x
    x = Dense(units, activation=activation, kernel_regularizer=reg, name=f"{block_name}_dense")(x)
    if batch_norm:
        x = BatchNormalization(name=f"{block_name}_bn")(x)
    if dropout and dropout > 0:
        x = Dropout(dropout, name=f"{block_name}_do")(x)
    if shortcut.shape[-1] != units:
        shortcut = Dense(units, activation=None, kernel_regularizer=reg, name=f"{block_name}_proj")(shortcut)
    x = Add(name=f"{block_name}_add")([x, shortcut])
    x = Activation(activation, name=f"{block_name}_act")(x)
    return x


def build_residual_denoising_ae(
    input_dim: int,
    latent_dim: int,
    enc_units,
    dec_units,
    activation="relu",
    output_activation="sigmoid",
    l2_reg: float = 0.0,
    dropout: float = 0.0,
    batch_norm: bool = False,
    noise_std: float = 0.1,
    noise_type: str = "gaussian",
    optimizer: str = "adam",
    lr: float = 1e-3,
    loss: str = "mse",
    compile_model: bool = True,
):
    reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    inp = Input(shape=(input_dim,), name="res_dae_inp")
    noise_layer = ScheduledNoise(init_sigma=noise_std, noise_type=noise_type, name="sched_noise")
    x = noise_layer(inp)

    for i, u in enumerate(enc_units):
        x = _residual_dense_block(x, int(u), activation, reg, batch_norm=batch_norm, dropout=dropout, block_name=f"res_dae_enc_{i}")

    z = Dense(latent_dim, activation=activation, kernel_regularizer=reg, name="res_dae_latent")(x)
    x = z
    for i, u in enumerate(dec_units):
        x = _residual_dense_block(x, int(u), activation, reg, batch_norm=batch_norm, dropout=dropout, block_name=f"res_dae_dec_{i}")

    outp = Dense(input_dim, activation=output_activation, name="res_dae_recon")(x)
    model = Model(inp, outp, name="RES_DAE")
    model.noise_layer = noise_layer
    if compile_model:
        model.compile(optimizer=_make_optimizer(optimizer, lr), loss=loss)
    return model


def build_sparse_denoising_ae(
    input_dim: int,
    latent_dim: int,
    enc_units,
    dec_units,
    activation="relu",
    output_activation="sigmoid",
    l2_reg: float = 0.0,
    dropout: float = 0.0,
    batch_norm: bool = False,
    noise_std: float = 0.1,
    noise_type: str = "gaussian",
    optimizer: str = "adam",
    lr: float = 1e-3,
    loss: str = "mse",
    sparsity_l1: float = 1e-5,
    compile_model: bool = True,
):
    reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    inp = Input(shape=(input_dim,), name="sparse_dae_inp")
    noise_layer = ScheduledNoise(init_sigma=noise_std, noise_type=noise_type, name="sched_noise")
    x = noise_layer(inp)

    for i, u in enumerate(enc_units):
        x = Dense(u, activation=activation, kernel_regularizer=reg, name=f"sparse_dae_enc_{u}_{i}")(x)
        if batch_norm:
            x = BatchNormalization(name=f"sparse_dae_enc_bn_{i}")(x)
        if dropout and dropout > 0:
            x = Dropout(dropout, name=f"sparse_dae_enc_do_{i}")(x)

    z = Dense(
        latent_dim,
        activation=activation,
        kernel_regularizer=reg,
        activity_regularizer=regularizers.l1(sparsity_l1),
        name="sparse_dae_latent",
    )(x)

    x = z
    for i, u in enumerate(dec_units):
        x = Dense(u, activation=activation, kernel_regularizer=reg, name=f"sparse_dae_dec_{u}_{i}")(x)
        if batch_norm:
            x = BatchNormalization(name=f"sparse_dae_dec_bn_{i}")(x)
        if dropout and dropout > 0:
            x = Dropout(dropout, name=f"sparse_dae_dec_do_{i}")(x)

    outp = Dense(input_dim, activation=output_activation, name="sparse_dae_recon")(x)
    model = Model(inp, outp, name="SPARSE_DAE")
    model.noise_layer = noise_layer
    if compile_model:
        model.compile(optimizer=_make_optimizer(optimizer, lr), loss=loss)
    return model


class Sampling(Layer):
    """Reparameterization layer that also contributes the beta-weighted KL loss."""

    def __init__(self, beta=0.001, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_sum(
            1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1,
        )
        self.add_loss(self.beta * tf.reduce_mean(kl_loss))

        def _sample():
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        training = _resolve_training_phase(training)
        if training is True:
            return _sample()
        if training is False:
            return z_mean
        return tf.cond(tf.cast(training, tf.bool), _sample, lambda: z_mean)



def build_denoising_vae(
    input_dim,
    latent_dim,
    enc_units,
    dec_units,
    activation,
    output_activation,
    noise_std,
    noise_type,
    beta,
    optimizer,
    lr,
):
    inp = Input(shape=(input_dim,), name="dvae_inp")
    noise_layer = ScheduledNoise(init_sigma=noise_std, noise_type=noise_type, name="sched_noise")
    x = noise_layer(inp)
    for i, u in enumerate(enc_units):
        x = Dense(u, activation=activation, name=f"dvae_enc_{u}_{i}")(x)
    z_mean = Dense(latent_dim, name="dvae_z_mean")(x)
    z_log_var = Dense(latent_dim, name="dvae_z_log_var")(x)
    encoder = Model(inp, [z_mean, z_log_var], name="DVAE_encoder")
    z = Sampling(beta=beta, name="dvae_sampling")([z_mean, z_log_var])
    x = z
    for i, u in enumerate(dec_units):
        x = Dense(u, activation=activation, name=f"dvae_dec_{u}_{i}")(x)
    outp = Dense(input_dim, activation=output_activation, name="dvae_recon")(x)
    model = Model(inp, outp, name="DVAE")
    model.encoder_model = encoder
    model.noise_layer = noise_layer
    model.compile(optimizer=_make_optimizer(optimizer, lr), loss="mse")
    return model


def assert_noise_layer_present(model):
    if not hasattr(model, "noise_layer"):
        raise ValueError("DAE model is missing model.noise_layer")
    if not any(layer is model.noise_layer for layer in model.layers):
        raise ValueError("ScheduledNoise layer is not connected to the model graph")


def verify_model_training_behavior(model, X_sample, model_label="", atol=1e-7):
    X_sample = np.asarray(X_sample[: min(len(X_sample), 16)]).astype(np.float32)

    if len(X_sample) == 0:
        print(f"[WARN] Cannot verify training behavior for {model_label}: empty sample.")
        return

    y_train_1 = model(X_sample, training=True).numpy()
    y_train_2 = model(X_sample, training=True).numpy()
    y_eval = model(X_sample, training=False).numpy()

    train_vs_eval_diff = float(np.mean(np.abs(y_train_1 - y_eval)))
    train_repeat_diff = float(np.mean(np.abs(y_train_1 - y_train_2)))

    has_noise = hasattr(model, "noise_layer")
    has_dvae_encoder = hasattr(model, "encoder_model")
    sigma = np.nan

    if has_noise:
        try:
            sigma = float(model.noise_layer.sigma_var.numpy())
        except Exception:
            sigma = np.nan

    print(
        f"[TrainingCheck] {model_label} | "
        f"has_noise={has_noise} | has_dvae_encoder={has_dvae_encoder} | sigma={sigma} | "
        f"mean_abs(train_eval_diff)={train_vs_eval_diff:.8e} | "
        f"mean_abs(train_repeat_diff)={train_repeat_diff:.8e}"
    )

    if has_noise and np.isfinite(sigma) and sigma > 0:
        if train_vs_eval_diff <= atol and train_repeat_diff <= atol:
            raise RuntimeError(
                f"[TrainingCheck] ScheduledNoise may be inactive for {model_label}. "
                f"training=True and training=False outputs are nearly identical while sigma={sigma}."
            )

    if has_dvae_encoder and has_noise:
        old_sigma = None
        try:
            old_sigma = float(model.noise_layer.sigma_var.numpy())
            model.noise_layer.set_sigma(0.0)

            y_sample_1 = model(X_sample, training=True).numpy()
            y_sample_2 = model(X_sample, training=True).numpy()
            sampling_repeat_diff = float(np.mean(np.abs(y_sample_1 - y_sample_2)))

            print(
                f"[TrainingCheck] {model_label} | "
                f"dvae_sampling_repeat_diff_no_noise={sampling_repeat_diff:.8e}"
            )

            if sampling_repeat_diff <= atol:
                raise RuntimeError(
                    f"[TrainingCheck] DVAE sampling may be inactive for {model_label}. "
                    "Two training=True forward passes are nearly identical with noise sigma forced to 0."
                )
        finally:
            if old_sigma is not None:
                model.noise_layer.set_sigma(old_sigma)

    if has_dvae_encoder and not hasattr(model, "encoder_model"):
        raise RuntimeError(
            f"[TrainingCheck] DVAE model {model_label} is missing encoder_model."
        )


# ============================================================
# 5) Noise schedules
# ============================================================
def get_noise_schedule(schedule_name, epochs, params=None):
    """Return an epoch-indexed sigma schedule callable for DAE noise decay."""
    if params is None:
        params = {}

    name = (schedule_name or "constant").lower()

    if name == "constant":
        sigma = float(params.get("sigma", 0.1))
        def schedule(epoch):
            return sigma
        return schedule

    if name == "linear":
        sigma_start = float(params.get("sigma_start", 0.2))
        sigma_end = float(params.get("sigma_end", 0.01))
        if epochs <= 1:
            def schedule(epoch):
                return sigma_end
            return schedule

        def schedule(epoch):
            t = min(max(epoch, 0), epochs - 1)
            alpha = t / float(epochs - 1)
            return sigma_start + (sigma_end - sigma_start) * alpha
        return schedule

    if name == "exponential":
        sigma_start = float(params.get("sigma_start", 0.2))
        sigma_min = float(params.get("sigma_min", 0.01))
        decay_rate = float(params.get("decay_rate", 3.0))

        def schedule(epoch):
            if epochs <= 1:
                return max(sigma_min, sigma_start)
            t = epoch / float(epochs - 1)
            sigma = sigma_start * math.exp(-decay_rate * t)
            return max(sigma_min, sigma)
        return schedule

    if name == "cosine":
        sigma_start = float(params.get("sigma_start", 0.2))
        sigma_end = float(params.get("sigma_end", 0.0))
        if epochs <= 1:
            def schedule(epoch):
                return sigma_end
            return schedule

        def schedule(epoch):
            t = min(max(epoch, 0), epochs - 1) / float(epochs - 1)
            return sigma_end + 0.5 * (sigma_start - sigma_end) * (1.0 + math.cos(math.pi * t))
        return schedule

    def make_scaled_schedule(sigma_start, sigma_end, g_func):
        g0 = g_func(0.0)
        g1 = g_func(1.0)

        if g0 == g1:
            def schedule(epoch):
                if epochs <= 1:
                    return float(sigma_end)
                t = min(max(epoch, 0), epochs - 1) / float(epochs - 1)
                return sigma_start + (sigma_end - sigma_start) * t
            return schedule

        def schedule_t(t):
            return sigma_end + (sigma_start - sigma_end) * (g_func(t) - g1) / (g0 - g1)

        def schedule(epoch):
            if epochs <= 1:
                return float(sigma_end)
            t = min(max(epoch, 0), epochs - 1) / float(epochs - 1)
            return float(schedule_t(t))

        return schedule

    if name == "fibonacci":
        sigma_start = float(params.get("sigma_start", 0.2))
        sigma_end = float(params.get("sigma_end", 0.01))

        fib = [1.0, 1.0]
        while len(fib) < max(epochs, 2):
            fib.append(fib[-1] + fib[-2])
        fib = fib[:max(epochs, 2)]
        max_fib = fib[-1]

        def schedule(epoch):
            if epochs <= 1:
                return sigma_end
            e = min(max(epoch, 0), epochs - 1)
            w = fib[-1 - e] / max_fib
            w_min = fib[0] / max_fib
            w_max = fib[-1] / max_fib
            if w_max == w_min:
                alpha = e / float(epochs - 1)
            else:
                alpha = (w - w_min) / (w_max - w_min)
            alpha = max(0.0, min(1.0, alpha))
            return sigma_end + (sigma_start - sigma_end) * alpha

        return schedule

    if name == "sigmoid":
        sigma_start = float(params.get("sigma_start", 0.2))
        sigma_end = float(params.get("sigma_end", 0.01))
        k = float(params.get("k", 10.0))
        t0 = float(params.get("t0", 0.5))

        def g(t):
            return 1.0 / (1.0 + math.exp(k * (t - t0)))

        return make_scaled_schedule(sigma_start, sigma_end, g)

    if name == "cauchy":
        sigma_start = float(params.get("sigma_start", 0.2))
        sigma_end = float(params.get("sigma_end", 0.01))
        gamma = float(params.get("gamma", 0.3))

        def g(t):
            return 1.0 / (1.0 + (t / gamma) ** 2)

        return make_scaled_schedule(sigma_start, sigma_end, g)

    if name == "laplace":
        sigma_start = float(params.get("sigma_start", 0.2))
        sigma_end = float(params.get("sigma_end", 0.01))
        b = float(params.get("b", 0.3))

        def g(t):
            return math.exp(-t / b)

        return make_scaled_schedule(sigma_start, sigma_end, g)

    if name == "logistic":
        sigma_start = float(params.get("sigma_start", 0.2))
        sigma_end = float(params.get("sigma_end", 0.01))
        k = float(params.get("k", 6.0))
        t0 = float(params.get("t0", 0.5))

        def g(t):
            return 1.0 / (1.0 + math.exp(k * (t - t0)))

        return make_scaled_schedule(sigma_start, sigma_end, g)

    raise ValueError(f"Unknown noise schedule '{schedule_name}'")


class NoiseDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, noise_layer, schedule_fn, verbose=True):
        super().__init__()
        self.noise_layer = noise_layer
        self.schedule_fn = schedule_fn
        self.verbose = verbose
        self.history = []

    def on_epoch_begin(self, epoch, logs=None):
        sigma = float(self.schedule_fn(epoch))
        self.noise_layer.set_sigma(sigma)
        self.history.append(sigma)
        if self.verbose:
            print(f"[NoiseSchedule] Epoch {epoch + 1} -> sigma={sigma:.6f}")


# ============================================================
# 6) Training
# ============================================================
def train_denoising_ae(
    model,
    X_train_np,
    epochs=30,
    batch_size=512,
    noise_schedule_name="constant",
    noise_schedule_params=None,
    save_epoch_recon=False,
):
    """Train a denoising autoencoder (DAE) with a fixed named sigma schedule."""
    schedule_fn = get_noise_schedule(noise_schedule_name, epochs, noise_schedule_params)
    callbacks = [NoiseDecayCallback(model.noise_layer, schedule_fn, verbose=False)]

    # The model is trained for a fixed number of epochs on benign training data only.
    # The mixed validation set is reserved exclusively for post-training threshold selection.
    hist = model.fit(
        x=X_train_np,
        y=X_train_np,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        callbacks=callbacks,
    )

    if save_epoch_recon:
        n_epochs = len(hist.history["loss"])
        sigma_hist = callbacks[0].history[:n_epochs]
        if len(sigma_hist) < n_epochs:
            sigma_hist = sigma_hist + [np.nan] * (n_epochs - len(sigma_hist))
        return pd.DataFrame({
            "Epoch": np.arange(1, n_epochs + 1),
            "Train_Recon": hist.history["loss"],
            "Val_Recon": [np.nan] * n_epochs,
            "Sigma": sigma_hist,
        })
    return None



# ============================================================
# 7) Evaluation and plots
# ============================================================
def reconstruct_errors(model, X_np, batch=1024):
    """Compute one reconstruction-error anomaly score per input sample."""
    batch = int(batch)
    if batch <= 0:
        raise ValueError(f"batch must be a positive integer, got {batch}")
    if len(X_np) == 0:
        raise ValueError("Cannot compute reconstruction errors for an empty input array")

    errs = []
    for i in range(0, len(X_np), batch):
        xb = X_np[i:i + batch].astype(np.float32)
        xr = model(xb, training=False)
        diff = xr.numpy() - xb

        if diff.ndim != 2:
            raise ValueError(
                f"Unsupported reconstruction output rank {diff.ndim}; "
                "all active AE variants should produce 2D sample-based outputs."
            )

        e = np.mean(diff ** 2, axis=1)
        errs.append(e)

    return np.concatenate(errs, axis=0)



def compute_dvae_score_components(model, X_np, batch=1024, beta=0.001):
    recon_errors = reconstruct_errors(model, X_np, batch=batch)

    if not hasattr(model, "encoder_model"):
        raise ValueError("DVAE model is missing encoder_model needed for ELBO scoring.")

    kl_values = []
    for i in range(0, len(X_np), batch):
        xb = X_np[i:i + batch].astype(np.float32)
        z_mean, z_log_var = model.encoder_model(xb, training=False)
        z_mean = z_mean.numpy()
        z_log_var = z_log_var.numpy()

        kl = -0.5 * np.sum(
            1.0 + z_log_var - np.square(z_mean) - np.exp(z_log_var),
            axis=1,
        )
        kl_values.append(kl)

    kl_values = np.concatenate(kl_values, axis=0)
    elbo_scores = recon_errors + float(beta) * kl_values

    return {
        "recon": recon_errors,
        "kl": kl_values,
        "elbo": elbo_scores,
    }


def select_threshold_from_scores(
    train_scores,
    val_scores,
    y_val_str,
    select_metric="MCC",
    percentile_grid=range(1, 101),
):
    train_scores = np.asarray(train_scores).reshape(-1)
    val_scores = np.asarray(val_scores).reshape(-1)

    if len(train_scores) == 0 or len(val_scores) == 0:
        raise ValueError("Cannot select threshold from empty score arrays.")
    if not np.all(np.isfinite(train_scores)):
        raise ValueError("Training anomaly scores contain NaN or infinite values.")
    if not np.all(np.isfinite(val_scores)):
        raise ValueError("Validation anomaly scores contain NaN or infinite values.")

    percentile_values = list(percentile_grid)
    if not percentile_values:
        raise ValueError("percentile_grid must contain at least one percentile.")

    rows = []
    for p in percentile_values:
        p = float(p)
        if p < 0.0 or p > 100.0:
            raise ValueError(f"Percentile values must be in [0, 100], got {p}")

        tau = float(np.percentile(train_scores, p))
        val_pred = np.where(val_scores > tau, "anomaly", "normal")

        metric_row = compute_binary_metrics_from_preds(
            y_true_str=y_val_str,
            y_pred_str=val_pred,
            scores=val_scores,
        )

        rows.append({
            "Percentile": p,
            "Threshold": tau,
            **metric_row,
        })

    df = pd.DataFrame(rows)
    if select_metric not in df.columns:
        raise ValueError(f"Validation metric column '{select_metric}' is not present in threshold sweep results.")

    valid_metric = df[select_metric].dropna()
    if valid_metric.empty:
        raise ValueError(
            f"Validation metric '{select_metric}' has no valid values. Cannot select a threshold."
        )

    best_idx = valid_metric.idxmax()
    best_row = df.loc[best_idx].copy()
    return df, best_row


def select_threshold_on_validation(
    model,
    X_train_norm,
    X_val,
    y_val_str,
    select_metric="MCC",
    percentile_grid=range(1, 101),
):
    """Sweep reconstruction-error thresholds on validation and return full sweep plus best row."""
    train_err = reconstruct_errors(model, X_train_norm)
    val_err = reconstruct_errors(model, X_val)
    return select_threshold_from_scores(
        train_scores=train_err,
        val_scores=val_err,
        y_val_str=y_val_str,
        select_metric=select_metric,
        percentile_grid=percentile_grid,
    )


def evaluate_fixed_threshold_from_scores(test_scores, y_test_str, threshold):
    test_scores = np.asarray(test_scores).reshape(-1)

    if len(test_scores) == 0:
        raise ValueError("Cannot evaluate empty test score array.")
    if not np.all(np.isfinite(test_scores)):
        raise ValueError("Test anomaly scores contain NaN or infinite values.")

    test_pred = np.where(test_scores > float(threshold), "anomaly", "normal")

    metrics = compute_binary_metrics_from_preds(
        y_true_str=y_test_str,
        y_pred_str=test_pred,
        scores=test_scores,
    )

    return metrics, test_scores, test_pred


def evaluate_fixed_threshold_on_test(model, X_test, y_test_str, threshold):
    """Evaluate test predictions using a fixed reconstruction-error threshold."""
    test_err = reconstruct_errors(model, X_test)
    return evaluate_fixed_threshold_from_scores(
        test_scores=test_err,
        y_test_str=y_test_str,
        threshold=threshold,
    )

def plot_error_distribution(name, errors, labels, out_jpg_path, bins=200, clip_percentile=99.5, score_label="Reconstruction error"):
    _ensure_parent_dir(out_jpg_path)

    errors = np.asarray(errors).reshape(-1)
    labels = np.asarray(labels)

    m_norm = (labels == "normal")
    m_anom = (labels == "anomaly")

    if errors.size == 0:
        print(f"[Plot] {name}: empty errors, skipping")
        return

    xmax = float(np.percentile(errors, clip_percentile))
    xmin = float(np.percentile(errors, 0.0))
    if xmax <= xmin:
        xmax = errors.max()

    plt.figure(figsize=(8, 5))
    plt.hist(errors[m_norm], bins=bins, range=(xmin, xmax), alpha=0.6, label=f"Normal (n={m_norm.sum():,})", density=True)
    plt.hist(errors[m_anom], bins=bins, range=(xmin, xmax), alpha=0.6, label=f"Attack (n={m_anom.sum():,})", density=True)
    plt.xlabel(score_label)
    plt.ylabel("Density")
    plt.title(f"{score_label} distribution | {name}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_jpg_path, dpi=150)
    print(f"[Plot] Saved {out_jpg_path}")
    plt.close()


def _ensure_parent_dir(out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def plot_recon_vs_epochs_df(epoch_df, out_jpg_path="plots/recon_vs_epochs.jpg"):
    _ensure_parent_dir(out_jpg_path)
    if "Epoch" not in epoch_df.columns or "Train_Recon" not in epoch_df.columns:
        raise ValueError("[EpochPlot] DataFrame missing required columns: Epoch, Train_Recon")

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_df["Epoch"].values, epoch_df["Train_Recon"].values, marker="o", label="train")
    if "Val_Recon" in epoch_df.columns and epoch_df["Val_Recon"].notna().any():
        plt.plot(epoch_df["Epoch"].values, epoch_df["Val_Recon"].values, marker="s", label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Mean reconstruction error")
    plt.title("Reconstruction error over epochs")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_jpg_path, dpi=150)
    print(f"[EpochPlot] Saved {out_jpg_path}")
    plt.close()


def plot_threshold_sweep(val_df, metric_name, out_jpg_path):
    _ensure_parent_dir(out_jpg_path)
    plt.figure(figsize=(8, 5))
    plt.plot(val_df["Percentile"].values, val_df[metric_name].values, marker="o")
    plt.xlabel("Percentile")
    plt.ylabel(metric_name)
    plt.title(f"Validation threshold sweep | {metric_name}")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_jpg_path, dpi=150)
    print(f"[ThresholdPlot] Saved {out_jpg_path}")
    plt.close()


def plot_all_schedule_curves(epochs, out_jpg_path, sigma_start=0.9, sigma_end=0.1, sigma_min=0.1):
    schedules = {
        "linear": {"sigma_start": sigma_start, "sigma_end": sigma_end},
        "exponential": {"sigma_start": sigma_start, "sigma_min": sigma_min, "decay_rate": 3.0},
        "cosine": {"sigma_start": sigma_start, "sigma_end": sigma_end},
        "fibonacci": {"sigma_start": sigma_start, "sigma_end": sigma_end},
        "sigmoid": {"sigma_start": sigma_start, "sigma_end": sigma_end, "k": 10.0, "t0": 0.5},
        "cauchy": {"sigma_start": sigma_start, "sigma_end": sigma_end, "gamma": 0.3},
        "laplace": {"sigma_start": sigma_start, "sigma_end": sigma_end, "b": 0.3},
        "logistic": {"sigma_start": sigma_start, "sigma_end": sigma_end, "k": 6.0, "t0": 0.5},
    }

    _ensure_parent_dir(out_jpg_path)
    plt.figure(figsize=(10, 6))
    epoch_axis = np.arange(1, epochs + 1)

    for name, params in schedules.items():
        fn = get_noise_schedule(name, epochs, params)
        values = [fn(e) for e in range(epochs)]
        plt.plot(epoch_axis, values, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Sigma")
    plt.title("Noise schedule curves")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_jpg_path, dpi=150)
    print(f"[SchedulePlot] Saved {out_jpg_path}")
    plt.close()


# ============================================================
# 8) Dataset helpers
# ============================================================
def clean_numeric_dataframe(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        med = df[c].median()
        if np.isnan(med):
            med = 0.0
        df[c] = df[c].fillna(med)
    return df


def drop_zero_variance(df, dataset_name, out_dir="."):
    diag = pd.DataFrame({
        "min": df.min(),
        "max": df.max(),
        "std": df.std(ddof=0),
        "nunique": df.nunique(),
    })
    zv = diag[diag["std"] == 0.0].sort_index()
    print(f"[{dataset_name}] Zero variance columns ({len(zv)}): {list(zv.index)}")
    os.makedirs(out_dir, exist_ok=True)
    zv.to_csv(os.path.join(out_dir, f"{dataset_name}_dropped_zero_variance_cols.csv"), index=True)
    if len(zv) > 0:
        df = df.drop(columns=list(zv.index))
    return df


def drop_artificial_index_columns(df, dataset_name="", out_dir=None, verbose=True):
    df = df.copy()
    drop_cols = []
    for c in df.columns:
        name = str(c).strip()
        low = name.lower()
        if (
            low.startswith("unnamed:")
            or low in {"index", "level_0", "__index_level_0__"}
        ):
            drop_cols.append(c)

    if drop_cols:
        if verbose:
            print(f"[{dataset_name}] Dropping artificial CSV index columns: {drop_cols}")
        df = df.drop(columns=drop_cols, errors="ignore")
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame({"dropped_column": drop_cols}).to_csv(
                os.path.join(out_dir, f"{dataset_name}_dropped_artificial_index_cols.csv"),
                index=False,
            )
    return df


def assert_no_artificial_index_columns(columns, dataset_name=""):
    bad_cols = []
    for c in columns:
        name = str(c).strip()
        low = name.lower()
        if (
            low.startswith("unnamed:")
            or low in {"index", "level_0", "__index_level_0__"}
        ):
            bad_cols.append(c)
    if bad_cols:
        raise ValueError(
            f"[{dataset_name}] Artificial CSV index columns remain in final feature set: {bad_cols}"
        )


def assert_no_label_like_columns(columns, dataset_name=""):
    forbidden = {"label", "class", "attack", "attack_cat", "target", "category"}
    bad = [c for c in columns if str(c).strip().lower() in forbidden]
    if bad:
        raise ValueError(
            f"[{dataset_name}] Label-like columns remain in final feature set: {bad}"
        )


def fit_scale_train_and_transform(train_X, val_X, test_X):
    scaler = MinMaxScaler()
    X_train = train_X.values.astype(np.float32)
    X_val = val_X.values.astype(np.float32)
    X_test = test_X.values.astype(np.float32)

    if np.isinf(X_train).any() or np.isnan(X_train).any():
        raise ValueError("Training normals contain NaN or inf after cleanup")

    scaler.fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_val, X_test, scaler


def binary_strings_from_numeric_labels(y, normal_label):
    y = pd.Series(y)
    return y.apply(lambda x: "normal" if str(x).strip() == str(normal_label) else "anomaly")


def _is_binary_like_series(s):
    s = pd.Series(s).dropna()
    if s.empty:
        return False
    vals = set(str(v).strip().lower() for v in s.unique())
    binary_tokens = {
        "0", "1",
        "normal", "anomaly", "benign", "attack",
        "false", "true",
    }
    return vals.issubset(binary_tokens)


def _prepare_mixed_tabular_features(df, dataset_name, label_cols_to_drop, run_info_dir="."):
    X_df = df.drop(columns=list(set(label_cols_to_drop)), errors="ignore").copy()
    X_df = drop_artificial_index_columns(
        X_df,
        dataset_name=dataset_name,
        out_dir=run_info_dir,
    )

    cat_cols = X_df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    for c in cat_cols:
        X_df[c] = X_df[c].astype(str)
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)

    X_df = X_df.select_dtypes(include=[np.number, "bool"])
    if X_df.shape[1] == 0:
        raise ValueError(f"[{dataset_name}] No numeric features found after preprocessing")

    X_df = clean_numeric_dataframe(X_df)
    X_df = drop_zero_variance(X_df, dataset_name, out_dir=run_info_dir)
    return X_df


def _detect_unsw_label_column(df):
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    binary_priority = ["label", "attack", "class"]
    multiclass_priority = ["attack_cat"]

    for c in binary_priority:
        if c in cols_lower:
            col = cols_lower[c]
            if _is_binary_like_series(df[col]):
                return col, "binary"

    for c in multiclass_priority:
        if c in cols_lower:
            return cols_lower[c], "multiclass"

    candidates = [c for c in cols if ("label" in c.lower()) or ("attack" in c.lower()) or ("class" in c.lower())]
    for c in candidates:
        if _is_binary_like_series(df[c]):
            return c, "binary"
    if candidates:
        return candidates[0], "multiclass"

    raise ValueError(f"[UNSW_NB15] Could not find label column. Columns: {cols}")


def _unsw_to_binary_label_series(s):
    s = pd.Series(s)
    if np.issubdtype(s.dtype, np.number):
        return s.apply(lambda v: 0 if float(v) == 0.0 else 1).astype(int)

    def _map(v):
        token = str(v).strip().lower()
        if token in ("0", "normal", "benign", "background", "false"):
            return 0
        if token in ("1", "anomaly", "attack", "true"):
            return 1
        if "normal" in token or "benign" in token or "background" in token:
            return 0
        return 1

    return s.apply(_map).astype(int)


def ctu13_label_to_binary(s):
    s = pd.Series(s)

    def _map(v):
        token = str(v).strip().lower()

        if token in {"0", "normal", "benign", "background", "false"}:
            return 0

        if token in {"1", "anomaly", "attack", "botnet", "malicious", "true"}:
            return 1

        if "normal" in token or "benign" in token or "background" in token:
            return 0

        return 1

    return s.apply(_map).astype(int)


def split_normals_attacks_for_val_test(
    X_df,
    y_series,
    normal_label,
    train_frac=0.6,
    val_frac=0.2,
    test_frac=0.2,
    shuffle_seed=42,
    train_normals_only=True,
    existing_split_indices=None,
):
    rng = np.random.RandomState(shuffle_seed)

    if len(X_df) != len(y_series):
        raise ValueError(
            f"Feature and label lengths differ: len(X_df)={len(X_df)} vs len(y_series)={len(y_series)}"
        )
    if len(X_df) < 3:
        raise ValueError(
            f"At least 3 rows are required to create train/validation/test splits; got {len(X_df)}"
        )
    validate_split_fractions(train_frac, val_frac, test_frac)

    vals = y_series.values
    mask_norm = vals == normal_label

    n_total = len(X_df)

    if existing_split_indices is None:
        all_idx = np.arange(n_total)
        rng.shuffle(all_idx)

        n_train_total = int(round(train_frac * n_total))
        n_val_total = int(round(val_frac * n_total))
        n_test_total = int(round(test_frac * n_total))
        n_train_total = max(1, min(n_train_total, n_total - 2))
        n_val_total = max(1, min(n_val_total, n_total - n_train_total - 1))
        n_test_total = max(1, min(n_test_total, n_total - n_train_total - n_val_total))
        remainder = n_total - (n_train_total + n_val_total + n_test_total)
        n_test_total += remainder

        train_window_idx = all_idx[:n_train_total]
        train_idx = train_window_idx
        val_idx = all_idx[n_train_total:n_train_total + n_val_total]
        test_idx = all_idx[n_train_total + n_val_total:n_train_total + n_val_total + n_test_total]

        if train_normals_only:
            train_norm_mask = mask_norm[train_idx]
            train_idx = train_idx[train_norm_mask]
            if len(train_idx) == 0:
                raise ValueError(
                    "No benign samples were found in the randomized training split. "
                    "Try a different shuffle_seed."
                )

        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
    else:
        train_window_idx, train_idx, val_idx, test_idx = apply_existing_split_indices(
            existing_split_indices,
            n_total=n_total,
        )
        n_train_total = len(train_window_idx)

    X_train = X_df.iloc[train_idx].copy()
    y_train = y_series.iloc[train_idx].copy()

    X_val = X_df.iloc[val_idx].copy()
    y_val = y_series.iloc[val_idx].copy()

    X_test = X_df.iloc[test_idx].copy()
    y_test = y_series.iloc[test_idx].copy()

    y_train_arr = y_train.values
    y_val_arr = y_val.values
    y_test_arr = y_test.values
    val_normals = int(np.sum(y_val_arr == normal_label))
    test_normals = int(np.sum(y_test_arr == normal_label))

    train_window_normals = int(np.sum(mask_norm[train_window_idx]))
    split_counts = {
        "dataset_total": int(n_total),
        "train_window_total": int(n_train_total),
        "train_window_normals": train_window_normals,
        "train_window_anomalies": int(n_train_total - train_window_normals),
        "train_total": int(len(train_idx)),
        "val_total": int(len(val_idx)),
        "test_total": int(len(test_idx)),
        "train_normals": int(np.sum(y_train_arr == normal_label)),
        "train_anomalies": int(np.sum(y_train_arr != normal_label)),
        "val_normals": val_normals,
        "val_anomalies": int(len(val_idx) - val_normals),
        "test_normals": test_normals,
        "test_anomalies": int(len(test_idx) - test_normals),
    }
    split_indices = {
        "train_window": train_window_idx.copy(),
        "train": train_idx.copy(),
        "val": val_idx.copy(),
        "test": test_idx.copy(),
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, split_counts, split_indices


def save_split_csvs(
    dataset_name,
    run_info_dir,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    seed,
    filename_prefix=None,
):
    os.makedirs(run_info_dir, exist_ok=True)
    split_paths = {}

    def _to_csv(X_part, y_part, split_name):
        df_out = X_part.reset_index(drop=True).copy()
        y_aligned = pd.Series(y_part).reset_index(drop=True)
        df_out["label"] = y_aligned
        base_name = f"{dataset_name}_{split_name}_seed{seed}.csv"
        if filename_prefix:
            base_name = f"{filename_prefix}_{base_name}"
        out_path = os.path.join(run_info_dir, base_name)
        df_out.to_csv(out_path, index=False)
        print(f"[SplitCSV] Saved {split_name} split to {out_path}")
        split_paths[split_name] = out_path

    _to_csv(X_train, y_train, "train")
    _to_csv(X_val, y_val, "val")
    _to_csv(X_test, y_test, "test")
    return split_paths


def get_split_csv_paths(dataset_name, run_info_dir, seed, filename_prefix=None):
    split_paths = {}
    for split_name in ("train", "val", "test"):
        base_name = f"{dataset_name}_{split_name}_seed{seed}.csv"
        if filename_prefix:
            base_name = f"{filename_prefix}_{base_name}"
        split_paths[split_name] = os.path.join(run_info_dir, base_name)
    return split_paths


# ============================================================
# 9) NSL KDD loader
# ============================================================
def load_nslkdd_dataset(cfg, run_info_dir=".", train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42, existing_split_indices=None):
    _print_split_plan("NSL_KDD", train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
    dataset_dir = _default_dataset_dir()
    train_path = _resolve_dataset_path(
        cfg["train_path"],
        os.path.join(dataset_dir, "KDDTrain+.txt"),
        os.path.join(dataset_dir, "NSL_KDD", "KDDTrain+.txt"),
    )
    test_path = _resolve_dataset_path(
        cfg["test_path"],
        os.path.join(dataset_dir, "KDDTest+.txt"),
        os.path.join(dataset_dir, "NSL_KDD", "KDDTest+.txt"),
    )
    normal_label = cfg.get("normal_label", "normal")
    shuffle_seed = int(seed)

    col_names_43 = [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"
    ]

    print(f"[NSL_KDD] Reading train: {train_path}")
    print(f"[NSL_KDD] Reading test:  {test_path}")

    def _read_nsl_split(path, split_name):
        df_raw = pd.read_csv(path, header=None)
        if df_raw.shape[1] in (42, 43):
            return df_raw

        if df_raw.shape[1] == 1:
            df_raw_ws = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
            if df_raw_ws.shape[1] in (42, 43):
                print(
                    f"[NSL_KDD] {split_name} detected as whitespace-delimited "
                    f"({df_raw_ws.shape[1]} columns)."
                )
                return df_raw_ws

        raise ValueError(f"Unexpected {split_name} columns: {df_raw.shape[1]}")

    df_train_raw = _read_nsl_split(train_path, "train")
    df_test_raw = _read_nsl_split(test_path, "test")

    if df_train_raw.shape[1] == 43:
        df_train_raw.columns = col_names_43
    elif df_train_raw.shape[1] == 42:
        df_train_raw.columns = col_names_43[:-1]

    if df_test_raw.shape[1] == 43:
        df_test_raw.columns = col_names_43
    elif df_test_raw.shape[1] == 42:
        df_test_raw.columns = col_names_43[:-1]

    df_train_raw = df_train_raw.drop(columns=["difficulty"], errors="ignore")
    df_test_raw = df_test_raw.drop(columns=["difficulty"], errors="ignore")

    full_train = df_train_raw.copy()
    full_test = df_test_raw.copy()

    raw_feature_cols = [c for c in full_train.columns if c != "label"]
    feat_cols = list(raw_feature_cols)
    cat_cols = [c for c in ["protocol_type", "service", "flag"] if c in feat_cols]

    joined = pd.concat([full_train[feat_cols], full_test[feat_cols]], axis=0, ignore_index=True)
    for c in cat_cols:
        joined[c] = joined[c].astype(str)

    joined_enc = pd.get_dummies(joined, columns=cat_cols, drop_first=False)
    joined_enc = drop_artificial_index_columns(
        joined_enc,
        dataset_name="NSL_KDD",
        out_dir=run_info_dir,
    )
    joined_enc = clean_numeric_dataframe(joined_enc)
    joined_enc = drop_zero_variance(joined_enc, "NSL_KDD", out_dir=run_info_dir)

    train_enc = joined_enc.iloc[:len(full_train)].reset_index(drop=True)
    test_enc = joined_enc.iloc[len(full_train):].reset_index(drop=True)

    y_train_raw = full_train["label"].astype(str).reset_index(drop=True)
    y_test_raw = full_test["label"].astype(str).reset_index(drop=True)

    y_train_bin = y_train_raw.apply(lambda s: "normal" if s.strip().lower() == normal_label.lower() else "anomaly")
    y_test_bin = y_test_raw.apply(lambda s: "normal" if s.strip().lower() == normal_label.lower() else "anomaly")

    all_X = pd.concat([train_enc, test_enc], axis=0, ignore_index=True)
    all_y = pd.concat([y_train_bin, y_test_bin], axis=0, ignore_index=True)
    X_train_norm_df, y_train, X_val_df, y_val, X_test_df, y_test, split_counts, split_indices = split_normals_attacks_for_val_test(
        X_df=all_X,
        y_series=all_y,
        normal_label="normal",
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        shuffle_seed=shuffle_seed,
        train_normals_only=True,
        existing_split_indices=existing_split_indices,
    )
    y_val_str = y_val.values
    y_test_str = y_test.values

    X_train_norm, X_val, X_test, scaler = fit_scale_train_and_transform(X_train_norm_df, X_val_df, X_test_df)
    assert_no_artificial_index_columns(list(X_train_norm_df.columns), "NSL_KDD")

    _print_split_overview(
        "NSL_KDD",
        split_counts["dataset_total"],
        split_counts["train_window_total"],
        split_counts["train_total"],
        split_counts["val_total"],
        split_counts["test_total"],
    )
    _print_train_window_stats(
        "NSL_KDD",
        split_counts["train_window_total"],
        split_counts["train_window_normals"],
    )
    _print_dist("NSL validation", y_val_str, normal_token="normal")
    _print_dist("NSL test", y_test_str, normal_token="normal")

    return {
        "X_train_norm": X_train_norm,
        "X_val": X_val,
        "y_val_str": y_val_str,
        "X_test": X_test,
        "y_test_str": y_test_str,
        "input_dim": X_train_norm.shape[1],
        "feature_cols": list(X_train_norm_df.columns),
        "raw_feature_cols": list(raw_feature_cols),
        "counts": split_counts,
        "split_indices": split_indices,
        "scaler": scaler,
        "raw_splits": {
            "X_train": X_train_norm_df,
            "y_train": y_train,
            "X_val": X_val_df,
            "y_val": y_val,
            "X_test": X_test_df,
            "y_test": y_test,
        },
    }


# ============================================================
# 10) UNSW loader
# ============================================================
def load_unsw_dataset(cfg, run_info_dir=".", train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42, existing_split_indices=None):
    _print_split_plan("UNSW_NB15", train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
    dataset_dir = _default_dataset_dir()
    train_path = _resolve_dataset_path(
        cfg["train_path"],
        os.path.join(dataset_dir, "UNSW_NB15_training-set.csv"),
        os.path.join(dataset_dir, "UNSW_NB15", "UNSW_NB15_training-set.csv"),
    )
    test_path = _resolve_dataset_path(
        cfg["test_path"],
        os.path.join(dataset_dir, "UNSW_NB15_testing-set.csv"),
        os.path.join(dataset_dir, "UNSW_NB15", "UNSW_NB15_testing-set.csv"),
    )
    normal_label = cfg.get("normal_label", 0)
    shuffle_seed = int(seed)

    print(f"[UNSW_NB15] Reading training set from: {train_path}")
    print(f"[UNSW_NB15] Reading testing set from:  {test_path}")
    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"[UNSW_NB15] Combined raw shape: {df.shape}")

    label_col, _ = _detect_unsw_label_column(df)
    y_series = _unsw_to_binary_label_series(df[label_col]).reset_index(drop=True)

    drop_cols = [label_col]
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "attack_cat" and c not in drop_cols:
            drop_cols.append(c)
    print(f"[UNSW_NB15] Dropping label columns from features: {drop_cols}")
    raw_feature_cols = [c for c in df.columns if c not in set(drop_cols)]

    feature_df = _prepare_mixed_tabular_features(
        df,
        "UNSW_NB15",
        label_cols_to_drop=drop_cols,
        run_info_dir=run_info_dir,
    )

    X_train_df, y_train, X_val_df, y_val, X_test_df, y_test, split_counts, split_indices = split_normals_attacks_for_val_test(
        X_df=feature_df,
        y_series=y_series,
        normal_label=normal_label,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        shuffle_seed=shuffle_seed,
        train_normals_only=True,
        existing_split_indices=existing_split_indices,
    )

    y_val_str = binary_strings_from_numeric_labels(y_val, normal_label).values
    y_test_str = binary_strings_from_numeric_labels(y_test, normal_label).values

    X_train_norm, X_val, X_test, scaler = fit_scale_train_and_transform(X_train_df, X_val_df, X_test_df)
    assert_no_artificial_index_columns(list(X_train_df.columns), "UNSW_NB15")

    _print_split_overview(
        "UNSW_NB15",
        split_counts["dataset_total"],
        split_counts["train_window_total"],
        split_counts["train_total"],
        split_counts["val_total"],
        split_counts["test_total"],
    )
    _print_train_window_stats(
        "UNSW_NB15",
        split_counts["train_window_total"],
        split_counts["train_window_normals"],
    )
    _print_dist("UNSW validation", y_val_str, normal_token="normal")
    _print_dist("UNSW test", y_test_str, normal_token="normal")

    return {
        "X_train_norm": X_train_norm,
        "X_val": X_val,
        "y_val_str": y_val_str,
        "X_test": X_test,
        "y_test_str": y_test_str,
        "input_dim": X_train_norm.shape[1],
        "feature_cols": list(X_train_df.columns),
        "raw_feature_cols": list(raw_feature_cols),
        "counts": split_counts,
        "split_indices": split_indices,
        "scaler": scaler,
        "raw_splits": {
            "X_train": X_train_df,
            "y_train": y_train,
            "X_val": X_val_df,
            "y_val": y_val,
            "X_test": X_test_df,
            "y_test": y_test,
        },
    }


# ============================================================
# 11) CTU13 loader
# ============================================================
def load_ctu13_dataset(cfg, run_info_dir=".", train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42, existing_split_indices=None):
    _print_split_plan("CTU13", train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
    dataset_dir = _default_dataset_dir()
    normal_path = _resolve_dataset_path(
        cfg["normal_path"],
        os.path.join(dataset_dir, "CTU13_Normal_Traffic.csv"),
        os.path.join(dataset_dir, "CTU13", "CTU13_Normal_Traffic.csv"),
    )
    attack_path = _resolve_dataset_path(
        cfg["attack_path"],
        os.path.join(dataset_dir, "CTU13_Attack_Traffic.csv"),
        os.path.join(dataset_dir, "CTU13", "CTU13_Attack_Traffic.csv"),
    )
    normal_label = cfg.get("normal_label", 0)
    shuffle_seed = int(seed)

    print(f"[CTU13] Reading normal traffic from: {normal_path}")
    print(f"[CTU13] Reading attack traffic from: {attack_path}")

    df_normal = pd.read_csv(normal_path, low_memory=False)
    df_attack = pd.read_csv(attack_path, low_memory=False)
    df_normal.columns = [str(c).strip() for c in df_normal.columns]
    df_attack.columns = [str(c).strip() for c in df_attack.columns]

    label_col = None
    for c in df_normal.columns:
        if str(c).strip().lower() == "label":
            label_col = c
            break

    if label_col is None:
        raise ValueError(
            "[CTU13] Existing label column 'Label' was not found. "
            "CTU13 loader expects the CSV to already contain labels."
        )

    if label_col not in df_attack.columns:
        raise ValueError(
            f"[CTU13] Label column '{label_col}' found in normal CSV but missing from attack CSV."
        )

    all_cols = sorted(set(df_normal.columns).union(set(df_attack.columns)))
    df_normal = df_normal.reindex(columns=all_cols)
    df_attack = df_attack.reindex(columns=all_cols)

    df = pd.concat([df_normal, df_attack], axis=0, ignore_index=True, sort=False)
    print(f"[CTU13] Combined raw shape: {df.shape}")

    y_series = ctu13_label_to_binary(df[label_col]).reset_index(drop=True)
    label_cols_to_drop = [label_col]
    for c in df.columns:
        cl = str(c).strip().lower()
        if (
            cl in {"class", "attack", "attack_cat", "target", "category"}
            and c not in label_cols_to_drop
        ):
            label_cols_to_drop.append(c)

    print(f"[CTU13] Dropping label columns from features: {label_cols_to_drop}")
    raw_feature_cols = [c for c in df.columns if c not in set(label_cols_to_drop)]
    X_df = _prepare_mixed_tabular_features(
        df,
        "CTU13",
        label_cols_to_drop=label_cols_to_drop,
        run_info_dir=run_info_dir,
    )

    X_train_df, y_train, X_val_df, y_val, X_test_df, y_test, split_counts, split_indices = split_normals_attacks_for_val_test(
        X_df=X_df,
        y_series=y_series,
        normal_label=normal_label,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        shuffle_seed=shuffle_seed,
        train_normals_only=True,
        existing_split_indices=existing_split_indices,
    )

    y_val_str = binary_strings_from_numeric_labels(y_val, normal_label).values
    y_test_str = binary_strings_from_numeric_labels(y_test, normal_label).values

    X_train_norm, X_val, X_test, scaler = fit_scale_train_and_transform(X_train_df, X_val_df, X_test_df)
    assert_no_artificial_index_columns(list(X_train_df.columns), "CTU13")
    assert_no_label_like_columns(list(X_train_df.columns), "CTU13")

    _print_split_overview(
        "CTU13",
        split_counts["dataset_total"],
        split_counts["train_window_total"],
        split_counts["train_total"],
        split_counts["val_total"],
        split_counts["test_total"],
    )
    _print_train_window_stats(
        "CTU13",
        split_counts["train_window_total"],
        split_counts["train_window_normals"],
    )
    _print_dist("CTU13 validation", y_val_str, normal_token="normal")
    _print_dist("CTU13 test", y_test_str, normal_token="normal")

    return {
        "X_train_norm": X_train_norm,
        "X_val": X_val,
        "y_val_str": y_val_str,
        "X_test": X_test,
        "y_test_str": y_test_str,
        "input_dim": X_train_norm.shape[1],
        "feature_cols": list(X_train_df.columns),
        "raw_feature_cols": list(raw_feature_cols),
        "counts": split_counts,
        "split_indices": split_indices,
        "scaler": scaler,
        "raw_splits": {
            "X_train": X_train_df,
            "y_train": y_train,
            "X_val": X_val_df,
            "y_val": y_val,
            "X_test": X_test_df,
            "y_test": y_test,
        },
    }


# ============================================================
# 12) HIKARI loader
# ============================================================
def hikari_detect_label_column(df):
    cols = [c.strip() for c in df.columns]
    for c in cols:
        if c.lower() == "traffic_category":
            return c
    for c in cols:
        if c.lower() == "label":
            return c
    for c in cols:
        if "label" in c.lower():
            return c
    for c in cols:
        if "attack" in c.lower():
            return c
    raise ValueError(f"[HIKARI] Could not find label column. Columns: {cols}")


def hikari_make_binary(df, label_col):
    s = df[label_col]
    if np.issubdtype(s.dtype, np.number):
        df["is_attack"] = s.apply(lambda v: 1 if float(v) == 1.0 else 0)
        return df

    def mapper(v):
        y = str(v).strip().upper()
        if ("BACKGROUND" in y) or ("BENIGN" in y) or ("NORMAL" in y):
            return 0
        return 1

    df["is_attack"] = s.apply(mapper)
    return df


def load_hikari_dataset(cfg, run_info_dir=".", train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42, existing_split_indices=None):
    _print_split_plan("HIKARI2021", train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
    dataset_dir = _default_dataset_dir()
    data_path = _resolve_dataset_path(
        cfg["data_path"],
        os.path.join(dataset_dir, "HIKARI2021.csv"),
        os.path.join(dataset_dir, "HIKARI2021", "HIKARI2021.csv"),
    )
    normal_label = cfg.get("normal_label", 0)
    shuffle_seed = int(seed)

    print(f"[HIKARI2021] Reading dataset from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    label_col = hikari_detect_label_column(df)
    df = hikari_make_binary(df, label_col)

    y_all = df["is_attack"].values.astype(int)

    drop_cols = [label_col, "is_attack"]
    extra_label_like = [c for c in df.columns if c.lower() in ("traffic_category", "label", "attack") and c not in drop_cols]
    drop_cols.extend(extra_label_like)
    raw_feature_cols = [c for c in df.columns if c not in set(drop_cols)]

    X_df = df.drop(columns=list(set(drop_cols)), errors="ignore")
    X_df = drop_artificial_index_columns(
        X_df,
        dataset_name="HIKARI2021",
        out_dir=run_info_dir,
    )
    X_df = X_df.select_dtypes(include=[np.number])

    if X_df.shape[1] == 0:
        raise ValueError("[HIKARI] No numeric features found after dropping labels")

    X_df = clean_numeric_dataframe(X_df)
    X_df = drop_zero_variance(X_df, "HIKARI2021", out_dir=run_info_dir)

    y_series = pd.Series(y_all).reset_index(drop=True)

    X_train_df, y_train, X_val_df, y_val, X_test_df, y_test, split_counts, split_indices = split_normals_attacks_for_val_test(
        X_df=X_df,
        y_series=y_series,
        normal_label=normal_label,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        shuffle_seed=shuffle_seed,
        train_normals_only=True,
        existing_split_indices=existing_split_indices,
    )

    y_val_str = binary_strings_from_numeric_labels(y_val, normal_label).values
    y_test_str = binary_strings_from_numeric_labels(y_test, normal_label).values

    X_train_norm, X_val, X_test, scaler = fit_scale_train_and_transform(X_train_df, X_val_df, X_test_df)
    assert_no_artificial_index_columns(list(X_train_df.columns), "HIKARI2021")

    _print_split_overview(
        "HIKARI2021",
        split_counts["dataset_total"],
        split_counts["train_window_total"],
        split_counts["train_total"],
        split_counts["val_total"],
        split_counts["test_total"],
    )
    _print_train_window_stats(
        "HIKARI2021",
        split_counts["train_window_total"],
        split_counts["train_window_normals"],
    )
    _print_dist("HIKARI validation", y_val_str, normal_token="normal")
    _print_dist("HIKARI test", y_test_str, normal_token="normal")

    return {
        "X_train_norm": X_train_norm,
        "X_val": X_val,
        "y_val_str": y_val_str,
        "X_test": X_test,
        "y_test_str": y_test_str,
        "input_dim": X_train_norm.shape[1],
        "feature_cols": list(X_train_df.columns),
        "raw_feature_cols": list(raw_feature_cols),
        "counts": split_counts,
        "split_indices": split_indices,
        "scaler": scaler,
        "raw_splits": {
            "X_train": X_train_df,
            "y_train": y_train,
            "X_val": X_val_df,
            "y_val": y_val,
            "X_test": X_test_df,
            "y_test": y_test,
        },
    }


# ============================================================
# 13) Dataset dispatcher
# ============================================================
def load_dataset(dataset_name, dataset_cfg, run_info_dir=".", train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42, existing_split_indices=None):
    """Dispatch to the dataset-specific loader while keeping split behavior consistent across datasets."""
    dataset_name = dataset_name.upper()
    if dataset_name == "NSL_KDD":
        return load_nslkdd_dataset(dataset_cfg, run_info_dir=run_info_dir, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed, existing_split_indices=existing_split_indices)
    if dataset_name == "UNSW_NB15":
        return load_unsw_dataset(dataset_cfg, run_info_dir=run_info_dir, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed, existing_split_indices=existing_split_indices)
    if dataset_name == "CTU13":
        return load_ctu13_dataset(dataset_cfg, run_info_dir=run_info_dir, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed, existing_split_indices=existing_split_indices)
    if dataset_name == "HIKARI2021":
        return load_hikari_dataset(dataset_cfg, run_info_dir=run_info_dir, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed, existing_split_indices=existing_split_indices)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


# ============================================================
# 14) Dataset configs
# ============================================================
DATASET_CONFIGS = {
    "NSL_KDD": {
        "train_path": os.path.join(_default_dataset_dir(), "KDDTrain+.txt"),
        "test_path": os.path.join(_default_dataset_dir(), "KDDTest+.txt"),
        "normal_label": "normal",
        "shuffle_seed": 42,
        "ae": {
            "latent_dim": 24,
            "enc_units": (96, 64),
            "dec_units": (64, 96),
            "activation": "relu",
            "output_activation": "sigmoid",
            "l2_reg": 0.0,
            "dropout": 0.0,
            "batch_norm": False,
            "optimizer": "adam",
            "lr": 1e-3,
            "loss": "mse",
            "epochs": 20,
            "batch_size": 2048,
        },
        "dae": {
            "noise_std_init": 0.2,
            "noise_std_final": 0.02,
        }
    },
    "UNSW_NB15": {
        "train_path": os.path.join(_default_dataset_dir(), "UNSW_NB15_training-set.csv"),
        "test_path": os.path.join(_default_dataset_dir(), "UNSW_NB15_testing-set.csv"),
        "normal_label": 0,
        "shuffle_seed": 42,
        "ae": {
            "latent_dim": 64,
            "enc_units": (192, 128),
            "dec_units": (128, 192),
            "activation": "relu",
            "output_activation": "sigmoid",
            "l2_reg": 0.0,
            "dropout": 0.0,
            "batch_norm": False,
            "optimizer": "adam",
            "lr": 1e-3,
            "loss": "mse",
            "epochs": 20,
            "batch_size": 2048,
        },
        "dae": {
            "noise_std_init": 0.5,
            "noise_std_final": 0.25,
        }
    },
    "CTU13": {
        "attack_path": os.path.join(_default_dataset_dir(), "CTU13_Attack_Traffic.csv"),
        "normal_path": os.path.join(_default_dataset_dir(), "CTU13_Normal_Traffic.csv"),
        "normal_label": 0,
        "shuffle_seed": 42,
        "ae": {
            "latent_dim": 16,
            "enc_units": (48, 32),
            "dec_units": (32, 48),
            "activation": "relu",
            "output_activation": "sigmoid",
            "l2_reg": 0.0,
            "dropout": 0.0,
            "batch_norm": False,
            "optimizer": "adam",
            "lr": 1e-3,
            "loss": "mse",
            "epochs": 20,
            "batch_size": 2048,
        },
        "dae": {
            "noise_std_init": 0.5,
            "noise_std_final": 0.25,
        }
    },
    "HIKARI2021": {
        "data_path": os.path.join(_default_dataset_dir(), "HIKARI2021.csv"),
        "normal_label": 0,
        "shuffle_seed": 42,
        "ae": {
            "latent_dim": 32,
            "enc_units": (80, 64),
            "dec_units": (64, 80),
            "activation": "relu",
            "output_activation": "sigmoid",
            "l2_reg": 0.0,
            "dropout": 0.0,
            "batch_norm": False,
            "optimizer": "adam",
            "lr": 1e-3,
            "loss": "mse",
            "epochs": 20,
            "batch_size": 2048,
        },
        "dae": {
            "noise_std_init": 0.2,
            "noise_std_final": 0.02,
        }
    }
}


# ============================================================
# 15) Shared schedule grid
# ============================================================
CONST_SIGMA_VALUES = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90,
]

NOISE_RANGE_GRID = [
    {"name": "R1",  "sigma_start": 0.9, "sigma_end": 0.8, "sigma_min": 0.8, "decay_rate": 3.0},
    {"name": "R2",  "sigma_start": 0.9, "sigma_end": 0.7, "sigma_min": 0.7, "decay_rate": 3.0},
    {"name": "R3",  "sigma_start": 0.9, "sigma_end": 0.6, "sigma_min": 0.6, "decay_rate": 3.0},
    {"name": "R4",  "sigma_start": 0.9, "sigma_end": 0.5, "sigma_min": 0.5, "decay_rate": 3.0},
    {"name": "R5",  "sigma_start": 0.9, "sigma_end": 0.4, "sigma_min": 0.4, "decay_rate": 3.0},
    {"name": "R6",  "sigma_start": 0.9, "sigma_end": 0.3, "sigma_min": 0.3, "decay_rate": 3.0},
    {"name": "R7",  "sigma_start": 0.9, "sigma_end": 0.2, "sigma_min": 0.2, "decay_rate": 3.0},
    {"name": "R8",  "sigma_start": 0.9, "sigma_end": 0.1, "sigma_min": 0.1, "decay_rate": 3.0},
    {"name": "R9",  "sigma_start": 0.8, "sigma_end": 0.7, "sigma_min": 0.7, "decay_rate": 3.0},
    {"name": "R10", "sigma_start": 0.8, "sigma_end": 0.6, "sigma_min": 0.6, "decay_rate": 3.0},
    {"name": "R11", "sigma_start": 0.8, "sigma_end": 0.5, "sigma_min": 0.5, "decay_rate": 3.0},
    {"name": "R12", "sigma_start": 0.8, "sigma_end": 0.4, "sigma_min": 0.4, "decay_rate": 3.0},
    {"name": "R13", "sigma_start": 0.8, "sigma_end": 0.3, "sigma_min": 0.3, "decay_rate": 3.0},
    {"name": "R14", "sigma_start": 0.8, "sigma_end": 0.2, "sigma_min": 0.2, "decay_rate": 3.0},
    {"name": "R15", "sigma_start": 0.8, "sigma_end": 0.1, "sigma_min": 0.1, "decay_rate": 3.0},
    {"name": "R16", "sigma_start": 0.7, "sigma_end": 0.6, "sigma_min": 0.6, "decay_rate": 3.0},
    {"name": "R17", "sigma_start": 0.7, "sigma_end": 0.5, "sigma_min": 0.5, "decay_rate": 3.0},
    {"name": "R18", "sigma_start": 0.7, "sigma_end": 0.4, "sigma_min": 0.4, "decay_rate": 3.0},
    {"name": "R19", "sigma_start": 0.7, "sigma_end": 0.3, "sigma_min": 0.3, "decay_rate": 3.0},
    {"name": "R20", "sigma_start": 0.7, "sigma_end": 0.2, "sigma_min": 0.2, "decay_rate": 3.0},
    {"name": "R21", "sigma_start": 0.7, "sigma_end": 0.1, "sigma_min": 0.1, "decay_rate": 3.0},
    {"name": "R22", "sigma_start": 0.6, "sigma_end": 0.5, "sigma_min": 0.5, "decay_rate": 3.0},
    {"name": "R23", "sigma_start": 0.6, "sigma_end": 0.4, "sigma_min": 0.4, "decay_rate": 3.0},
    {"name": "R24", "sigma_start": 0.6, "sigma_end": 0.3, "sigma_min": 0.3, "decay_rate": 3.0},
    {"name": "R25", "sigma_start": 0.6, "sigma_end": 0.2, "sigma_min": 0.2, "decay_rate": 3.0},
    {"name": "R26", "sigma_start": 0.6, "sigma_end": 0.1, "sigma_min": 0.1, "decay_rate": 3.0},
    {"name": "R27", "sigma_start": 0.5, "sigma_end": 0.4, "sigma_min": 0.4, "decay_rate": 3.0},
    {"name": "R28", "sigma_start": 0.5, "sigma_end": 0.3, "sigma_min": 0.3, "decay_rate": 3.0},
    {"name": "R29", "sigma_start": 0.5, "sigma_end": 0.2, "sigma_min": 0.2, "decay_rate": 3.0},
    {"name": "R30", "sigma_start": 0.5, "sigma_end": 0.1, "sigma_min": 0.1, "decay_rate": 3.0},
    {"name": "R31", "sigma_start": 0.4, "sigma_end": 0.3, "sigma_min": 0.3, "decay_rate": 3.0},
    {"name": "R32", "sigma_start": 0.4, "sigma_end": 0.2, "sigma_min": 0.2, "decay_rate": 3.0},
    {"name": "R33", "sigma_start": 0.4, "sigma_end": 0.1, "sigma_min": 0.1, "decay_rate": 3.0},
    {"name": "R34", "sigma_start": 0.3, "sigma_end": 0.2, "sigma_min": 0.2, "decay_rate": 3.0},
    {"name": "R35", "sigma_start": 0.3, "sigma_end": 0.1, "sigma_min": 0.1, "decay_rate": 3.0},
    {"name": "R36", "sigma_start": 0.2, "sigma_end": 0.1, "sigma_min": 0.1, "decay_rate": 3.0},
]

ALL_DATASETS = ["NSL_KDD", "UNSW_NB15", "CTU13", "HIKARI2021"]
ALL_SCHEDULES = [
    "constant",
    "linear",
    "exponential",
    "cosine",
    "fibonacci",
    "sigmoid",
    "cauchy",
    "laplace",
    "logistic",
]

OUTPUT_ROOT_DIR = "generated_outputs"
OUTPUT_SUBDIRS = {
    "stats": "statistical_outputs",
    "runs": "runs",
}
# Default seeds used when --seeds is not provided.
DEFAULT_SEEDS = [42, 7, 123]
# Main-study fixed noise type default.
DEFAULT_NOISE_TYPE = "gaussian"


def ensure_output_directories(root_dir=OUTPUT_ROOT_DIR):
    paths = {}
    os.makedirs(root_dir, exist_ok=True)
    for key, sub in OUTPUT_SUBDIRS.items():
        path = os.path.join(root_dir, sub)
        os.makedirs(path, exist_ok=True)
        paths[key] = path
    return paths


def make_run_id(dataset_name, seed):
    return f"{dataset_name}_seed{int(seed)}"


def make_seed_run_folder_name(dataset_name, seed):
    return f"{dataset_name}_seed_{int(seed)}"


def make_model_seed_folder_name(ae_variant, seed):
    ae_variant_label = normalize_ae_variant(ae_variant)
    return f"{ae_variant_label}_seed_{int(seed)}"


def ensure_run_output_directory(runs_root_dir, dataset_name, seed, ae_variant="FF_DAE"):
    ae_variant_label = normalize_ae_variant(ae_variant)
    dataset_dir = os.path.join(runs_root_dir, dataset_name)

    seed_folder_name = make_seed_run_folder_name(dataset_name, seed)
    shared_seed_run_dir = os.path.join(dataset_dir, seed_folder_name)

    model_folder_name = make_model_seed_folder_name(ae_variant_label, seed)
    model_seed_run_dir = os.path.join(shared_seed_run_dir, model_folder_name)

    subdirs = {
        "root": model_seed_run_dir,
        "plots": os.path.join(model_seed_run_dir, "plots"),
        "metrics": os.path.join(model_seed_run_dir, "metrics"),
        "run_info": os.path.join(model_seed_run_dir, "run_info"),
        "splits": os.path.join(shared_seed_run_dir, "splits"),
    }

    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    return subdirs


def save_split_indices(split_indices, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_paths = {}
    canonical_paths = get_split_index_paths(out_dir)
    for split_name, out_path in canonical_paths.items():
        if split_name not in split_indices:
            continue
        idx_df = pd.DataFrame({"Index": pd.Series(split_indices[split_name], dtype=int)})
        idx_df.to_csv(out_path, index=False)
        out_paths[split_name] = out_path
    for split_name, idx_values in split_indices.items():
        if split_name in out_paths:
            continue
        idx_df = pd.DataFrame({"Index": pd.Series(idx_values, dtype=int)})
        out_path = os.path.join(out_dir, f"{split_name}_indices.csv")
        idx_df.to_csv(out_path, index=False)
        out_paths[split_name] = out_path
    return out_paths


def get_split_index_paths(splits_dir):
    """Return canonical CSV paths for saved train-window/train/validation/test split indices."""
    return {
        "train_window": os.path.join(splits_dir, "train_window_indices.csv"),
        "train": os.path.join(splits_dir, "train_indices.csv"),
        "val": os.path.join(splits_dir, "val_indices.csv"),
        "test": os.path.join(splits_dir, "test_indices.csv"),
    }


def split_indices_exist(splits_dir):
    """Return True only when all expected split-index CSV files exist."""
    paths = get_split_index_paths(splits_dir)
    present = {name: os.path.exists(path) for name, path in paths.items()}
    if all(present.values()):
        return True
    missing_train_window = present["train"] and present["val"] and present["test"] and not present["train_window"]
    if missing_train_window:
        raise ValueError(
            "Saved split indices are missing train_window_indices.csv. "
            "Delete the incomplete splits folder "
            "or regenerate the split so train_window_indices.csv can be created."
        )
    return False


def _read_split_index_csv(path, split_name):
    if not os.path.exists(path):
        raise ValueError(f"Saved {split_name} split index file does not exist: {path}")

    df = pd.read_csv(path)
    if "Index" not in df.columns:
        raise ValueError(f"Saved {split_name} split index file must contain an 'Index' column: {path}")

    raw_values = df["Index"]
    numeric_values = pd.to_numeric(raw_values, errors="raise")
    if numeric_values.isna().any():
        raise ValueError(f"Saved {split_name} split indices contain missing values: {path}")

    arr = numeric_values.to_numpy()
    if not np.all(np.equal(arr, arr.astype(np.int64))):
        raise ValueError(f"Saved {split_name} split indices must be integer values: {path}")
    return arr.astype(np.int64)


def load_split_indices(splits_dir):
    """Load saved train-window/train/validation/test split indices from CSV files."""
    paths = get_split_index_paths(splits_dir)
    split_indices = {
        split_name: _read_split_index_csv(path, split_name)
        for split_name, path in paths.items()
    }
    _validate_split_indices(split_indices)
    return split_indices


def _validate_split_indices(split_indices, n_total=None):
    required = ("train_window", "train", "val", "test")
    missing = [name for name in required if name not in split_indices]
    if missing:
        raise ValueError(f"Split indices are missing required split(s): {missing}")

    normalized = {}
    for split_name in required:
        arr = np.asarray(split_indices[split_name])
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{split_name} split indices must be an integer array; got dtype={arr.dtype}")
        arr = arr.astype(np.int64, copy=False).reshape(-1)
        if n_total is not None and len(arr) > 0:
            out_of_range = arr[(arr < 0) | (arr >= n_total)]
            if len(out_of_range) > 0:
                sample = out_of_range[:10].tolist()
                raise ValueError(
                    f"{split_name} split indices contain values outside dataset range [0, {n_total - 1}]. "
                    f"Examples: {sample}"
                )
        normalized[split_name] = arr

    for split_name, arr in normalized.items():
        if len(arr) == 0:
            raise ValueError(f"{split_name} split indices must not be empty")
        unique_count = len(np.unique(arr))
        if unique_count != len(arr):
            raise ValueError(f"{split_name} split indices contain duplicate values")

    train_window_set = set(normalized["train_window"].tolist())
    train_set = set(normalized["train"].tolist())
    val_set = set(normalized["val"].tolist())
    test_set = set(normalized["test"].tolist())

    if not train_set.issubset(train_window_set):
        raise ValueError("train_indices must be a subset of train_window_indices")

    overlap_checks = (
        ("train", train_set, "val", val_set),
        ("train", train_set, "test", test_set),
        ("val", val_set, "test", test_set),
        ("train_window", train_window_set, "val", val_set),
        ("train_window", train_window_set, "test", test_set),
    )
    for left_name, left_set, right_name, right_set in overlap_checks:
        overlap = left_set.intersection(right_set)
        if overlap:
            idx_int = int(next(iter(overlap)))
            raise ValueError(
                f"Split indices overlap: index {idx_int} appears in both "
                f"{left_name} and {right_name}"
            )

    return normalized


def apply_existing_split_indices(existing_split_indices, n_total=None, **kwargs):
    """Validate loaded split indices and return train-window/train/validation/test integer arrays.

    When dataset components are provided, this can also build the standard dataset
    dictionary shape returned by load_dataset(...). The runner currently relies on
    the tuple-returning split helper, but this richer path keeps split reuse logic
    available for future dataset-loader refactors.
    """
    normalized = _validate_split_indices(existing_split_indices, n_total=n_total)
    train_window_idx = normalized["train_window"]
    train_idx = normalized["train"]
    val_idx = normalized["val"]
    test_idx = normalized["test"]

    if "X_df" not in kwargs or "y_series" not in kwargs:
        return train_window_idx, train_idx, val_idx, test_idx

    X_df = kwargs["X_df"]
    y_series = kwargs["y_series"]
    normal_label = kwargs["normal_label"]
    raw_feature_cols = kwargs.get("raw_feature_cols", list(X_df.columns))
    y_to_str_func = kwargs.get("y_to_str_func")

    X_train_df = X_df.iloc[train_idx].copy()
    y_train = y_series.iloc[train_idx].copy()
    X_val_df = X_df.iloc[val_idx].copy()
    y_val = y_series.iloc[val_idx].copy()
    X_test_df = X_df.iloc[test_idx].copy()
    y_test = y_series.iloc[test_idx].copy()

    y_train_arr = y_train.values
    y_val_arr = y_val.values
    y_test_arr = y_test.values
    train_normals = int(np.sum(y_train_arr == normal_label))
    val_normals = int(np.sum(y_val_arr == normal_label))
    test_normals = int(np.sum(y_test_arr == normal_label))
    train_window_normals = int(np.sum((y_series.values == normal_label)[train_window_idx]))
    split_counts = {
        "dataset_total": int(len(X_df)),
        "train_window_total": int(len(train_window_idx)),
        "train_window_normals": train_window_normals,
        "train_window_anomalies": int(len(train_window_idx) - train_window_normals),
        "train_total": int(len(train_idx)),
        "val_total": int(len(val_idx)),
        "test_total": int(len(test_idx)),
        "train_normals": train_normals,
        "train_anomalies": int(len(train_idx) - train_normals),
        "val_normals": val_normals,
        "val_anomalies": int(len(val_idx) - val_normals),
        "test_normals": test_normals,
        "test_anomalies": int(len(test_idx) - test_normals),
    }
    X_train_norm, X_val, X_test, scaler = fit_scale_train_and_transform(X_train_df, X_val_df, X_test_df)
    if y_to_str_func is None:
        y_val_str = y_val.values
        y_test_str = y_test.values
    else:
        y_val_str = y_to_str_func(y_val).values
        y_test_str = y_to_str_func(y_test).values

    return {
        "X_train_norm": X_train_norm,
        "X_val": X_val,
        "y_val_str": y_val_str,
        "X_test": X_test,
        "y_test_str": y_test_str,
        "input_dim": X_train_norm.shape[1],
        "feature_cols": list(X_train_df.columns),
        "raw_feature_cols": list(raw_feature_cols),
        "counts": split_counts,
        "split_indices": {
            "train_window": train_window_idx.copy(),
            "train": train_idx.copy(),
            "val": val_idx.copy(),
            "test": test_idx.copy(),
        },
        "scaler": scaler,
        "raw_splits": {
            "X_train": X_train_df,
            "y_train": y_train,
            "X_val": X_val_df,
            "y_val": y_val,
            "X_test": X_test_df,
            "y_test": y_test,
        },
    }

def save_scaler(scaler, out_path):
    _ensure_parent_dir(out_path)
    with open(out_path, "wb") as f:
        pickle.dump(scaler, f)
    return out_path


def copy_or_link_artifact_into_run_dir(src_path, dst_dir):
    if not src_path or not os.path.exists(src_path):
        return None
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return dst_path
    shutil.copy2(src_path, dst_path)
    return dst_path


def save_run_manifest(manifest_path, manifest_payload):
    _ensure_parent_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    return manifest_path


def save_dataset_run_metadata(
    meta_path,
    run_id,
    dataset_name,
    dataset_cfg,
    dataset_loaded_info,
    selected_schedules,
    select_metric,
    schedule_defs,
    seed,
    split_fractions,
    fixed_experiment_config,
    ae_variant="FF_DAE",
):
    """Persist run-level metadata and fixed-config schedule sweep context."""
    schedule_sweep = []
    for schedule_def in schedule_defs:
        schedule_sweep.append({
            "model_name": schedule_def.get("model_name"),
            "range_name": schedule_def.get("range_name"),
            "schedule_name": schedule_def.get("schedule_name"),
            "schedule_params": schedule_def.get("schedule_params"),
            "sigma_start": schedule_def.get("sigma_start"),
            "sigma_end": schedule_def.get("sigma_end"),
            "sigma_min": schedule_def.get("sigma_min"),
            "decay_rate": schedule_def.get("decay_rate"),
        })

    raw_cols = list(dataset_loaded_info.get("raw_feature_cols", []))
    final_cols = list(dataset_loaded_info.get("feature_cols", []))

    ae_variant_label = normalize_ae_variant(ae_variant)
    meta = {
        "RunID": run_id,
        "dataset": dataset_name,
        "selection_metric": select_metric,
        "AEVariant": ae_variant_label,
        "ae_variant": ae_variant_label,
        "run_scope": "RunID identifies one dataset + one seed execution; ConfigID identifies one DAE experiment within that run.",
        "deterministic_init_seed_rule": "sha256(Dataset, Seed, AEVariant, BaseSeed) mapped to a 31-bit integer; same initialization is used for all schedules within the same Dataset, Seed, and AEVariant.",
        "selected_schedules": selected_schedules,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": dataset_loaded_info.get("counts", {}),
        "input_dim": int(dataset_loaded_info.get("input_dim", 0)),
        "feature_count": len(final_cols),
        "feature_columns": final_cols,
        "raw_feature_count": len(raw_cols),
        "raw_feature_columns": raw_cols,
        "feature_column_note": "feature_count and feature_columns refer to final model input features after preprocessing. raw_feature_count and raw_feature_columns refer to original non-label columns before preprocessing.",
        "schedule_sweep": schedule_sweep,
        "schedule_run_count": len(schedule_sweep),
        "fixed_experiment_config": fixed_experiment_config,
        "fixed_parameters": {
            "shuffle_seed": int(seed),
            "normal_label": dataset_cfg.get("normal_label"),
            "split_fractions": split_fractions,
            "ae": dataset_cfg.get("ae", {}),
            "dae_base_config": dataset_cfg.get("dae", {}),
            "noise_range_grid_size": len(NOISE_RANGE_GRID),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def save_dataset_info_json(
    run_info_dir,
    dataset_name,
    input_dim,
    feature_columns,
    train_shape,
    val_shape,
    test_shape,
    counts,
    raw_feature_columns=None,
    normal_label=None,
    shuffle_seed=None,
    split_fractions=None,
    run_id=None,
    split_indices=None,
    ae_variant="FF_DAE",
):
    """Save dataset split/count metadata for one dataset+seed execution."""
    os.makedirs(run_info_dir, exist_ok=True)

    dataset_total = int(counts.get("dataset_total", 0))
    train_window_total = int(counts.get("train_window_total", 0))
    train_window_normals = int(counts.get("train_window_normals", 0))
    train_total = int(counts.get("train_total", 0))
    val_total = int(counts.get("val_total", 0))
    test_total = int(counts.get("test_total", 0))
    val_normals = int(counts.get("val_normals", 0))
    val_anomalies = int(counts.get("val_anomalies", 0))
    test_normals = int(counts.get("test_normals", 0))
    test_anomalies = int(counts.get("test_anomalies", 0))

    raw_cols = list(raw_feature_columns or [])
    final_cols = list(feature_columns or [])

    ae_variant_label = normalize_ae_variant(ae_variant)
    payload = {
        "dataset": dataset_name,
        "RunID": run_id,
        "AEVariant": ae_variant_label,
        "ae_variant": ae_variant_label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dim": int(input_dim),
        "feature_count": len(final_cols),
        "feature_columns": final_cols,
        "raw_feature_count": len(raw_cols),
        "raw_feature_columns": raw_cols,
        "feature_column_note": "feature_count and feature_columns refer to final model input features after preprocessing. raw_feature_count and raw_feature_columns refer to original non-label columns before preprocessing.",
        "train_shape": list(train_shape),
        "val_shape": list(val_shape),
        "test_shape": list(test_shape),
        "split_info": {
            "dataset_total": dataset_total,
            "initial_train_window_total": train_window_total,
            "initial_train_window_percent": _pct(train_window_total, dataset_total),
            "benign_in_initial_train_window": train_window_normals,
            "benign_percent_in_initial_train_window": _pct(train_window_normals, train_window_total),
            "ae_training_benign_subset_total": train_total,
            "ae_training_benign_subset_percent_of_full_dataset": _pct(train_total, dataset_total),
            "validation_mixed_total": val_total,
            "validation_percent": _pct(val_total, dataset_total),
            "test_mixed_total": test_total,
            "test_percent": _pct(test_total, dataset_total),
            "validation_normal_total": val_normals,
            "validation_normal_percent": _pct(val_normals, val_total),
            "validation_anomaly_total": val_anomalies,
            "validation_anomaly_percent": _pct(val_anomalies, val_total),
            "test_normal_total": test_normals,
            "test_normal_percent": _pct(test_normals, test_total),
            "test_anomaly_total": test_anomalies,
            "test_anomaly_percent": _pct(test_anomalies, test_total),
        },
        "normal_label": normal_label,
        "shuffle_seed": shuffle_seed,
        "split_fractions": split_fractions or {},
        "split_indices": split_indices or {},
    }

    seed_suffix = "na" if shuffle_seed is None else str(int(shuffle_seed))
    dataset_token = str(dataset_name)
    out_path = os.path.join(run_info_dir, f"{dataset_token}_dataset_info_seed{seed_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[DatasetInfo] Saved dataset split info to {out_path}")
    return out_path


def parse_cli():
    """Parse command-line options for dataset/schedule selection and reproducible grid execution."""
    parser = argparse.ArgumentParser(
        description="Fixed-config paired decay-vs-CONST DAE runner with validation threshold selection"
    )

    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["ALL"],
        help="Dataset(s) to run: ALL NSL_KDD UNSW_NB15 CTU13 HIKARI2021"
    )

    parser.add_argument(
        "--schedules",
        nargs="+",
        default=["ALL"],
        help="Schedules to run: ALL CONSTANT LINEAR EXPONENTIAL COSINE FIBONACCI SIGMOID CAUCHY LAPLACE LOGISTIC."
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plot generation"
    )

    parser.add_argument(
        "--save_epoch_recon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-epoch train/validation reconstruction rows into the dataset train_val metrics CSV (enabled by default; use --no-save_epoch_recon to disable)"
    )

    parser.add_argument(
        "--select_metric",
        type=str,
        default="MCC",
        choices=["MCC", "F1_Score"],
        help="Validation metric used to select the threshold"
    )

    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.6,
        help="Fraction of full dataset allocated to the initial train window (default: 0.6)"
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction of full dataset allocated to validation split (default: 0.2)"
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Fraction of full dataset allocated to test split (default: 0.2)"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated random seeds, e.g. --seeds 42,7,123"
    )


    parser.add_argument(
        "--noise_type",
        type=str,
        default=DEFAULT_NOISE_TYPE,
        choices=sorted(ScheduledNoise.ALLOWED_NOISE_TYPES),
        help="Single active DAE noise type for main study (default: gaussian).",
    )

    parser.add_argument(
        "--ae_variant",
        type=str,
        default="all",
        help="AE variant(s) to run. Use 'all' or comma-separated values from: ff_dae, dvae, res_dae, sparse_dae. Default: all.",
    )

    parser.add_argument(
        "--vae_beta",
        type=float,
        default=0.001,
        help="KL-loss beta weight for the DVAE variant."
    )

    parser.add_argument(
        "--sparsity_l1",
        type=float,
        default=1e-5,
        help="L1 activity regularization strength for SPARSE_DAE latent activations."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted experiments by skipping completed ConfigID runs and rerunning incomplete ConfigID runs from deterministic initialization."
    )

    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = list(DEFAULT_SEEDS)
    else:
        try:
            args.seeds = parse_seed_list(args.seeds)
        except ValueError as exc:
            parser.error(str(exc))
    args.noise_type = str(args.noise_type).strip().lower()
    try:
        args.ae_variants = resolve_ae_variants(args.ae_variant)
    except ValueError as exc:
        parser.error(str(exc))
    if args.sparsity_l1 < 0:
        parser.error("--sparsity_l1 must be non-negative")
    if args.vae_beta < 0:
        parser.error("--vae_beta must be non-negative")
    validate_split_fractions(args.train_frac, args.val_frac, args.test_frac, parser=parser)
    return args


def resolve_ae_variants(ae_variant_arg):
    allowed = ["ff_dae", "dvae", "res_dae", "sparse_dae"]
    value = str(ae_variant_arg).strip().lower()

    if value == "all":
        return allowed

    tokens = [v.strip().lower() for v in value.split(",") if v.strip()]

    if not tokens:
        raise ValueError("No AE variants were provided.")

    invalid = [v for v in tokens if v not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported ae_variant value(s): {invalid}. "
            f"Allowed values are: all,{','.join(allowed)}"
        )

    deduped = []
    for v in tokens:
        if v not in deduped:
            deduped.append(v)

    return deduped


def parse_seed_list(seed_arg):
    tokens = [t.strip() for t in str(seed_arg).split(",") if t.strip() != ""]
    if not tokens:
        raise ValueError("Seeds list must not be empty. Use e.g. --seeds 42 or --seeds 42,7,123")
    try:
        seeds = [int(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid --seeds value '{seed_arg}'. Expected comma-separated integers.") from exc
    return seeds


def validate_split_fractions(train_frac, val_frac, test_frac, tol=1e-6, parser=None):
    fracs = [float(train_frac), float(val_frac), float(test_frac)]
    if not all(f > 0.0 for f in fracs):
        msg = f"Split fractions must be > 0. Got train={train_frac}, val={val_frac}, test={test_frac}"
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)
    total = sum(fracs)
    if abs(total - 1.0) > tol:
        msg = (
            f"Split fractions must sum to 1.0 within tolerance {tol}. "
            f"Got train+val+test={total:.8f} (train={train_frac}, val={val_frac}, test={test_frac})"
        )
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)


def resolve_datasets(dataset_args):
    """Resolve CLI dataset tokens into canonical dataset names."""
    ds = [d.upper() for d in dataset_args]
    if "ALL" in ds:
        return ALL_DATASETS
    resolved = []
    for d in ds:
        if d not in ALL_DATASETS:
            raise ValueError(f"Unknown dataset: {d}")
        resolved.append(d)
    return resolved


def resolve_schedules(schedule_args):
    """Resolve CLI schedule tokens into canonical schedule identifiers."""
    sch = [s.lower() for s in schedule_args]
    if "all" in sch:
        return ALL_SCHEDULES
    resolved = []
    for s in sch:
        if s not in ALL_SCHEDULES:
            raise ValueError(f"Unknown schedule: {s}")
        resolved.append(s)
    return resolved


# ============================================================
# 17) Summary table helper
# ============================================================
def _append_rows(csv_path, rows):
    """Append rows to a CSV while preserving a consistent union schema.

    Different metric row types do not always expose the same columns (for example,
    selected-threshold rows include SelectionMetric while epoch-history rows do
    not). Pandas' plain append mode writes the incoming column order without
    reconciling it with an existing header, which can silently corrupt mixed-row
    CSVs. This helper aligns rows to the existing header and rewrites the file
    only when new columns must be introduced.
    """
    if not rows:
        return

    df = pd.DataFrame(rows)
    _ensure_parent_dir(csv_path)

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        df.to_csv(csv_path, index=False)
        return

    try:
        existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except pd.errors.EmptyDataError:
        df.to_csv(csv_path, index=False)
        return

    extra_cols = [col for col in df.columns if col not in existing_cols]
    all_cols = existing_cols + extra_cols

    for col in existing_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df.reindex(columns=all_cols)

    if extra_cols:
        existing_df = pd.read_csv(csv_path)
        for col in extra_cols:
            existing_df[col] = np.nan
        existing_df = existing_df.reindex(columns=all_cols)
        pd.concat([existing_df, df], ignore_index=True).to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def normalize_ae_variant(ae_variant):
    key = str(ae_variant).strip().lower()
    return {
        "ff": "FF_DAE",
        "dae": "FF_DAE",
        "ff_dae": "FF_DAE",

        "dvae": "DVAE",

        "res": "RES_DAE",
        "res_dae": "RES_DAE",

        "sparse": "SPARSE_DAE",
        "sparse_dae": "SPARSE_DAE",
    }.get(key, str(ae_variant).strip().upper())


def _common_export_fields(run_id, dataset_name, model_name, range_name, seed, config_id, batch_size, learning_rate, enc_units, dec_units, latent, epochs, noise_type, ae_variant=""):
    return {"RunID": run_id, "Dataset": dataset_name, "Seed": seed, "AEVariant": ae_variant, "Model": model_name, "RangeName": range_name, "ConfigID": config_id, "BatchSize": batch_size, "LearningRate": learning_rate, "EncUnits": enc_units, "DecUnits": dec_units, "Latent": latent, "EpochsCfg": epochs, "NoiseType": noise_type}

def append_epoch_history_rows(
    train_val_csv,
    run_id,
    dataset_name,
    model_name,
    range_name,
    epoch_df,
    sigma_start=np.nan,
    sigma_end=np.nan,
    sigma_min=np.nan,
    decay_rate=np.nan,
    seed=np.nan,
    config_id="",
    batch_size=np.nan,
    learning_rate=np.nan,
    enc_units="",
    dec_units="",
    latent=np.nan,
    epochs=np.nan,
    noise_type="",
    ae_variant="",
    loss_interpretation="reconstruction_loss",
    config_init_seed=np.nan,
):
    if epoch_df is None or len(epoch_df) == 0:
        return

    rows = []
    for _, r in epoch_df.iterrows():
        rows.append({
            "RowType": "epoch_history",
            **_common_export_fields(run_id, dataset_name, model_name, range_name, seed, config_id, batch_size, learning_rate, enc_units, dec_units, latent, epochs, noise_type, ae_variant),
            "Epoch": r.get("Epoch", np.nan),
            "Train_Recon": r.get("Train_Recon", np.nan),
            "Val_Recon": r.get("Val_Recon", np.nan),
            "LossInterpretation": loss_interpretation,
            "Sigma": r.get("Sigma", np.nan),
            "Percentile": np.nan,
            "Threshold": np.nan,
            "Val_Accuracy": np.nan,
            "Val_Precision": np.nan,
            "Val_Recall": np.nan,
            "Val_F1_Score": np.nan,
            "Val_MCC": np.nan,
            "Val_ROC_AUC": np.nan,
            "Val_PR_AUC": np.nan,
            "Val_TN": np.nan,
            "Val_FP": np.nan,
            "Val_FN": np.nan,
            "Val_TP": np.nan,
            "SigmaStart": sigma_start,
            "SigmaEnd": sigma_end,
            "SigmaMin": sigma_min,
            "DecayRate": decay_rate,
            "ConfigInitSeed": config_init_seed,
        })
    _append_rows(train_val_csv, rows)


def append_validation_sweep_rows(
    train_val_csv,
    run_id,
    dataset_name,
    model_name,
    range_name,
    val_df,
    sigma_start=np.nan,
    sigma_end=np.nan,
    sigma_min=np.nan,
    decay_rate=np.nan,
    seed=np.nan,
    config_id="",
    batch_size=np.nan,
    learning_rate=np.nan,
    enc_units="",
    dec_units="",
    latent=np.nan,
    epochs=np.nan,
    noise_type="",
    ae_variant="",
    anomaly_score_type="recon",
    vae_beta=np.nan,
    config_init_seed=np.nan,
):
    rows = []
    for _, r in val_df.iterrows():
        rows.append({
            "RowType": "val_threshold",
            **_common_export_fields(run_id, dataset_name, model_name, range_name, seed, config_id, batch_size, learning_rate, enc_units, dec_units, latent, epochs, noise_type, ae_variant),
            "Epoch": np.nan,
            "Train_Recon": np.nan,
            "Val_Recon": np.nan,
            "AnomalyScoreType": anomaly_score_type,
            "VAEBeta": vae_beta,
            "Sigma": np.nan,
            "Percentile": r.get("Percentile", np.nan),
            "Threshold": r.get("Threshold", np.nan),
            "Val_Accuracy": r.get("Accuracy", np.nan),
            "Val_Precision": r.get("Precision", np.nan),
            "Val_Recall": r.get("Recall", np.nan),
            "Val_F1_Score": r.get("F1_Score", np.nan),
            "Val_MCC": r.get("MCC", np.nan),
            "Val_ROC_AUC": r.get("ROC_AUC", np.nan),
            "Val_PR_AUC": r.get("PR_AUC", np.nan),
            "Val_TN": r.get("TN", np.nan),
            "Val_FP": r.get("FP", np.nan),
            "Val_FN": r.get("FN", np.nan),
            "Val_TP": r.get("TP", np.nan),
            "SigmaStart": sigma_start,
            "SigmaEnd": sigma_end,
            "SigmaMin": sigma_min,
            "DecayRate": decay_rate,
            "ConfigInitSeed": config_init_seed,
        })
    _append_rows(train_val_csv, rows)


def append_selected_validation_row(
    train_val_csv,
    run_id,
    dataset_name,
    model_name,
    range_name,
    selected_row,
    sigma_start=np.nan,
    sigma_end=np.nan,
    sigma_min=np.nan,
    decay_rate=np.nan,
    seed=np.nan,
    config_id="",
    batch_size=np.nan,
    learning_rate=np.nan,
    enc_units="",
    dec_units="",
    latent=np.nan,
    epochs=np.nan,
    noise_type="",
    ae_variant="",
    selection_metric="MCC",
    anomaly_score_type="recon",
    vae_beta=np.nan,
    config_init_seed=np.nan,
):
    row = {
        "RowType": "val_selected_threshold",
        "RunID": run_id,
        "Dataset": dataset_name,
        "Seed": seed,
        "AEVariant": ae_variant,
        "Model": model_name,
        "RangeName": range_name,
        "ConfigID": config_id,
        "BatchSize": batch_size,
        "LearningRate": learning_rate,
        "EncUnits": enc_units,
        "DecUnits": dec_units,
        "Latent": latent,
        "EpochsCfg": epochs,
        "NoiseType": noise_type,
        "SelectionMetric": selection_metric,
        "AnomalyScoreType": anomaly_score_type,
        "VAEBeta": vae_beta,
        "Epoch": np.nan,
        "Train_Recon": np.nan,
        "Val_Recon": np.nan,
        "Sigma": np.nan,
        "Percentile": selected_row.get("Percentile", np.nan),
        "Threshold": selected_row.get("Threshold", np.nan),
        "Val_Accuracy": selected_row.get("Accuracy", np.nan),
        "Val_Precision": selected_row.get("Precision", np.nan),
        "Val_Recall": selected_row.get("Recall", np.nan),
        "Val_F1_Score": selected_row.get("F1_Score", np.nan),
        "Val_MCC": selected_row.get("MCC", np.nan),
        "Val_ROC_AUC": selected_row.get("ROC_AUC", np.nan),
        "Val_PR_AUC": selected_row.get("PR_AUC", np.nan),
        "Val_TN": selected_row.get("TN", np.nan),
        "Val_FP": selected_row.get("FP", np.nan),
        "Val_FN": selected_row.get("FN", np.nan),
        "Val_TP": selected_row.get("TP", np.nan),
        "SigmaStart": sigma_start,
        "SigmaEnd": sigma_end,
        "SigmaMin": sigma_min,
        "DecayRate": decay_rate,
        "ConfigInitSeed": config_init_seed,
    }
    _append_rows(train_val_csv, [row])
    return row


def append_final_test_row(test_csv, row):
    _append_rows(test_csv, [row])


def _has_valid_final_row(row, ae_variant_label):
    for col in ["MCC", "F1_Score", "Accuracy"]:
        val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if not np.isfinite(val):
            return False

    if ae_variant_label == "DVAE":
        if str(row.get("AnomalyScoreType", "")).strip().lower() != "elbo":
            return False
        beta = pd.to_numeric(pd.Series([row.get("VAEBeta")]), errors="coerce").iloc[0]
        if not np.isfinite(beta):
            return False

    if ae_variant_label == "SPARSE_DAE":
        s = pd.to_numeric(pd.Series([row.get("SparsityL1")]), errors="coerce").iloc[0]
        if not np.isfinite(s):
            return False

    return True


def is_config_completed(metrics_test_csv, selected_rows_csv_path, config_id, ae_variant=None):
    if not (os.path.exists(metrics_test_csv) and os.path.exists(selected_rows_csv_path)):
        return False
    try:
        test_df = pd.read_csv(metrics_test_csv)
        selected_df = pd.read_csv(selected_rows_csv_path)
    except Exception:
        return False
    if "ConfigID" not in test_df.columns or "ConfigID" not in selected_df.columns:
        return False

    test_rows = test_df[test_df["ConfigID"].astype(str) == str(config_id)]
    selected_rows = selected_df[selected_df["ConfigID"].astype(str) == str(config_id)]
    if test_rows.empty or selected_rows.empty:
        return False

    final_row = test_rows.iloc[-1].to_dict()
    variant = ae_variant if ae_variant is not None else final_row.get("AEVariant", "")
    variant_label = normalize_ae_variant(variant)
    return _has_valid_final_row(final_row, variant_label)


def cleanup_partial_config_artifacts(paths_to_check, config_id):
    csv_keys = ("train_val_metrics_csv", "metrics_test_csv", "selected_rows_csv_path")
    for key in csv_keys:
        path = paths_to_check.get(key)
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "ConfigID" in df.columns:
            df = df[df["ConfigID"].astype(str) != str(config_id)]
        df.to_csv(path, index=False)

    for key in ("epoch_plot_jpg", "threshold_plot_jpg", "error_plot_jpg"):
        path = paths_to_check.get(key)
        if path and os.path.exists(path):
            os.remove(path)


def map_model_to_family(model_name):
    model_name = str(model_name)
    family_by_suffix = {
        "CONST": "CONST",
        "LINEAR": "LINEAR",
        "EXP": "EXPONENTIAL",
        "COSINE": "COSINE",
        "FIB": "FIBONACCI",
        "SIGMOID": "SIGMOID",
        "CAUCHY": "CAUCHY",
        "LAPLACE": "LAPLACE",
        "LOGISTIC": "LOGISTIC",
    }
    for suffix, family in family_by_suffix.items():
        if model_name.endswith(f"_{suffix}"):
            return family
    return model_name


def make_pair_core_id(
    dataset_name,
    seed,
    ae_variant,
    noise_type,
    selection_metric,
    ae_cfg,
    anomaly_score_type="recon",
    vae_beta=np.nan,
):
    # PairCoreID identifies one shared fixed training context per dataset/seed/AE variant.
    beta_token = "NA" if pd.isna(vae_beta) else str(float(vae_beta))
    sparsity_l1 = ae_cfg.get("sparsity_l1", np.nan)
    sparsity_token = "NA" if pd.isna(sparsity_l1) else str(float(sparsity_l1))

    return (
        f"Dataset={dataset_name}|Seed={int(seed)}|AEVariant={str(ae_variant)}|"
        f"NoiseType={str(noise_type)}|SelectionMetric={str(selection_metric)}|"
        f"BatchSize={int(ae_cfg['batch_size'])}|LearningRate={float(ae_cfg['lr'])}|"
        f"EncUnits={str(tuple(ae_cfg['enc_units']))}|DecUnits={str(tuple(ae_cfg['dec_units']))}|"
        f"Latent={int(ae_cfg['latent_dim'])}|Epochs={int(ae_cfg['epochs'])}|"
        f"SparsityL1={sparsity_token}|AnomalyScoreType={str(anomaly_score_type)}|VAEBeta={beta_token}"
    )


def run_and_record_model(
    run_id,
    model,
    model_name,
    range_name,
    dataset_name,
    ae_cfg,
    X_train_norm,
    X_val,
    y_val_str,
    X_test,
    y_test_str,
    train_val_csv,
    metrics_test_csv,
    epoch_plot_jpg,
    threshold_plot_jpg,
    error_plot_jpg,
    generate_plots,
    save_epoch_recon=False,
    schedule_name=None,
    schedule_params=None,
    sigma_start=np.nan,
    sigma_end=np.nan,
    sigma_min=np.nan,
    decay_rate=np.nan,
    select_metric="MCC",
    seed=42,
    config_id="",
    noise_type="gaussian",
    ae_variant="",
    selected_rows_csv_path=None,
    config_init_seed=np.nan,
):
    """Train one configured model run, select a validation threshold, and export test metrics."""
    try:
        epoch_history_df = train_denoising_ae(
            model,
            X_train_norm,
            epochs=ae_cfg["epochs"],
            batch_size=ae_cfg["batch_size"],
            noise_schedule_name=schedule_name,
            noise_schedule_params=schedule_params,
            save_epoch_recon=save_epoch_recon,
        )

        if save_epoch_recon:
            append_epoch_history_rows(
                train_val_csv=train_val_csv,
                run_id=run_id,
                dataset_name=dataset_name,
                model_name=model_name,
                range_name=range_name,
                epoch_df=epoch_history_df,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                sigma_min=sigma_min,
                decay_rate=decay_rate,
                seed=seed,
                config_id=config_id,
                batch_size=ae_cfg["batch_size"],
                learning_rate=ae_cfg["lr"],
                enc_units=str(tuple(ae_cfg["enc_units"])),
                dec_units=str(tuple(ae_cfg["dec_units"])),
                latent=ae_cfg["latent_dim"],
                epochs=ae_cfg["epochs"],
                noise_type=noise_type,
                ae_variant=ae_variant,
                loss_interpretation=(
                    "dvae_total_loss_mse_plus_beta_kl"
                    if normalize_ae_variant(ae_variant) == "DVAE"
                    else "reconstruction_loss"
                ),
                config_init_seed=config_init_seed,
            )

        if generate_plots and save_epoch_recon and epoch_history_df is not None:
            plot_recon_vs_epochs_df(epoch_history_df, out_jpg_path=epoch_plot_jpg)

        normalized_variant = normalize_ae_variant(ae_variant)
        is_dvae = normalized_variant == "DVAE"
        anomaly_score_type = "elbo" if is_dvae else "recon"
        vae_beta = float(ae_cfg.get("vae_beta", 0.001)) if is_dvae else np.nan
        if is_dvae:
            print(f"[DVAE] Using ELBO anomaly score: recon + beta*KL | beta={vae_beta}")
        train_components = val_components = test_components = None

        if is_dvae:
            train_components = compute_dvae_score_components(
                model,
                X_train_norm,
                beta=vae_beta,
            )
            val_components = compute_dvae_score_components(
                model,
                X_val,
                beta=vae_beta,
            )
            test_components = compute_dvae_score_components(
                model,
                X_test,
                beta=vae_beta,
            )

            val_df, best_val_row = select_threshold_from_scores(
                train_scores=train_components["elbo"],
                val_scores=val_components["elbo"],
                y_val_str=y_val_str,
                select_metric=select_metric,
                percentile_grid=range(1, 101),
            )
        else:
            val_df, best_val_row = select_threshold_on_validation(
                model=model,
                X_train_norm=X_train_norm,
                X_val=X_val,
                y_val_str=y_val_str,
                select_metric=select_metric,
                percentile_grid=range(1, 101),
            )

        append_validation_sweep_rows(
            train_val_csv=train_val_csv,
            run_id=run_id,
            dataset_name=dataset_name,
            model_name=model_name,
            range_name=range_name,
            val_df=val_df,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            sigma_min=sigma_min,
            decay_rate=decay_rate,
            seed=seed,
            config_id=config_id,
            batch_size=ae_cfg["batch_size"],
            learning_rate=ae_cfg["lr"],
            enc_units=str(tuple(ae_cfg["enc_units"])),
            dec_units=str(tuple(ae_cfg["dec_units"])),
            latent=ae_cfg["latent_dim"],
            epochs=ae_cfg["epochs"],
            noise_type=noise_type,
            ae_variant=ae_variant,
            anomaly_score_type=anomaly_score_type,
            vae_beta=vae_beta,
            config_init_seed=config_init_seed,
        )
        selected_row_record = append_selected_validation_row(
            train_val_csv=train_val_csv,
            run_id=run_id,
            dataset_name=dataset_name,
            model_name=model_name,
            range_name=range_name,
            selected_row=best_val_row,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            sigma_min=sigma_min,
            decay_rate=decay_rate,
            seed=seed,
            config_id=config_id,
            batch_size=ae_cfg["batch_size"],
            learning_rate=ae_cfg["lr"],
            enc_units=str(tuple(ae_cfg["enc_units"])),
            dec_units=str(tuple(ae_cfg["dec_units"])),
            latent=ae_cfg["latent_dim"],
            epochs=ae_cfg["epochs"],
            noise_type=noise_type,
            ae_variant=ae_variant,
            selection_metric=select_metric,
            anomaly_score_type=anomaly_score_type,
            vae_beta=vae_beta,
            config_init_seed=config_init_seed,
        )
        if selected_rows_csv_path:
            _append_rows(selected_rows_csv_path, [selected_row_record])

        if generate_plots:
            plot_threshold_sweep(val_df, metric_name=select_metric, out_jpg_path=threshold_plot_jpg)

        if is_dvae:
            test_metrics, test_err, _ = evaluate_fixed_threshold_from_scores(
                test_scores=test_components["elbo"],
                y_test_str=y_test_str,
                threshold=best_val_row["Threshold"],
            )
        else:
            test_metrics, test_err, _ = evaluate_fixed_threshold_on_test(
                model=model,
                X_test=X_test,
                y_test_str=y_test_str,
                threshold=best_val_row["Threshold"],
            )

        def _component_mean(components, component_name):
            if components is None:
                return np.nan
            return float(np.mean(components[component_name]))

        enc_str = "_".join(map(str, ae_cfg["enc_units"]))
        dec_str = "_".join(map(str, ae_cfg["dec_units"]))

        row = {
            "RunID": run_id,
            "Dataset": dataset_name,
            "Seed": seed,
            "AEVariant": ae_variant,
            "ConfigID": config_id,
            "ConfigInitSeed": config_init_seed,
            "Model": model_name,
            "Family": map_model_to_family(model_name),
            "RangeName": range_name,
            "ArchName": f"{model_name}_{enc_str}_{ae_cfg['latent_dim']}_{dec_str}_{range_name}",
            "Epochs": ae_cfg["epochs"],
            "BatchSize": ae_cfg["batch_size"],
            "LearningRate": ae_cfg["lr"],
            "EncUnits": str(tuple(ae_cfg["enc_units"])),
            "DecUnits": str(tuple(ae_cfg["dec_units"])),
            "Latent": ae_cfg["latent_dim"],
            "Enc": str(ae_cfg["enc_units"]),
            "Dec": str(ae_cfg["dec_units"]),
            "NoiseType": noise_type,
            "SelectionMetric": select_metric,
            "AnomalyScoreType": anomaly_score_type,
            "VAEBeta": vae_beta,
            "RecurrentEvalMode": "not_applicable",
            "WindowSize": np.nan,
            "WindowStride": np.nan,
            "SparsityL1": ae_cfg.get("sparsity_l1", np.nan),
            "PairCoreID": make_pair_core_id(
                dataset_name,
                seed,
                ae_variant,
                noise_type,
                select_metric,
                ae_cfg,
                anomaly_score_type=anomaly_score_type,
                vae_beta=vae_beta,
            ),
            "SelectedPercentile": best_val_row["Percentile"],
            "SelectedThreshold": best_val_row["Threshold"],
            "SigmaStart": sigma_start,
            "SigmaEnd": sigma_end,
            "SigmaMin": sigma_min,
            "DecayRate": decay_rate,
            "Train_ReconScore_Mean": _component_mean(train_components, "recon"),
            "Train_KLScore_Mean": _component_mean(train_components, "kl"),
            "Train_ELBOScore_Mean": _component_mean(train_components, "elbo"),
            "Val_ReconScore_Mean": _component_mean(val_components, "recon"),
            "Val_KLScore_Mean": _component_mean(val_components, "kl"),
            "Val_ELBOScore_Mean": _component_mean(val_components, "elbo"),
            "Test_ReconScore_Mean": _component_mean(test_components, "recon"),
            "Test_KLScore_Mean": _component_mean(test_components, "kl"),
            "Test_ELBOScore_Mean": _component_mean(test_components, "elbo"),
            **test_metrics
        }
        append_final_test_row(metrics_test_csv, row)

        if generate_plots:
            score_label = "ELBO anomaly score" if is_dvae else "Reconstruction error"
            plot_error_distribution(
                f"{model_name}_{range_name}_{dataset_name}",
                test_err,
                y_test_str,
                out_jpg_path=error_plot_jpg,
                score_label=score_label,
            )

        return row, selected_row_record
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


# ============================================================
# 19) Dataset runner and exports
# ============================================================

def model_name_for_ae_variant(base_model_name, ae_variant):
    ae_variant = normalize_ae_variant(ae_variant)
    if ae_variant == "FF_DAE":
        return base_model_name
    suffix = str(base_model_name).replace("DAE_", "", 1)
    return f"{ae_variant}_{suffix}"


def config_id_for_ae_variant(ae_variant, model_name, range_name, noise_type, seed):
    ae_variant = normalize_ae_variant(ae_variant)
    family_token = str(model_name)
    for prefix in ("DVAE_", "RES_DAE_", "SPARSE_DAE_", "DAE_"):
        if family_token.startswith(prefix):
            family_token = family_token[len(prefix):]
            break
    return f"{ae_variant}_{family_token}_{range_name}_{str(noise_type).lower()}_seed{int(seed)}"


def build_schedule_definitions(selected_schedules):
    """Construct the fixed schedule-definition grid for the selected schedule families."""
    schedule_defs = []

    if "constant" in selected_schedules:
        for sigma_const in CONST_SIGMA_VALUES:
            sigma_label = f"{sigma_const:.2f}".replace(".", "p")
            schedule_defs.append({
                "model_name": "DAE_CONST",
                "range_name": f"SIG_{sigma_label}",
                "schedule_name": "constant",
                "schedule_params": {"sigma": sigma_const},
                "sigma_start": sigma_const,
                "sigma_end": sigma_const,
                "sigma_min": sigma_const,
                "decay_rate": np.nan,
            })

    for range_cfg in NOISE_RANGE_GRID:
        if "linear" in selected_schedules:
            schedule_defs.append({
                "model_name": "DAE_LINEAR",
                "range_name": range_cfg["name"],
                "schedule_name": "linear",
                "schedule_params": {"sigma_start": range_cfg["sigma_start"], "sigma_end": range_cfg["sigma_end"]},
                "sigma_start": range_cfg["sigma_start"],
                "sigma_end": range_cfg["sigma_end"],
                "sigma_min": np.nan,
                "decay_rate": np.nan,
            })

        if "exponential" in selected_schedules:
            schedule_defs.append({
                "model_name": "DAE_EXP",
                "range_name": range_cfg["name"],
                "schedule_name": "exponential",
                "schedule_params": {
                    "sigma_start": range_cfg["sigma_start"],
                    "sigma_min": range_cfg["sigma_min"],
                    "decay_rate": range_cfg["decay_rate"],
                },
                "sigma_start": range_cfg["sigma_start"],
                "sigma_end": np.nan,
                "sigma_min": range_cfg["sigma_min"],
                "decay_rate": range_cfg["decay_rate"],
            })

        if "cosine" in selected_schedules:
            schedule_defs.append({
                "model_name": "DAE_COSINE",
                "range_name": range_cfg["name"],
                "schedule_name": "cosine",
                "schedule_params": {"sigma_start": range_cfg["sigma_start"], "sigma_end": range_cfg["sigma_end"]},
                "sigma_start": range_cfg["sigma_start"],
                "sigma_end": range_cfg["sigma_end"],
                "sigma_min": np.nan,
                "decay_rate": np.nan,
            })

        if "fibonacci" in selected_schedules:
            schedule_defs.append({
                "model_name": "DAE_FIB",
                "range_name": range_cfg["name"],
                "schedule_name": "fibonacci",
                "schedule_params": {"sigma_start": range_cfg["sigma_start"], "sigma_end": range_cfg["sigma_end"]},
                "sigma_start": range_cfg["sigma_start"],
                "sigma_end": range_cfg["sigma_end"],
                "sigma_min": np.nan,
                "decay_rate": np.nan,
            })

        if "sigmoid" in selected_schedules:
            schedule_defs.append({
                "model_name": "DAE_SIGMOID",
                "range_name": range_cfg["name"],
                "schedule_name": "sigmoid",
                "schedule_params": {"sigma_start": range_cfg["sigma_start"], "sigma_end": range_cfg["sigma_end"], "k": 10.0, "t0": 0.5},
                "sigma_start": range_cfg["sigma_start"],
                "sigma_end": range_cfg["sigma_end"],
                "sigma_min": np.nan,
                "decay_rate": np.nan,
            })

        if "cauchy" in selected_schedules:
            schedule_defs.append({
                "model_name": "DAE_CAUCHY",
                "range_name": range_cfg["name"],
                "schedule_name": "cauchy",
                "schedule_params": {"sigma_start": range_cfg["sigma_start"], "sigma_end": range_cfg["sigma_end"], "gamma": 0.3},
                "sigma_start": range_cfg["sigma_start"],
                "sigma_end": range_cfg["sigma_end"],
                "sigma_min": np.nan,
                "decay_rate": np.nan,
            })

        if "laplace" in selected_schedules:
            schedule_defs.append({
                "model_name": "DAE_LAPLACE",
                "range_name": range_cfg["name"],
                "schedule_name": "laplace",
                "schedule_params": {"sigma_start": range_cfg["sigma_start"], "sigma_end": range_cfg["sigma_end"], "b": 0.3},
                "sigma_start": range_cfg["sigma_start"],
                "sigma_end": range_cfg["sigma_end"],
                "sigma_min": np.nan,
                "decay_rate": np.nan,
            })

        if "logistic" in selected_schedules:
            schedule_defs.append({
                "model_name": "DAE_LOGISTIC",
                "range_name": range_cfg["name"],
                "schedule_name": "logistic",
                "schedule_params": {"sigma_start": range_cfg["sigma_start"], "sigma_end": range_cfg["sigma_end"], "k": 6.0, "t0": 0.5},
                "sigma_start": range_cfg["sigma_start"],
                "sigma_end": range_cfg["sigma_end"],
                "sigma_min": np.nan,
                "decay_rate": np.nan,
            })

    return schedule_defs



def build_fixed_experiment_config_for_dataset(dataset_name, ae_cfg, cli_args, ae_variant=None):
    """Build a fixed DAE descriptor for a dataset run."""
    active_variant = ae_variant if ae_variant is not None else cli_args.ae_variant
    active_variant_label = normalize_ae_variant(active_variant)

    fixed_experiment_config = {
        "dataset": dataset_name,
        "vae_beta": float(cli_args.vae_beta),
        "sparsity_l1": float(cli_args.sparsity_l1),
        "sparsity_l1_note": "Used only by SPARSE_DAE; exported as NaN for non-sparse variants.",
        "architecture": {
            "enc_units": list(ae_cfg.get("enc_units", [])),
            "dec_units": list(ae_cfg.get("dec_units", [])),
            "latent_dim": ae_cfg.get("latent_dim"),
            "activation": ae_cfg.get("activation"),
            "output_activation": ae_cfg.get("output_activation"),
            "batch_norm": ae_cfg.get("batch_norm"),
            "dropout": ae_cfg.get("dropout"),
            "l2_reg": ae_cfg.get("l2_reg"),
        },
        "training": {
            "optimizer": ae_cfg.get("optimizer"),
            "learning_rate": ae_cfg.get("lr"),
            "loss": ae_cfg.get("loss"),
            "batch_size": ae_cfg.get("batch_size"),
            "epochs": ae_cfg.get("epochs"),
            "fit_time_validation_used": False,
            "threshold_validation_used": True,
            "threshold_validation_type": "mixed",
            "training_epochs_fixed": True,
        },
        "noise_type": str(cli_args.noise_type).lower(),
        "ae_variant": (
            "MULTI"
            if str(active_variant).strip().lower() == "all"
            else active_variant_label
        ),
    }
    fixed_experiment_config["epochs"] = ae_cfg.get("epochs")

    dae_configs = [{"HyperparamID": "HP_FIXED_DAE", "NoiseType": str(cli_args.noise_type).lower()}]
    return fixed_experiment_config, dae_configs


def _build_plot_paths(plots_dir, model_name, range_name, config_id, dataset_name, seed):
    return {
        "epoch": os.path.join(plots_dir, f"recon_vs_epochs_{model_name}_{range_name}_{config_id}_{dataset_name}_seed{seed}.jpg"),
        "threshold": os.path.join(plots_dir, f"threshold_sweep_{model_name}_{range_name}_{config_id}_{dataset_name}_seed{seed}.jpg"),
        "error": os.path.join(plots_dir, f"recon_error_{model_name}_{range_name}_{config_id}_{dataset_name}_seed{seed}.jpg"),
    }

def run_single_dataset(
    dataset_name,
    selected_schedules,
    generate_plots=True,
    save_epoch_recon=False,
    select_metric="MCC",
    seed=42,
    train_frac=0.6,
    val_frac=0.2,
    test_frac=0.2,
    reset_outputs=False,
    cli_args=None,
    global_run_counter=None,
    global_total_runs=None,
):
    """Run the full fixed experiment grid for one dataset/seed and write reviewer-facing artifacts."""
    run_id = make_run_id(dataset_name=dataset_name, seed=seed)
    cfg = copy.deepcopy(DATASET_CONFIGS[dataset_name])
    cfg["shuffle_seed"] = int(seed)
    ae_cfg = cfg["ae"]
    ae_cfg["vae_beta"] = float(cli_args.vae_beta)
    dae_cfg = cfg["dae"]
    ae_variant_key = str(cli_args.ae_variant).strip().lower()
    ae_variant_label = normalize_ae_variant(ae_variant_key)
    fixed_experiment_config, dae_configs = build_fixed_experiment_config_for_dataset(
        dataset_name,
        ae_cfg,
        cli_args,
        ae_variant=ae_variant_key,
    )

    print("\n" + "=" * 90)
    print(f"[Main] Processing dataset: {dataset_name} | seed={seed}")
    print("=" * 90)
    print(f"[Config] Fixed DAE config for dataset={dataset_name}: {dae_configs[0]}")
    print(f"[Config] Active noise type: {dae_configs[0]['NoiseType']}")

    output_dirs = ensure_output_directories()
    run_dirs = ensure_run_output_directory(output_dirs["runs"], dataset_name, seed, ae_variant=ae_variant_label)
    if reset_outputs:
        if os.path.exists(run_dirs["root"]):
            shutil.rmtree(run_dirs["root"])
        run_dirs = ensure_run_output_directory(output_dirs["runs"], dataset_name, seed, ae_variant=ae_variant_label)

    splits_dir = run_dirs["splits"]
    if split_indices_exist(splits_dir):
        existing_split_indices = load_split_indices(splits_dir)
        print(f"[Split] Existing split indices found. Reusing saved indices from: {splits_dir}")
    else:
        existing_split_indices = None
        print(f"[Split] No saved split indices found. Creating new split for {dataset_name} seed={seed}")

    data = load_dataset(
        dataset_name,
        cfg,
        run_info_dir=run_dirs["run_info"],
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        existing_split_indices=existing_split_indices,
    )

    X_train_norm = data["X_train_norm"]
    X_val = data["X_val"]
    y_val_str = data["y_val_str"]
    X_test = data["X_test"]
    y_test_str = data["y_test_str"]
    input_dim = data["input_dim"]
    X_train_model = X_train_norm
    X_val_model = X_val
    y_val_model = y_val_str
    X_test_model = X_test
    y_test_model = y_test_str

    print(f"[Main] Active AE variant: {ae_variant_label}")
    if ae_variant_label == "RES_DAE":
        print("[Main] RES_DAE uses sample-based residual dense autoencoder evaluation.")
    if ae_variant_label == "SPARSE_DAE":
        print(f"[Main] SPARSE_DAE uses sample-based sparse latent activity regularization | sparsity_l1={float(cli_args.sparsity_l1)}")

    print(f"[Main] Input dim for {dataset_name}: {input_dim}")
    print(f"[Main] Train normals: {X_train_norm.shape}")
    print(f"[Main] Validation  : {X_val.shape}")
    print(f"[Main] Test        : {X_test.shape}")

    dataset_info_aggregate_path = save_dataset_info_json(
        run_info_dir=run_dirs["run_info"],
        dataset_name=dataset_name,
        input_dim=input_dim,
        feature_columns=data.get("feature_cols", []),
        raw_feature_columns=data.get("raw_feature_cols", []),
        train_shape=X_train_norm.shape,
        val_shape=X_val.shape,
        test_shape=X_test.shape,
        counts=data.get("counts", {}),
        normal_label=cfg.get("normal_label"),
        shuffle_seed=seed,
        split_fractions={"train_frac": train_frac, "val_frac": val_frac, "test_frac": test_frac},
        run_id=run_id,
        split_indices={k: [int(x) for x in v] for k, v in data.get("split_indices", {}).items()},
        ae_variant=ae_variant_label,
    )

    run_info_json = os.path.join(
        run_dirs["run_info"],
        f"{dataset_name}_dataset_run_info_seed{seed}.json",
    )

    run_train_val_csv = os.path.join(run_dirs["metrics"], f"{dataset_name}_train_val_metrics_seed{seed}.csv")
    run_test_metrics_csv = os.path.join(run_dirs["metrics"], f"{dataset_name}_final_test_metrics_seed{seed}.csv")
    run_selected_threshold_csv = os.path.join(run_dirs["metrics"], f"{dataset_name}_selected_threshold_rows_seed{seed}.csv")

    if generate_plots:
        schedule_plot_epochs = int(fixed_experiment_config["epochs"])
        plot_all_schedule_curves(
            epochs=schedule_plot_epochs,
            out_jpg_path=os.path.join(run_dirs["plots"], f"schedule_curves_{dataset_name}_seed{seed}.jpg"),
            sigma_start=0.9,
            sigma_end=0.1,
            sigma_min=0.1,
        )

    def make_dae(local_ae_cfg, local_noise_type):
        if ae_variant_key == "ff_dae":
            model = build_denoising_ae(
                input_dim=input_dim,
                latent_dim=local_ae_cfg["latent_dim"],
                enc_units=local_ae_cfg["enc_units"],
                dec_units=local_ae_cfg["dec_units"],
                activation=local_ae_cfg["activation"],
                output_activation=local_ae_cfg["output_activation"],
                l2_reg=local_ae_cfg["l2_reg"],
                dropout=local_ae_cfg["dropout"],
                batch_norm=local_ae_cfg["batch_norm"],
                noise_std=dae_cfg["noise_std_init"],
                noise_type=local_noise_type,
                optimizer=local_ae_cfg["optimizer"],
                lr=local_ae_cfg["lr"],
                loss=local_ae_cfg["loss"],
                compile_model=True,
            )
            return model
        if ae_variant_key == "dvae":
            model = build_denoising_vae(
                input_dim=input_dim,
                latent_dim=local_ae_cfg["latent_dim"],
                enc_units=local_ae_cfg["enc_units"],
                dec_units=local_ae_cfg["dec_units"],
                activation=local_ae_cfg["activation"],
                output_activation=local_ae_cfg["output_activation"],
                noise_std=dae_cfg["noise_std_init"],
                noise_type=local_noise_type,
                beta=local_ae_cfg.get("vae_beta", cli_args.vae_beta),
                optimizer=local_ae_cfg["optimizer"],
                lr=local_ae_cfg["lr"],
            )
            return model
        if ae_variant_key == "res_dae":
            model = build_residual_denoising_ae(
                input_dim=input_dim,
                latent_dim=local_ae_cfg["latent_dim"],
                enc_units=local_ae_cfg["enc_units"],
                dec_units=local_ae_cfg["dec_units"],
                activation=local_ae_cfg["activation"],
                output_activation=local_ae_cfg["output_activation"],
                l2_reg=local_ae_cfg["l2_reg"],
                dropout=local_ae_cfg["dropout"],
                batch_norm=local_ae_cfg["batch_norm"],
                noise_std=dae_cfg["noise_std_init"],
                noise_type=local_noise_type,
                optimizer=local_ae_cfg["optimizer"],
                lr=local_ae_cfg["lr"],
                loss=local_ae_cfg["loss"],
                compile_model=True,
            )
            return model
        if ae_variant_key == "sparse_dae":
            model = build_sparse_denoising_ae(
                input_dim=input_dim,
                latent_dim=local_ae_cfg["latent_dim"],
                enc_units=local_ae_cfg["enc_units"],
                dec_units=local_ae_cfg["dec_units"],
                activation=local_ae_cfg["activation"],
                output_activation=local_ae_cfg["output_activation"],
                l2_reg=local_ae_cfg["l2_reg"],
                dropout=local_ae_cfg["dropout"],
                batch_norm=local_ae_cfg["batch_norm"],
                noise_std=dae_cfg["noise_std_init"],
                noise_type=local_noise_type,
                optimizer=local_ae_cfg["optimizer"],
                lr=local_ae_cfg["lr"],
                loss=local_ae_cfg["loss"],
                sparsity_l1=float(local_ae_cfg.get("sparsity_l1", cli_args.sparsity_l1)),
                compile_model=True,
            )
            return model
        raise ValueError(f"Unsupported --ae_variant '{ae_variant_key}'")
        
    schedule_defs = build_schedule_definitions(selected_schedules)
    total_runs_for_dataset_seed = len(schedule_defs)
    total_runs_all_seeds = total_runs_for_dataset_seed * len(cli_args.seeds)

    print(f"[RunPlan] Schedule runs for this dataset/seed/AEVariant: {total_runs_for_dataset_seed}")
    print(f"[RunPlan] Schedule runs across all seeds for this AEVariant: {total_runs_all_seeds}")

    run_info_path = save_dataset_run_metadata(
        run_info_json,
        run_id,
        dataset_name,
        cfg,
        data,
        selected_schedules,
        select_metric,
        schedule_defs,
        seed=seed,
        split_fractions={"train_frac": train_frac, "val_frac": val_frac, "test_frac": test_frac},
        fixed_experiment_config=fixed_experiment_config,
        ae_variant=ae_variant_label,
    )
    if existing_split_indices is None:
        run_split_csv_paths = save_split_csvs(
            dataset_name=dataset_name,
            run_info_dir=run_dirs["splits"],
            X_train=data["raw_splits"]["X_train"],
            y_train=data["raw_splits"]["y_train"],
            X_val=data["raw_splits"]["X_val"],
            y_val=data["raw_splits"]["y_val"],
            X_test=data["raw_splits"]["X_test"],
            y_test=data["raw_splits"]["y_test"],
            seed=seed,
            filename_prefix=None,
        )
        run_split_idx_paths = save_split_indices(data.get("split_indices", {}), splits_dir)
        print(f"[Split] Saved split indices to: {splits_dir}")
        print("[Split] Saved train_window_indices.csv, train_indices.csv, val_indices.csv, test_indices.csv")
    else:
        run_split_csv_paths = get_split_csv_paths(dataset_name, run_dirs["splits"], seed, filename_prefix=None)
        run_split_idx_paths = get_split_index_paths(splits_dir)
    scaler_path = save_scaler(data.get("scaler"), os.path.join(run_dirs["run_info"], "minmax_scaler.pkl"))

    local_run_counter = 0
    model_behavior_verified = False
    resume_skipped_completed = 0
    resume_rerun_incomplete_or_missing = 0
    resume_newly_completed = 0
    hp = dae_configs[0]
    for schedule_def in schedule_defs:
        local_ae_cfg = copy.deepcopy(ae_cfg)
        local_ae_cfg["vae_beta"] = float(cli_args.vae_beta) if ae_variant_key == "dvae" else np.nan
        local_ae_cfg["sparsity_l1"] = (
            float(cli_args.sparsity_l1)
            if ae_variant_key == "sparse_dae"
            else np.nan
        )
        variant_model_name = model_name_for_ae_variant(schedule_def["model_name"], ae_variant_key)
        config_id = config_id_for_ae_variant(ae_variant_label, variant_model_name, schedule_def["range_name"], hp["NoiseType"], seed)
        if global_run_counter is not None and global_total_runs is not None:
            global_run_counter[0] += 1
            print(
                f"[Run {global_run_counter[0]}/{global_total_runs}] "
                f"{variant_model_name} | range={schedule_def['range_name']} | config={config_id}"
            )
        else:
            local_run_counter += 1
            print(
                f"[Run {local_run_counter}/{total_runs_for_dataset_seed}] "
                f"{variant_model_name} | range={schedule_def['range_name']} | config={config_id}"
            )

        dae_plot_paths = _build_plot_paths(run_dirs["plots"], variant_model_name, schedule_def["range_name"], config_id, dataset_name, seed)
        if cli_args.resume:
            completed = is_config_completed(
                metrics_test_csv=run_test_metrics_csv,
                selected_rows_csv_path=run_selected_threshold_csv,
                config_id=config_id,
                ae_variant=ae_variant_label,
            )
            if completed:
                resume_skipped_completed += 1
                print(f"[Resume] Completed ConfigID found, skipping: {config_id}")
                continue

            resume_rerun_incomplete_or_missing += 1
            print(f"[Resume] ConfigID incomplete or missing, rerunning from deterministic seed: {config_id}")
            cleanup_partial_config_artifacts(
                paths_to_check={
                    "train_val_metrics_csv": run_train_val_csv,
                    "metrics_test_csv": run_test_metrics_csv,
                    "selected_rows_csv_path": run_selected_threshold_csv,
                    "epoch_plot_jpg": dae_plot_paths["epoch"],
                    "threshold_plot_jpg": dae_plot_paths["threshold"],
                    "error_plot_jpg": dae_plot_paths["error"],
                },
                config_id=config_id,
            )
        config_init_seed = make_deterministic_init_seed(
            dataset_name=dataset_name,
            seed=seed,
            ae_variant=ae_variant_label,
            base_seed=seed,
        )
        print(
            f"[Seed] Model initialization seed: {config_init_seed} | "
            f"dataset={dataset_name} | seed={seed} | ae_variant={ae_variant_label} | config={config_id}"
        )
        if not model_behavior_verified:
            print(
                f"[TrainingCheck] Running one-time disposable model behavior check for "
                f"dataset={dataset_name} | seed={seed} | ae_variant={ae_variant_label}"
            )
            set_config_random_seeds(config_init_seed)
            probe_model = make_dae(local_ae_cfg, hp["NoiseType"])
            try:
                assert_noise_layer_present(probe_model)
                verify_model_training_behavior(
                    probe_model,
                    X_train_model,
                    model_label=f"{ae_variant_label}_training_behavior_probe",
                )
            finally:
                del probe_model
                tf.keras.backend.clear_session()
                gc.collect()
            model_behavior_verified = True

        # Build the production model after the disposable probe so the check cannot
        # consume training RNG state or mutate layer state (for example BatchNorm
        # moving statistics) used by the measured experiment.
        set_config_random_seeds(config_init_seed)
        model_obj = make_dae(local_ae_cfg, hp["NoiseType"])
        assert_noise_layer_present(model_obj)

        run_and_record_model(
            run_id=run_id,
            model=model_obj,
            model_name=variant_model_name,
            range_name=schedule_def["range_name"],
            dataset_name=dataset_name,
            ae_cfg=local_ae_cfg,
            X_train_norm=X_train_model,
            X_val=X_val_model,
            y_val_str=y_val_model,
            X_test=X_test_model,
            y_test_str=y_test_model,
            train_val_csv=run_train_val_csv,
            metrics_test_csv=run_test_metrics_csv,
            epoch_plot_jpg=dae_plot_paths["epoch"],
            threshold_plot_jpg=dae_plot_paths["threshold"],
            error_plot_jpg=dae_plot_paths["error"],
            generate_plots=generate_plots,
            save_epoch_recon=save_epoch_recon,
            schedule_name=schedule_def["schedule_name"],
            schedule_params=schedule_def["schedule_params"],
            sigma_start=schedule_def["sigma_start"],
            sigma_end=schedule_def["sigma_end"],
            sigma_min=schedule_def["sigma_min"],
            decay_rate=schedule_def["decay_rate"],
            select_metric=select_metric,
            seed=seed,
            config_id=config_id,
            noise_type=hp["NoiseType"],
            ae_variant=ae_variant_label,
            selected_rows_csv_path=run_selected_threshold_csv,
            config_init_seed=config_init_seed,
        )
        if cli_args.resume:
            resume_newly_completed += 1

    manifest_payload = {
        "RunID": run_id,
        "dataset": dataset_name,
        "seed": int(seed),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_schedules": list(selected_schedules),
        "selection_metric": select_metric,
        "AEVariant": ae_variant_label,
        "ae_variant": ae_variant_label,
        "run_scope": "RunID identifies one dataset + one seed execution; ConfigID identifies one DAE experiment within that run.",
        "deterministic_init_seed_rule": "sha256(Dataset, Seed, AEVariant, BaseSeed) mapped to a 31-bit integer; same initialization is used for all schedules within the same Dataset, Seed, and AEVariant.",
        "split_fractions": {"train_frac": train_frac, "val_frac": val_frac, "test_frac": test_frac},
        "fixed_experiment_config": fixed_experiment_config,
        "expected_run_counts": {
            "total_runs_for_dataset_seed": total_runs_for_dataset_seed,
        },
        "artifacts": {
            "per_run": {
                "root": run_dirs["root"],
                "metrics_dir": run_dirs["metrics"],
                "plots_dir": run_dirs["plots"],
                "run_info_dir": run_dirs["run_info"],
                "shared_splits_dir": run_dirs["splits"],
                "manifest": os.path.join(run_dirs["root"], "manifest.json"),
                "dataset_info_json": dataset_info_aggregate_path,
                "dataset_run_info_json": run_info_path,
                "split_csvs": run_split_csv_paths,
                "split_indices": run_split_idx_paths,
                "scaler": scaler_path,
                "selected_threshold_rows_csv": run_selected_threshold_csv if os.path.exists(run_selected_threshold_csv) else None,
                "per_run_final_metrics_csv": run_test_metrics_csv if os.path.exists(run_test_metrics_csv) else None,
                "per_run_plots": [],
            },
            "run_metric_files": {
                "train_val_metrics_run": run_train_val_csv if os.path.exists(run_train_val_csv) else None,
                "final_test_metrics_run": run_test_metrics_csv if os.path.exists(run_test_metrics_csv) else None,
                "selected_threshold_rows": run_selected_threshold_csv if os.path.exists(run_selected_threshold_csv) else None,
            },
        },
    }
    manifest_path = save_run_manifest(os.path.join(run_dirs["root"], "manifest.json"), manifest_payload)

    print(f"\n[Main] Completed dataset: {dataset_name}")
    print(f"[Main] RunID: {run_id}")
    print(f"[Main] Run manifest: {manifest_path}")
    print("[Main] Metrics saved in:")
    print(f"  {run_train_val_csv}")
    print(f"  {run_test_metrics_csv}")
    print(f"  {run_selected_threshold_csv}")
    if cli_args.resume:
        print(
            f"[ResumeSummary] dataset={dataset_name} seed={seed} ae_variant={ae_variant_label} | "
            f"skipped_completed={resume_skipped_completed} | "
            f"rerun_incomplete_or_missing={resume_rerun_incomplete_or_missing} | "
            f"newly_completed={resume_newly_completed}"
        )


def assemble_all_final_test_metrics(
    output_root=OUTPUT_ROOT_DIR,
    output_filename="all_final_test_metrics.csv",
):
    """Rebuild the master final-test metrics CSV from per-run metric files."""
    runs_root = os.path.join(output_root, OUTPUT_SUBDIRS["runs"])
    source_pattern = "_final_test_metrics_seed"
    source_paths = []

    if os.path.isdir(runs_root):
        for dirpath, _, filenames in os.walk(runs_root):
            for filename in filenames:
                if filename.endswith(".csv") and source_pattern in filename:
                    source_paths.append(os.path.join(dirpath, filename))

    source_paths = sorted(source_paths)
    if not source_paths:
        raise FileNotFoundError(
            f"No final-test metric CSV files matching '*_final_test_metrics_seed*.csv' "
            f"were found under {runs_root}"
        )

    required_columns = [
        "Dataset",
        "Seed",
        "AEVariant",
        "ConfigID",
        "Model",
        "Family",
        "RangeName",
        "MCC",
        "F1_Score",
        "Accuracy",
    ]

    frames = []
    for source_path in source_paths:
        try:
            df = pd.read_csv(source_path)
        except Exception as exc:
            raise ValueError(f"Unable to read final-test metric CSV: {source_path}") from exc

        if df.empty:
            raise ValueError(f"Final-test metric CSV is empty: {source_path}")

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Final-test metric CSV is missing required columns {missing_columns}: "
                f"{source_path}"
            )

        df = df.copy()
        df["SourceFile"] = os.path.basename(source_path)
        df["SourcePath"] = os.path.relpath(source_path, output_root)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(ignore_index=True)

    duplicate_key = ["Dataset", "Seed", "AEVariant", "ConfigID"]
    duplicate_mask = combined.duplicated(subset=duplicate_key, keep=False)
    if duplicate_mask.any():
        sample_columns = duplicate_key + ["SourcePath"]
        duplicate_sample = (
            combined.loc[duplicate_mask, sample_columns]
            .sort_values(sample_columns, kind="mergesort")
            .head(20)
        )
        raise ValueError(
            "Conflicting final-test metric rows share the same "
            "Dataset × Seed × AEVariant × ConfigID key after exact duplicate removal. "
            "Sample:\n"
            f"{duplicate_sample.to_string(index=False)}"
        )

    first_columns = [
        "RunID",
        "Dataset",
        "Seed",
        "AEVariant",
        "ConfigID",
        "ConfigInitSeed",
        "Model",
        "Family",
        "RangeName",
        "PairCoreID",
        "ArchName",
        "Epochs",
        "BatchSize",
        "LearningRate",
        "EncUnits",
        "DecUnits",
        "Latent",
        "Enc",
        "Dec",
        "NoiseType",
        "SelectionMetric",
        "AnomalyScoreType",
        "VAEBeta",
        "SparsityL1",
        "SigmaStart",
        "SigmaEnd",
        "SigmaMin",
        "DecayRate",
        "SelectedPercentile",
        "SelectedThreshold",
        "Threshold",
        "ValidationThreshold",
        "ScoreMean",
        "ScoreStd",
        "ScoreMin",
        "ScoreMax",
        "TN",
        "FP",
        "FN",
        "TP",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "MCC",
        "ROC_AUC",
        "PR_AUC",
    ]
    provenance_columns = ["SourceFile", "SourcePath"]
    ordered_columns = [col for col in first_columns if col in combined.columns]
    additional_columns = sorted(
        col
        for col in combined.columns
        if col not in set(ordered_columns + provenance_columns)
    )
    combined = combined[ordered_columns + additional_columns + provenance_columns]

    sort_columns = [
        col
        for col in [
            "Dataset",
            "Seed",
            "AEVariant",
            "Family",
            "Model",
            "SigmaStart",
            "SigmaEnd",
            "RangeName",
            "ConfigID",
        ]
        if col in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="mergesort")
    combined = combined.reset_index(drop=True)

    final_path = os.path.join(output_root, output_filename)
    temp_path = final_path + ".tmp"
    os.makedirs(output_root, exist_ok=True)
    try:
        combined.to_csv(temp_path, index=False)
        os.replace(temp_path, final_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    unique_configurations = combined[duplicate_key].drop_duplicates().shape[0]
    print(f"[MasterMetrics] Source files: {len(source_paths)}")
    print(f"[MasterMetrics] Source files loaded: {len(frames)}")
    print(f"[MasterMetrics] Combined rows: {len(combined)}")
    print(f"[MasterMetrics] Unique configurations: {unique_configurations}")
    print("[MasterMetrics] Rows by Dataset:")
    print(combined.groupby("Dataset", dropna=False).size().to_string())
    print("[MasterMetrics] Rows by AEVariant:")
    print(combined.groupby("AEVariant", dropna=False).size().to_string())
    print("[MasterMetrics] Rows by Dataset × Seed × AEVariant:")
    print(combined.groupby(["Dataset", "Seed", "AEVariant"], dropna=False).size().to_string())
    print(f"[MasterMetrics] Saved: {final_path}")

    return final_path


# ============================================================
# 19) Main
# ============================================================
def main():
    args = parse_cli()

    print("[Device] CPU-only mode enabled.")

    selected_datasets = resolve_datasets(args.dataset)
    selected_schedules = resolve_schedules(args.schedules)
    if not selected_schedules:
        raise ValueError("No active schedules selected. Select at least one DAE schedule.")
    generate_plots = not args.no_plots

    print(f"[CLI] Datasets selected : {selected_datasets}")
    print(f"[CLI] Schedules selected: {selected_schedules}")
    print(f"[CLI] Generate plots    : {generate_plots}")
    print(f"[CLI] Save epoch recon  : {args.save_epoch_recon}")
    print(f"[CLI] Threshold metric  : {args.select_metric}")
    print(f"[CLI] Seeds            : {args.seeds}")
    print(f"[CLI] Active noise type : {args.noise_type}")
    print(f"[CLI] AE variant option : {args.ae_variant}")
    active_ae_variants = [normalize_ae_variant(v) for v in args.ae_variants]
    print(f"[Main] Active AE variants: {', '.join(active_ae_variants)}")
    print(
        f"[CLI] Split fractions  : train={args.train_frac:.4f}, "
        f"val={args.val_frac:.4f}, test={args.test_frac:.4f}"
    )
    print("[Main] Using fixed per-dataset DAE configuration for paired decay-vs-CONST runs.")

    planned_counts = {}
    for dataset_name in selected_datasets:
        ae_cfg_fixed = copy.deepcopy(DATASET_CONFIGS[dataset_name]["ae"])
        build_fixed_experiment_config_for_dataset(
            dataset_name,
            ae_cfg_fixed,
            args,
        )
        schedule_defs = build_schedule_definitions(selected_schedules)
        total_runs_for_dataset_seed = len(schedule_defs)
        planned_counts[dataset_name] = total_runs_for_dataset_seed

    if planned_counts:
        print("[Main] Planned run counts:")
        grand_total = 0
        for dataset_name in selected_datasets:
            per_dataset_seed = planned_counts.get(dataset_name, 0)
            dataset_total = per_dataset_seed * len(args.ae_variants) * len(args.seeds)
            grand_total += dataset_total
            print(
                f"  {dataset_name:<10} : per_dataset_seed={per_dataset_seed}, "
                f"ae_variants={len(args.ae_variants)}, seeds={len(args.seeds)}, total={dataset_total}"
            )
        print(f"[Main] Grand total planned runs: {grand_total}")
    else:
        grand_total = 0

    global_run_counter = [0]
    global_total_runs = grand_total

    for seed in args.seeds:
        set_random_seeds(seed)
        print(f"\n[Main] Starting seed={seed}")
        for dataset_name in selected_datasets:
            for ae_variant in args.ae_variants:
                ae_variant_label = normalize_ae_variant(ae_variant)
                reset_outputs = False
                local_args = copy.copy(args)
                local_args.ae_variant = ae_variant

                local_selected_schedules = list(selected_schedules)

                if not local_selected_schedules:
                    print(
                        f"[Main] No schedules left for {dataset_name} | seed={seed} | "
                        f"ae_variant={ae_variant_label}; skipping."
                    )
                    continue

                try:
                    run_single_dataset(
                        dataset_name=dataset_name,
                        selected_schedules=local_selected_schedules,
                        generate_plots=generate_plots,
                        save_epoch_recon=args.save_epoch_recon,
                        select_metric=args.select_metric,
                        seed=seed,
                        train_frac=args.train_frac,
                        val_frac=args.val_frac,
                        test_frac=args.test_frac,
                        reset_outputs=reset_outputs,
                        cli_args=local_args,
                        global_run_counter=global_run_counter,
                        global_total_runs=global_total_runs,
                    )
                finally:
                    tf.keras.backend.clear_session()
                    gc.collect()
                    print(
                        f"[Main] Freed memory after completing dataset: {dataset_name} | "
                        f"seed={seed} | ae_variant={ae_variant_label}"
                    )

    assemble_all_final_test_metrics(output_root=OUTPUT_ROOT_DIR)

if __name__ == "__main__":
    main()

