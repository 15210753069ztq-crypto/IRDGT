import argparse
import json
import os
import pickle as pkl
import random
import sys
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.dataloader import (
    get_adjacent,
    get_grid_node_map_maxtrix,
    get_mask,
    normal_and_generate_dataset_time,
)
from lib.early_stop import EarlyStopping
from lib.utils import Scaler_Chi, Scaler_NYC, compute_loss, mask_loss, predict_and_evaluate
from model.HGCN import GSNet, GridOnlyNet, OriginalGSNet


DEFAULT_CONFIG = {
    "west_east_map": 20,
    "north_south_map": 20,
    "patience": 10,
    "delta": 1e-6,
    "seed": 2019,
    "data_type": "nyc",
    "all_data_filename": "data/nyc/all_data.pkl",
    "mask_filename": "data/nyc/risk_mask.pkl",
    "road_adj_filename": "data/nyc/road_adj.pkl",
    "risk_adj_filename": "data/nyc/risk_adj.pkl",
    "poi_adj_filename": "data/nyc/poi_adj.pkl",
    "grid_node_filename": "data/nyc/grid_node_map.pkl",
    "recent_prior": 3,
    "week_prior": 4,
    "one_day_period": 24,
    "days_of_week": 7,
    "pre_len": 1,
    "train_rate": 0.6,
    "valid_rate": 0.2,
    "training_epoch": 200,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "batch_size": 32,
    "learning_rate": 0.00001,
    "num_of_gru_layers": 5,
    "gru_hidden_size": 256,
    "gcn_num_filter": 64,
    "model_type": "dshgnn",
    "use_adaptive_feature_roles": False,
    "use_weather": True,
    "use_weather_subgraph": False,
    "use_weather_as_relation": False,
    "use_weather_rank_head": True,
    "weather_top_k": 12,
    "weather_history_len": 24,
    "use_poi_in_grid": False,
    "use_grid_branch": True,
    "use_graph_gru": True,
    "propagation_backbone": "gcn",
    "exclude_risk_in_functional_adj": False,
    "use_conditional_local_gate": True,
    "graph_feature_mode": "core",
    "use_prior_outputs": True,
    "role_formula": "legacy",
    "propagation_max_lag": 3,
    "propagation_max_edges": 800,
    "propagation_max_time": 512,
    "conditional_shift_tau": 0.25,
    "conditional_shift_scale": 8.0,
    "dvp_propagation_lambda": 1.0,
    "conditional_gate_mode": "legacy_suppress",
    "conditional_gate_strength": 1.0,
    "use_semantic_poi_adj": False,
    "monitor_metric": "val_rmse",
    "ranking_loss_weight": 0.0,
    "ranking_margin": 0.01,
    "ranking_max_negatives": 64,
    "high_frequency_loss_weight": 0.0,
    "high_frequency_hours": [6, 7, 8, 15, 16, 17, 18],
    "baseline_results": None,
    "print_baseline_comparison": False,
    "save_predictions": True,
    "prediction_output_dir": "outputs/predictions",
    "save_role_summaries": True,
    "role_summary_output_dir": "outputs/role_summaries",
    "test_mode": False,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to a JSON config file.")
    parser.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--test_mode", action="store_true", help="Use the first 100 samples.")
    parser.add_argument("--training_epoch", type=int, default=None)
    parser.add_argument("--recent_prior", type=int, default=None)
    parser.add_argument("--week_prior", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--model_type", choices=["dshgnn", "original_gsnet", "grid_only"], default=None)
    parser.add_argument("--propagation_backbone", choices=["gcn", "gt"], default=None)
    parser.add_argument("--exclude_risk_in_functional_adj", action="store_true")
    parser.add_argument("--disable_conditional_local_gate", action="store_true")
    parser.add_argument("--ranking_loss_weight", type=float, default=None)
    parser.add_argument("--high_frequency_loss_weight", type=float, default=None)
    return parser.parse_args()


def load_config(args):
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fp:
            config.update(json.load(fp))
        config["config_path"] = args.config
    if args.gpus is not None:
        config["gpus"] = args.gpus
    if args.test_mode:
        config["test_mode"] = True
    if args.exclude_risk_in_functional_adj:
        config["exclude_risk_in_functional_adj"] = True
    if args.disable_conditional_local_gate:
        config["use_conditional_local_gate"] = False
    for key in (
        "training_epoch",
        "recent_prior",
        "week_prior",
        "learning_rate",
        "model_type",
        "propagation_backbone",
        "ranking_loss_weight",
        "high_frequency_loss_weight",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def get_device(config):
    if config.get("gpus") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpus"])
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_graph_feature(x, high_x, grid_node_map, config):
    north_south_map = config["north_south_map"]
    west_east_map = config["west_east_map"]
    all_data_filename = config["all_data_filename"]
    if config.get("graph_feature_mode", "core") == "all":
        graph_indices = list(range(x.shape[2]))
    elif x.shape[2] >= 48 or "nyc" in all_data_filename:
        graph_indices = [0, 46, 47]
    elif x.shape[2] >= 41 or "chicago" in all_data_filename:
        graph_indices = [0, 39, 40]
    else:
        raise ValueError(f"Cannot infer graph feature indices from {all_data_filename}")

    num_grids = north_south_map * west_east_map
    graph_x = x[:, :, graph_indices, :, :].reshape((x.shape[0], x.shape[1], -1, num_grids))
    high_graph_x = high_x[:, :, graph_indices, :, :].reshape(
        (high_x.shape[0], high_x.shape[1], -1, num_grids)
    )
    return np.dot(graph_x, grid_node_map), np.dot(high_graph_x, grid_node_map)


def get_monitor_score(monitor_metric, val_loss, val_rmse, val_recall, val_map):
    metric = monitor_metric.lower()
    if metric == "val_rmse":
        return val_rmse
    if metric == "val_loss":
        return val_loss
    if metric == "val_recall":
        return -val_recall
    if metric == "val_map":
        return -val_map
    raise ValueError(f"Unsupported monitor_metric: {monitor_metric}")


def print_result_comparison(current_result, baseline_result):
    if not baseline_result:
        return
    baseline_name = baseline_result.get("name", "baseline")
    print("Result comparison against %s:" % baseline_name, flush=True)
    print("metric,current,%s,delta,better" % baseline_name, flush=True)
    metrics = (
        ("RMSE", "rmse", False),
        ("Recall", "recall", True),
        ("MAP", "map", True),
        ("RMSE*", "high_rmse", False),
        ("Recall*", "high_recall", True),
        ("MAP*", "high_map", True),
    )
    for display_name, key, higher_is_better in metrics:
        current_value = current_result.get(key)
        baseline_value = baseline_result.get(key)
        if current_value is None or baseline_value is None:
            continue
        delta = current_value - baseline_value
        better = delta > 0 if higher_is_better else delta < 0
        print(
            "%s,%.4f,%.4f,%+.4f,%s"
            % (display_name, current_value, baseline_value, delta, "yes" if better else "no"),
            flush=True,
        )


def build_result_snapshot(
    epoch,
    global_step,
    val_loss,
    val_rmse,
    val_recall,
    val_map,
    test_rmse,
    test_recall,
    test_map,
    high_test_rmse,
    high_test_recall,
    high_test_map,
    prediction=None,
    label=None,
):
    snapshot = {
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "val_recall": val_recall,
        "val_map": val_map,
        "rmse": test_rmse,
        "recall": test_recall,
        "map": test_map,
        "high_rmse": high_test_rmse,
        "high_recall": high_test_recall,
        "high_map": high_test_map,
    }
    if prediction is not None and label is not None:
        snapshot["prediction"] = prediction
        snapshot["label"] = label
    return snapshot


def print_result_snapshot(name, snapshot):
    if snapshot is None:
        return
    print(
        "%s epoch: %s, val RMSE: %.4f, val MAP: %.4f"
        % (name, snapshot["epoch"], snapshot["val_rmse"], snapshot["val_map"]),
        flush=True,
    )
    print(
        "%s test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f"
        % (name, snapshot["rmse"], snapshot["recall"], snapshot["map"]),
        flush=True,
    )
    print(
        "%s high test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f"
        % (name, snapshot["high_rmse"], snapshot["high_recall"], snapshot["high_map"]),
        flush=True,
    )


def infer_channel_names(config, num_channels):
    if num_channels >= 48 or "nyc" in config.get("all_data_filename", "").lower():
        names = ["risk"]
        names += ["hour_%02d" % idx for idx in range(24)]
        names += ["day_%d" % idx for idx in range(7)]
        names += ["holiday"]
        names += ["poi_%d" % idx for idx in range(7)]
        names += ["temperature", "clear", "cloudy", "rain", "snow", "mist", "inflow", "outflow"]
        return names[:num_channels]
    if num_channels >= 41 or "chicago" in config.get("all_data_filename", "").lower():
        names = ["risk"]
        names += ["hour_%02d" % idx for idx in range(24)]
        names += ["day_%d" % idx for idx in range(7)]
        names += ["holiday"]
        names += ["temperature", "clear", "cloudy", "rain", "snow", "mist", "inflow", "outflow"]
        return names[:num_channels]
    return ["channel_%d" % idx for idx in range(num_channels)]


def build_lagged_propagation_score(norm_train, grid_node_map, road_adj, config):
    max_lag = int(config.get("propagation_max_lag", 3))
    max_edges = int(config.get("propagation_max_edges", 800))
    max_time = int(config.get("propagation_max_time", 512))
    num_time, num_channels, north_south_map, west_east_map = norm_train.shape
    num_grids = north_south_map * west_east_map
    node_series = norm_train.reshape(num_time, num_channels, num_grids)
    node_series = np.dot(node_series, grid_node_map).astype(np.float32)
    if max_time > 0 and node_series.shape[0] > max_time:
        time_idx = np.linspace(0, node_series.shape[0] - 1, max_time).astype(np.int64)
        node_series = node_series[time_idx]
    edges = np.argwhere(road_adj > 0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if max_edges > 0 and edges.shape[0] > max_edges:
        edge_idx = np.linspace(0, edges.shape[0] - 1, max_edges).astype(np.int64)
        edges = edges[edge_idx]
    if edges.shape[0] == 0 or node_series.shape[0] <= 1:
        return np.zeros(num_channels, dtype=np.float32)

    src = edges[:, 0]
    dst = edges[:, 1]
    propagation_score = np.zeros(num_channels, dtype=np.float32)
    eps = 1e-6
    for channel_idx in range(num_channels):
        source_series = node_series[:, channel_idx, src]
        target_series = node_series[:, channel_idx, dst]
        sync_dist = np.sqrt(np.mean((source_series - target_series) ** 2, axis=0))
        best_aligned = sync_dist.copy()
        max_valid_lag = min(max_lag, node_series.shape[0] - 1)
        for lag in range(1, max_valid_lag + 1):
            forward_dist = np.sqrt(
                np.mean((source_series[:-lag] - target_series[lag:]) ** 2, axis=0)
            )
            backward_dist = np.sqrt(
                np.mean((source_series[lag:] - target_series[:-lag]) ** 2, axis=0)
            )
            best_aligned = np.minimum(best_aligned, forward_dist)
            best_aligned = np.minimum(best_aligned, backward_dist)
        gain = np.maximum(sync_dist - best_aligned, 0.0) / np.maximum(sync_dist, eps)
        propagation_score[channel_idx] = float(np.mean(gain))
    return propagation_score


def build_global_role_descriptor(config):
    all_data = pkl.load(open(config["all_data_filename"], "rb")).astype(np.float32)
    train_line = int(all_data.shape[0] * config["train_rate"])
    train_data = all_data[:train_line]
    if train_data.shape[1] == 48:
        scaler = Scaler_NYC(train_data)
    elif train_data.shape[1] == 41:
        scaler = Scaler_Chi(train_data)
    else:
        raise ValueError("Unsupported channel count for role descriptor: %s" % train_data.shape[1])
    norm_train = scaler.transform(train_data)
    mean_abs = np.abs(norm_train).mean(axis=(0, 2, 3))
    temporal_std = norm_train.std(axis=0).mean(axis=(1, 2))
    spatial_std = norm_train.std(axis=(2, 3)).mean(axis=0)
    if config.get("role_formula", "legacy") in ("dvp", "dvp_soft"):
        grid_node_map = get_grid_node_map_maxtrix(config["grid_node_filename"])
        road_adj = get_adjacent(config["road_adj_filename"])
        propagation_score = build_lagged_propagation_score(
            norm_train, grid_node_map, road_adj, config
        )
        return np.stack(
            [mean_abs, spatial_std, temporal_std, propagation_score], axis=-1
        ).astype(np.float32)
    if norm_train.shape[0] > 1:
        recent_delta = np.abs(np.diff(norm_train, axis=0)).mean(axis=(0, 2, 3))
    else:
        recent_delta = np.zeros_like(mean_abs)
    return np.stack(
        [mean_abs, temporal_std, spatial_std, recent_delta], axis=-1
    ).astype(np.float32)


def set_global_role_descriptor(net, descriptor, device):
    base_model = _unwrap_model(net)
    feature_role_module = getattr(base_model, "feature_role_module", None)
    if feature_role_module is None:
        return
    feature_role_module.set_global_descriptor(torch.from_numpy(descriptor).to(device))


def _unwrap_model(net):
    return net.module if hasattr(net, "module") else net


def collect_adaptive_role_summary(
    net,
    dataloader,
    road_adj,
    risk_adj,
    poi_adj,
    grid_node_map,
    device,
    config,
):
    base_model = _unwrap_model(net)
    feature_role_module = getattr(base_model, "feature_role_module", None)
    if feature_role_module is None or not getattr(base_model, "use_adaptive_feature_roles", False):
        return None

    role_values = []
    conditional_gate_values = []
    net.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5:
                feature, target_time, graph_feature, weather_history, _ = batch
                weather_history = weather_history.to(device)
            else:
                feature, target_time, graph_feature, _ = batch
                weather_history = None
            net(
                feature.to(device),
                target_time.to(device),
                graph_feature.to(device),
                road_adj,
                risk_adj,
                poi_adj,
                grid_node_map,
                weather_history=weather_history,
            )
            role_weight = feature_role_module.last_role_weight
            if role_weight is not None:
                role_values.append(role_weight.detach().cpu().numpy())
            conditional_gate = getattr(feature_role_module, "last_conditional_gate", None)
            if conditional_gate is not None:
                conditional_gate_values.append(conditional_gate.detach().cpu().numpy())

    if not role_values:
        return None

    role_values = np.concatenate(role_values, axis=0)
    if conditional_gate_values:
        conditional_gate_values = np.concatenate(conditional_gate_values, axis=0)
        conditional_gate_mean = conditional_gate_values.mean(axis=0)
        conditional_gate_std = conditional_gate_values.std(axis=0)
    else:
        conditional_gate_mean = np.zeros(role_values.shape[1], dtype=np.float32)
        conditional_gate_std = np.zeros(role_values.shape[1], dtype=np.float32)
    mean_role = role_values.mean(axis=0)
    std_role = role_values.std(axis=0)
    channel_names = infer_channel_names(config, mean_role.shape[0])
    channels = []
    for idx, name in enumerate(channel_names):
        propagation = float(mean_role[idx, 0])
        node_prior = float(mean_role[idx, 1])
        conditional_prior = float(mean_role[idx, 2]) if mean_role.shape[1] > 2 else 0.0
        role_scores = {
            "propagation": propagation,
            "node_prior": node_prior,
            "conditional_prior": conditional_prior,
        }
        dominant_role = max(role_scores, key=role_scores.get)
        channels.append(
            {
                "index": idx,
                "name": name,
                "propagation": propagation,
                "node_prior": node_prior,
                "conditional_prior": conditional_prior,
                "prior": node_prior,
                "propagation_std": float(std_role[idx, 0]),
                "node_prior_std": float(std_role[idx, 1]),
                "conditional_prior_std": float(std_role[idx, 2]) if std_role.shape[1] > 2 else 0.0,
                "conditional_gate": float(conditional_gate_mean[idx]),
                "conditional_gate_std": float(conditional_gate_std[idx]),
                "prior_std": float(std_role[idx, 1]),
                "dominant_role": dominant_role,
                "margin": sorted(role_scores.values(), reverse=True)[0]
                - sorted(role_scores.values(), reverse=True)[1],
            }
        )
    return {
        "num_samples": int(role_values.shape[0]),
        "channels": channels,
        "top_propagation": sorted(
            channels, key=lambda item: item["propagation"], reverse=True
        )[:10],
        "top_node_prior": sorted(
            channels, key=lambda item: item["node_prior"], reverse=True
        )[:10],
        "top_conditional_prior": sorted(
            channels, key=lambda item: item["conditional_prior"], reverse=True
        )[:10],
        "top_prior": sorted(channels, key=lambda item: item["node_prior"], reverse=True)[:10],
    }


def save_role_summary(summary, output_dir, config, selector, metrics):
    if summary is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(
        output_dir,
        "%s_%s_role_summary_%d.json"
        % (config.get("model_type", "model"), selector, int(time())),
    )
    payload = {
        "selector": selector,
        "metrics": {
            key: metrics.get(key)
            for key in (
                "epoch",
                "global_step",
                "val_loss",
                "val_rmse",
                "val_recall",
                "val_map",
                "rmse",
                "recall",
                "map",
                "high_rmse",
                "high_recall",
                "high_map",
            )
        },
        "role_summary": summary,
        "config": config,
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print("saved adaptive role summary:", path, flush=True)
    return path


def _metric_reaches_best(metric_name, value, threshold):
    if value is None or threshold is None:
        return False
    if metric_name in ("rmse", "high_rmse"):
        return value <= threshold
    return value >= threshold


def save_prediction_values(
    prediction,
    label,
    metrics,
    baseline_results,
    output_dir,
    config,
    selector="best",
):
    if prediction is None or label is None:
        return None
    if metrics.get("rmse") is None:
        return None

    current_metrics = {
        "rmse": metrics.get("rmse"),
        "recall": metrics.get("recall"),
        "map": metrics.get("map"),
        "high_rmse": metrics.get("high_rmse"),
        "high_recall": metrics.get("high_recall"),
        "high_map": metrics.get("high_map"),
    }
    save_reason = "rmse_below_baseline"
    best_thresholds = config.get("prediction_save_best_thresholds")
    if best_thresholds:
        reached = [
            key for key, threshold in best_thresholds.items()
            if _metric_reaches_best(key, current_metrics.get(key), threshold)
        ]
        if not reached:
            return None
        save_reason = "best_metric_reached:" + ",".join(reached)
    else:
        baseline_rmse = None
        if baseline_results:
            baseline_rmse = baseline_results.get("rmse")
        if baseline_rmse is not None and metrics["rmse"] >= baseline_rmse:
            return None

    os.makedirs(output_dir, exist_ok=True)
    tag = "%s_%s_rmse_%.4f_%d" % (
        config.get("model_type", "model"),
        selector,
        metrics["rmse"],
        int(time()),
    )
    npz_path = os.path.join(output_dir, tag + ".npz")
    meta_path = os.path.join(output_dir, tag + ".json")
    np.savez_compressed(
        npz_path,
        prediction=prediction,
        label=label,
        rmse=metrics.get("rmse"),
        recall=metrics.get("recall"),
        map=metrics.get("map"),
        high_rmse=metrics.get("high_rmse"),
        high_recall=metrics.get("high_recall"),
        high_map=metrics.get("high_map"),
    )
    metadata = {
        "prediction_file": npz_path,
        "selector": selector,
        "rmse": metrics.get("rmse"),
        "recall": metrics.get("recall"),
        "map": metrics.get("map"),
        "high_rmse": metrics.get("high_rmse"),
        "high_recall": metrics.get("high_recall"),
        "high_map": metrics.get("high_map"),
        "epoch": metrics.get("epoch"),
        "global_step": metrics.get("global_step"),
        "val_loss": metrics.get("val_loss"),
        "val_rmse": metrics.get("val_rmse"),
        "val_recall": metrics.get("val_recall"),
        "val_map": metrics.get("val_map"),
        "baseline_results": baseline_results,
        "save_reason": save_reason,
        "config": config,
    }
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)
    print("saved best prediction artifact:", npz_path, flush=True)
    return npz_path


def save_prediction_artifact(early_stop, baseline_results, output_dir, config):
    metrics = {
        "rmse": early_stop.best_rmse,
        "recall": early_stop.best_recall,
        "map": early_stop.best_map,
        "high_rmse": early_stop.best_high_rmse,
        "high_recall": early_stop.best_high_recall,
        "high_map": early_stop.best_high_map,
    }
    return save_prediction_values(
        early_stop.best_pre,
        early_stop.best_label,
        metrics,
        baseline_results,
        output_dir,
        config,
        selector="early_stop",
    )


def save_prediction_snapshot_artifact(snapshot, baseline_results, output_dir, config, selector):
    if snapshot is None:
        return None
    return save_prediction_values(
        snapshot.get("prediction"),
        snapshot.get("label"),
        snapshot,
        baseline_results,
        output_dir,
        config,
        selector=selector,
    )


def training(
    net,
    training_epoch,
    train_loader,
    val_loader,
    test_loader,
    high_test_loader,
    road_adj,
    risk_adj,
    poi_adj,
    risk_mask,
    trainer,
    early_stop,
    device,
    scaler,
    grid_node_map,
    data_type="nyc",
    monitor_metric="val_rmse",
    ranking_loss_weight=0.0,
    ranking_margin=0.01,
    ranking_max_negatives=64,
    high_frequency_loss_weight=0.0,
    high_frequency_hours=None,
    baseline_results=None,
    print_baseline_comparison=False,
    save_predictions=False,
    prediction_output_dir="outputs/predictions",
    save_role_summaries=False,
    role_summary_output_dir="outputs/role_summaries",
    config_snapshot=None,
):
    global_step = 1
    test_rmse = test_recall = test_map = None
    high_test_rmse = high_test_recall = high_test_map = None
    test_inverse_trans_pre = test_inverse_trans_label = None
    comparison_printed = False
    best_by_val_rmse = None
    best_by_val_map = None

    for epoch in range(1, training_epoch + 1):
        net.train()
        batch = 1
        for batch_data in train_loader:
            start_time = time()
            if len(batch_data) == 5:
                train_feature, target_time, graph_feature, weather_history, train_label = batch_data
                weather_history = weather_history.to(device)
            else:
                train_feature, target_time, graph_feature, train_label = batch_data
                weather_history = None
            train_feature = train_feature.to(device)
            target_time = target_time.to(device)
            graph_feature = graph_feature.to(device)
            train_label = train_label.to(device)
            prediction = net(
                train_feature,
                target_time,
                graph_feature,
                road_adj,
                risk_adj,
                poi_adj,
                grid_node_map,
                weather_history=weather_history,
            )
            loss = mask_loss(
                prediction,
                train_label,
                risk_mask,
                data_type=data_type,
                ranking_loss_weight=ranking_loss_weight,
                ranking_margin=ranking_margin,
                ranking_max_negatives=ranking_max_negatives,
                target_time_feature=target_time,
                high_frequency_loss_weight=high_frequency_loss_weight,
                high_frequency_hours=high_frequency_hours,
            )
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            print(
                "global step: %s, epoch: %s, batch: %s, training loss: %.6f, time: %.2fs"
                % (global_step, epoch, batch, loss.cpu().item(), time() - start_time),
                flush=True,
            )
            batch += 1
            global_step += 1

        val_loss = compute_loss(
            net,
            val_loader,
            risk_mask,
            road_adj,
            risk_adj,
            poi_adj,
            grid_node_map,
            global_step - 1,
            device,
            data_type,
            ranking_loss_weight=ranking_loss_weight,
            ranking_margin=ranking_margin,
            ranking_max_negatives=ranking_max_negatives,
            high_frequency_loss_weight=high_frequency_loss_weight,
            high_frequency_hours=high_frequency_hours,
        )
        val_rmse, val_recall, val_map, _, _ = predict_and_evaluate(
            net, val_loader, risk_mask, road_adj, risk_adj, poi_adj, grid_node_map,
            global_step - 1, scaler, device
        )
        monitor_score = get_monitor_score(monitor_metric, val_loss, val_rmse, val_recall, val_map)
        print(
            "global step: %s, epoch: %s, val loss: %.6f, val RMSE: %.4f, "
            "val Recall: %.2f%%, val MAP: %.4f"
            % (global_step - 1, epoch, val_loss, val_rmse, val_recall, val_map),
            flush=True,
        )

        rmse_improved = (
            best_by_val_rmse is None
            or val_rmse < best_by_val_rmse["val_rmse"] - early_stop.delta
        )
        map_improved = (
            best_by_val_map is None
            or val_map > best_by_val_map["val_map"] + early_stop.delta
        )
        monitor_improved = early_stop.is_improvement(monitor_score)

        if monitor_improved or rmse_improved or map_improved:
            test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = (
                predict_and_evaluate(
                    net, test_loader, risk_mask, road_adj, risk_adj, poi_adj, grid_node_map,
                    global_step - 1, scaler, device
                )
            )
            high_test_rmse, high_test_recall, high_test_map, _, _ = predict_and_evaluate(
                net, high_test_loader, risk_mask, road_adj, risk_adj, poi_adj, grid_node_map,
                global_step - 1, scaler, device
            )
            print(
                "global step: %s, epoch: %s, test RMSE: %.4f, test Recall: %.2f%%, "
                "test MAP: %.4f, high test RMSE: %.4f, high test Recall: %.2f%%, "
                "high test MAP: %.4f"
                % (
                    global_step - 1,
                    epoch,
                    test_rmse,
                    test_recall,
                    test_map,
                    high_test_rmse,
                    high_test_recall,
                    high_test_map,
                ),
                flush=True,
            )
            current_snapshot = build_result_snapshot(
                epoch,
                global_step - 1,
                val_loss,
                val_rmse,
                val_recall,
                val_map,
                test_rmse,
                test_recall,
                test_map,
                high_test_rmse,
                high_test_recall,
                high_test_map,
                prediction=test_inverse_trans_pre,
                label=test_inverse_trans_label,
            )
            if save_role_summaries and (rmse_improved or map_improved):
                current_snapshot["role_summary"] = collect_adaptive_role_summary(
                    net,
                    test_loader,
                    road_adj,
                    risk_adj,
                    poi_adj,
                    grid_node_map,
                    device,
                    config_snapshot or {},
                )
            if rmse_improved:
                best_by_val_rmse = current_snapshot
            if map_improved:
                best_by_val_map = current_snapshot

        early_stop(
            monitor_score,
            test_rmse,
            test_recall,
            test_map,
            high_test_rmse,
            high_test_recall,
            high_test_map,
            test_inverse_trans_pre,
            test_inverse_trans_label,
        )
        if early_stop.early_stop:
            print("Early Stopping in global step: %s, epoch: %s" % (global_step, epoch), flush=True)
            print(
                "best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f"
                % (early_stop.best_rmse, early_stop.best_recall, early_stop.best_map),
                flush=True,
            )
            print(
                "best test high RMSE: %.4f,best test high Recall: %.2f%%,best high test MAP: %.4f"
                % (
                    early_stop.best_high_rmse,
                    early_stop.best_high_recall,
                    early_stop.best_high_map,
                ),
                flush=True,
            )
            print_result_snapshot("best by val RMSE", best_by_val_rmse)
            print_result_snapshot("best by val MAP", best_by_val_map)
            if print_baseline_comparison:
                print_result_comparison(
                    {
                        "rmse": early_stop.best_rmse,
                        "recall": early_stop.best_recall,
                        "map": early_stop.best_map,
                        "high_rmse": early_stop.best_high_rmse,
                        "high_recall": early_stop.best_high_recall,
                        "high_map": early_stop.best_high_map,
                    },
                    baseline_results,
                )
                comparison_printed = True
            break
    if not early_stop.early_stop:
        print_result_snapshot("best by val RMSE", best_by_val_rmse)
        print_result_snapshot("best by val MAP", best_by_val_map)
    if print_baseline_comparison and not comparison_printed:
        print_result_comparison(
            {
                "rmse": early_stop.best_rmse,
                "recall": early_stop.best_recall,
                "map": early_stop.best_map,
                "high_rmse": early_stop.best_high_rmse,
                "high_recall": early_stop.best_high_recall,
                "high_map": early_stop.best_high_map,
            },
            baseline_results,
        )
    if save_predictions:
        save_prediction_artifact(
            early_stop,
            baseline_results,
            prediction_output_dir,
            config_snapshot or {},
        )
        save_prediction_snapshot_artifact(
            best_by_val_rmse,
            baseline_results,
            prediction_output_dir,
            config_snapshot or {},
            "best_val_rmse",
        )
        save_prediction_snapshot_artifact(
            best_by_val_map,
            baseline_results,
            prediction_output_dir,
            config_snapshot or {},
            "best_val_map",
        )
    if save_role_summaries:
        save_role_summary(
            best_by_val_rmse.get("role_summary") if best_by_val_rmse else None,
            role_summary_output_dir,
            config_snapshot or {},
            "best_val_rmse",
            best_by_val_rmse or {},
        )
        save_role_summary(
            best_by_val_map.get("role_summary") if best_by_val_map else None,
            role_summary_output_dir,
            config_snapshot or {},
            "best_val_map",
            best_by_val_map or {},
        )
    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map


def main(config):
    set_seed(config.get("seed"))
    device = get_device(config)
    print("device:", device, flush=True)
    print("config:", json.dumps(config, ensure_ascii=False, sort_keys=True), flush=True)

    north_south_map = config["north_south_map"]
    west_east_map = config["west_east_map"]
    train_rate = config["train_rate"]
    valid_rate = config["valid_rate"]
    recent_prior = config["recent_prior"]
    week_prior = config["week_prior"]
    one_day_period = config["one_day_period"]
    days_of_week = config["days_of_week"]
    pre_len = config["pre_len"]
    weather_history_len = config.get("weather_history_len", 24)
    seq_len = recent_prior + week_prior

    grid_node_map = get_grid_node_map_maxtrix(config["grid_node_filename"])
    global_role_descriptor = None
    if bool(config.get("use_adaptive_feature_roles", False)):
        global_role_descriptor = build_global_role_descriptor(config)
    loaders = []
    scaler = None
    train_data_shape = None
    graph_feature_shape = None
    time_shape = None
    high_test_loader = None

    for idx, (
        x,
        y,
        target_times,
        weather_history,
        high_x,
        high_y,
        high_target_times,
        high_weather_history,
        scaler,
    ) in enumerate(
        normal_and_generate_dataset_time(
            config["all_data_filename"],
            train_rate=train_rate,
            valid_rate=valid_rate,
            recent_prior=recent_prior,
            week_prior=week_prior,
            one_day_period=one_day_period,
            days_of_week=days_of_week,
            pre_len=pre_len,
            weather_history_len=weather_history_len,
        )
    ):
        if config.get("test_mode", False):
            x = x[:100]
            y = y[:100]
            target_times = target_times[:100]
            weather_history = weather_history[:100]
            high_x = high_x[:100]
            high_y = high_y[:100]
            high_target_times = high_target_times[:100]
            high_weather_history = high_weather_history[:100]

        graph_x, high_graph_x = build_graph_feature(x, high_x, grid_node_map, config)

        print(
            "feature:",
            str(x.shape),
            "label:",
            str(y.shape),
            "time:",
            str(target_times.shape),
            "weather history:",
            str(weather_history.shape),
            "high feature:",
            str(high_x.shape),
            "high label:",
            str(high_y.shape),
        )
        print("graph_x:", str(graph_x.shape), "high_graph_x:", str(high_graph_x.shape))
        if idx == 0:
            train_data_shape = x.shape
            time_shape = target_times.shape
            graph_feature_shape = graph_x.shape

        loaders.append(
            Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(x),
                    torch.from_numpy(target_times),
                    torch.from_numpy(graph_x),
                    torch.from_numpy(weather_history),
                    torch.from_numpy(y),
                ),
                batch_size=config["batch_size"],
                shuffle=(idx == 0),
            )
        )
        if idx == 2:
            high_test_loader = Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(high_x),
                    torch.from_numpy(high_target_times),
                    torch.from_numpy(high_graph_x),
                    torch.from_numpy(high_weather_history),
                    torch.from_numpy(high_y),
                ),
                batch_size=config["batch_size"],
                shuffle=False,
            )

    train_loader, val_loader, test_loader = loaders
    nums_of_filter = [config["gcn_num_filter"], config["gcn_num_filter"]]
    model_type = config.get("model_type", "dshgnn")
    if model_type == "original_gsnet":
        model_class = OriginalGSNet
    elif model_type == "grid_only":
        model_class = GridOnlyNet
    else:
        model_class = GSNet
    model = model_class(
        train_data_shape[2],
        config["num_of_gru_layers"],
        seq_len,
        pre_len,
        config["gru_hidden_size"],
        time_shape[1],
        graph_feature_shape[2],
        nums_of_filter,
        north_south_map,
        west_east_map,
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
        model = nn.DataParallel(model)
    model.to(device)
    if hasattr(model, "use_weather"):
        model.use_weather = bool(config.get("use_weather", True))
    if hasattr(model, "module") and hasattr(model.module, "use_weather"):
        model.module.use_weather = bool(config.get("use_weather", True))
    if hasattr(model, "use_adaptive_feature_roles"):
        model.use_adaptive_feature_roles = bool(config.get("use_adaptive_feature_roles", False))
    if hasattr(model, "module") and hasattr(model.module, "use_adaptive_feature_roles"):
        model.module.use_adaptive_feature_roles = bool(
            config.get("use_adaptive_feature_roles", False)
        )
    if hasattr(model, "use_weather_subgraph"):
        model.use_weather_subgraph = bool(config.get("use_weather_subgraph", False))
    if hasattr(model, "module") and hasattr(model.module, "use_weather_subgraph"):
        model.module.use_weather_subgraph = bool(config.get("use_weather_subgraph", False))
    if hasattr(model, "use_weather_as_relation"):
        model.use_weather_as_relation = bool(config.get("use_weather_as_relation", False))
    if hasattr(model, "module") and hasattr(model.module, "use_weather_as_relation"):
        model.module.use_weather_as_relation = bool(config.get("use_weather_as_relation", False))
    if hasattr(model, "use_weather_rank_head"):
        model.use_weather_rank_head = bool(config.get("use_weather_rank_head", True))
    if hasattr(model, "module") and hasattr(model.module, "use_weather_rank_head"):
        model.module.use_weather_rank_head = bool(config.get("use_weather_rank_head", True))
    weather_top_k = int(config.get("weather_top_k", 12))
    if hasattr(model, "weather_graph_module"):
        model.weather_graph_module.top_k = weather_top_k
    if hasattr(model, "module") and hasattr(model.module, "weather_graph_module"):
        model.module.weather_graph_module.top_k = weather_top_k
    if hasattr(model, "use_poi_in_grid"):
        model.use_poi_in_grid = bool(config.get("use_poi_in_grid", False))
    if hasattr(model, "module") and hasattr(model.module, "use_poi_in_grid"):
        model.module.use_poi_in_grid = bool(config.get("use_poi_in_grid", False))
    if hasattr(model, "use_grid_branch"):
        model.use_grid_branch = bool(config.get("use_grid_branch", True))
    if hasattr(model, "module") and hasattr(model.module, "use_grid_branch"):
        model.module.use_grid_branch = bool(config.get("use_grid_branch", True))
    if hasattr(model, "use_graph_gru"):
        model.use_graph_gru = bool(config.get("use_graph_gru", True))
    if hasattr(model, "module") and hasattr(model.module, "use_graph_gru"):
        model.module.use_graph_gru = bool(config.get("use_graph_gru", True))
    if hasattr(model, "propagation_backbone"):
        model.propagation_backbone = config.get("propagation_backbone", "gcn")
    if hasattr(model, "module") and hasattr(model.module, "propagation_backbone"):
        model.module.propagation_backbone = config.get("propagation_backbone", "gcn")
    if hasattr(model, "exclude_risk_in_functional_adj"):
        model.exclude_risk_in_functional_adj = bool(
            config.get("exclude_risk_in_functional_adj", False)
        )
    if hasattr(model, "module") and hasattr(model.module, "exclude_risk_in_functional_adj"):
        model.module.exclude_risk_in_functional_adj = bool(
            config.get("exclude_risk_in_functional_adj", False)
        )
    if hasattr(model, "use_conditional_local_gate"):
        model.use_conditional_local_gate = bool(
            config.get("use_conditional_local_gate", True)
        )
    if hasattr(model, "module") and hasattr(model.module, "use_conditional_local_gate"):
        model.module.use_conditional_local_gate = bool(
            config.get("use_conditional_local_gate", True)
        )
    if hasattr(model, "use_prior_outputs"):
        model.use_prior_outputs = bool(config.get("use_prior_outputs", True))
    if hasattr(model, "module") and hasattr(model.module, "use_prior_outputs"):
        model.module.use_prior_outputs = bool(config.get("use_prior_outputs", True))
    if hasattr(model, "role_formula"):
        model.role_formula = config.get("role_formula", "legacy")
    if hasattr(model, "module") and hasattr(model.module, "role_formula"):
        model.module.role_formula = config.get("role_formula", "legacy")
    if hasattr(model, "conditional_shift_tau"):
        model.conditional_shift_tau = float(config.get("conditional_shift_tau", 0.25))
    if hasattr(model, "module") and hasattr(model.module, "conditional_shift_tau"):
        model.module.conditional_shift_tau = float(config.get("conditional_shift_tau", 0.25))
    if hasattr(model, "conditional_shift_scale"):
        model.conditional_shift_scale = float(config.get("conditional_shift_scale", 8.0))
    if hasattr(model, "module") and hasattr(model.module, "conditional_shift_scale"):
        model.module.conditional_shift_scale = float(config.get("conditional_shift_scale", 8.0))
    if hasattr(model, "dvp_propagation_lambda"):
        model.dvp_propagation_lambda = float(config.get("dvp_propagation_lambda", 1.0))
    if hasattr(model, "module") and hasattr(model.module, "dvp_propagation_lambda"):
        model.module.dvp_propagation_lambda = float(
            config.get("dvp_propagation_lambda", 1.0)
        )
    if hasattr(model, "conditional_gate_mode"):
        model.conditional_gate_mode = config.get(
            "conditional_gate_mode", "legacy_suppress"
        )
    if hasattr(model, "module") and hasattr(model.module, "conditional_gate_mode"):
        model.module.conditional_gate_mode = config.get(
            "conditional_gate_mode", "legacy_suppress"
        )
    if hasattr(model, "conditional_gate_strength"):
        model.conditional_gate_strength = float(
            config.get("conditional_gate_strength", 1.0)
        )
    if hasattr(model, "module") and hasattr(model.module, "conditional_gate_strength"):
        model.module.conditional_gate_strength = float(
            config.get("conditional_gate_strength", 1.0)
        )
    if global_role_descriptor is not None:
        set_global_role_descriptor(model, global_role_descriptor, device)
        print(
            "global role descriptor:",
            str(global_role_descriptor.shape),
            "source:",
            "train_split_full_timeline",
            flush=True,
        )
    print(model)

    num_of_parameters = sum(np.prod(parameters.shape) for _, parameters in model.named_parameters())
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    if config.get("optimizer", "adam").lower() != "adam":
        raise ValueError("Only adam optimizer is currently supported.")
    trainer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    early_stop = EarlyStopping(
        patience=config["patience"],
        delta=config["delta"],
        metric_name=config.get("monitor_metric", "val_rmse"),
    )

    risk_mask = get_mask(config["mask_filename"])
    road_adj = get_adjacent(config["road_adj_filename"])
    risk_adj = get_adjacent(config["risk_adj_filename"])
    poi_adj = None
    if config.get("use_semantic_poi_adj", True) and config["poi_adj_filename"] != "":
        poi_adj = get_adjacent(config["poi_adj_filename"])

    return training(
        model,
        config["training_epoch"],
        train_loader,
        val_loader,
        test_loader,
        high_test_loader,
        road_adj,
        risk_adj,
        poi_adj,
        risk_mask,
        trainer,
        early_stop,
        device,
        scaler,
        grid_node_map,
        data_type=config["data_type"],
        monitor_metric=config.get("monitor_metric", "val_rmse"),
        ranking_loss_weight=config.get("ranking_loss_weight", 0.0),
        ranking_margin=config.get("ranking_margin", 0.01),
        ranking_max_negatives=config.get("ranking_max_negatives", 64),
        high_frequency_loss_weight=config.get("high_frequency_loss_weight", 0.0),
        high_frequency_hours=config.get("high_frequency_hours"),
        baseline_results=config.get("baseline_results"),
        print_baseline_comparison=config.get("print_baseline_comparison", False),
        save_predictions=config.get("save_predictions", True),
        prediction_output_dir=config.get("prediction_output_dir", "outputs/predictions"),
        save_role_summaries=config.get("save_role_summaries", True),
        role_summary_output_dir=config.get("role_summary_output_dir", "outputs/role_summaries"),
        config_snapshot=config,
    )


if __name__ == "__main__":
    print("train")
    main(load_config(parse_args()))
