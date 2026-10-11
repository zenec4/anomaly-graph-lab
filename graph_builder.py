# ============================================================
# graph_builder.py - feature code for the window graphs (driver: build_day.py).
# Timestamps are parsed as ISO-8601 (see parse_timestamp_series).
# One window = one source file x one 5-minute tumbling window; all flows used; no label-derived features.
# ============================================================
import os
import re
import gc
import json
import math
import pickle
import random
import warnings
import ipaddress
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data


# ============================================================
# CONFIG
# ============================================================

WINDOW = "5min"            # window length (build_day.py --window overrides it)
MIN_NODES = 20             # windows with fewer nodes or edges are skipped
MIN_EDGES = 20
MAX_EDGES_PER_GRAPH = None # optional edge cap (not used)
GRAPH_HISTORY_LEN = 12     # previous windows used for the history part of the window attribute
RANDOM_SEED = 42
VERSION = "window-graphs-v1"


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

warnings.filterwarnings("ignore")


# ============================================================
# FEATURE SCHEMA
# ============================================================

NODE_COLS = [
    "out_flow_count",
    "in_flow_count",
    "out_bytes",
    "in_bytes",
    "out_packets",
    "in_packets",
    "out_degree",
    "in_degree",
    "total_degree",
    "unique_dst_ports",
    "unique_src_ports",
    "unique_protocols_out",
    "unique_protocols_in",
    "syn_out",
    "syn_in",
    "ack_out",
    "ack_in",
    "rst_out",
    "rst_in",
    "fin_out",
    "fin_in",
    "short_flow_out_share",
    "short_flow_in_share",
    "web_out_share",
    "web_in_share",
    "new_edge_out_share",
    "new_edge_in_share",
    "bytes_per_pkt_out",
    "bytes_per_pkt_in",
    "is_private_ip",
]

EDGE_COLS = [
    "cnt",
    "bytes_sum",
    "pkts_sum",
    "duration_sum",
    "duration_mean",
    "duration_std",
    "bytes_mean",
    "pkts_mean",
    "bpp_mean",
    "rate_bytes_mean",
    "rate_pkts_mean",
    "src_ports_nunique",
    "dst_ports_nunique",
    "proto_nunique",
    "dst_port_min",
    "dst_port_max",
    "syn_sum",
    "ack_sum",
    "rst_sum",
    "fin_sum",
    "psh_sum",
    "urg_sum",
    "syn_share",
    "rst_share",
    "fin_share",
    "short_flow_share",
    "web_share",
    "dns_share",
    "ssh_share",
    "ftp_share",
    "fwd_bytes_sum",
    "bwd_bytes_sum",
    "fwd_pkts_sum",
    "bwd_pkts_sum",
    "flow_iat_mean_mean",
    "flow_iat_std_mean",
    "pkt_len_mean_mean",
    "pkt_len_std_mean",
    "pkt_len_max_max",
    "pkt_len_min_min",
    "active_mean_mean",
    "idle_mean_mean",
    "new_edge_flag",
    "new_service_share",
    "dst_port_entropy",
    "proto_entropy",
]

# 47 base graph features * 4 temporal views = 188
GRAPH_BASE_COLS = [
    "n_flows",
    "n_nodes_full",
    "n_edges_full",
    "n_src_ips",
    "n_dst_ips",
    "n_src_ports",
    "n_dst_ports",
    "n_protocols",
    "bytes_total",
    "pkts_total",
    "duration_total",
    "duration_mean",
    "bytes_per_flow",
    "pkts_per_flow",
    "bytes_per_pkt",
    "edge_density",
    "avg_out_degree",
    "avg_in_degree",
    "max_out_degree",
    "max_in_degree",
    "top_src_share",
    "top_dst_share",
    "top_edge_share",
    "src_entropy",
    "dst_entropy",
    "edge_entropy",
    "dst_port_entropy",
    "protocol_entropy",
    "syn_total",
    "ack_total",
    "rst_total",
    "fin_total",
    "psh_total",
    "urg_total",
    "syn_share",
    "rst_share",
    "fin_share",
    "web_share",
    "dns_share",
    "ssh_share",
    "ftp_share",
    "short_flow_share",
    "new_edge_share",
    "new_service_share",
    "burst_10s_max",
    "burst_30s_max",
    "burst_60s_max",
]

assert len(NODE_COLS) == 30
assert len(EDGE_COLS) == 46
assert len(GRAPH_BASE_COLS) == 47


# ============================================================
# COLUMN ALIASES
# ============================================================

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


ALIASES = {
    "timestamp": [
        "timestamp",
        "time",
        "date",
    ],
    "src_ip": [
        "srcip",
        "srcipaddr",
        "sourceip",
        "sourceaddress",
        "sourceipaddress",
        "srcipaddress",
        "src ip",
        "source ip",
    ],
    "dst_ip": [
        "dstip",
        "dstipaddr",
        "destinationip",
        "destinationaddress",
        "destinationipaddress",
        "dstipaddress",
        "dst ip",
        "destination ip",
    ],
    "src_port": [
        "srcport",
        "sourceport",
        "src port",
        "source port",
    ],
    "dst_port": [
        "dstport",
        "destinationport",
        "dst port",
        "destination port",
    ],
    "protocol": [
        "protocol",
        "proto",
    ],
    "label": [
        "label",
        "attack",
        "class",
    ],
    "flow_duration": [
        "flowduration",
        "duration",
        "flow duration",
    ],
    "tot_fwd_pkts": [
        "totfwdpkts",
        "totalfwdpackets",
        "total fwd packets",
        "tot fwd pkts",
    ],
    "tot_bwd_pkts": [
        "totbwdpkts",
        "totalbwdpackets",
        "total bwd packets",
        "tot bwd pkts",
    ],
    "totlen_fwd_pkts": [
        "totlenfwdpkts",
        "total length of fwd packets",
        "totlen fwd pkts",
    ],
    "totlen_bwd_pkts": [
        "totlenbwdpkts",
        "total length of bwd packets",
        "totlen bwd pkts",
    ],
    "flow_iat_mean": [
        "flowiatmean",
        "flow iat mean",
    ],
    "flow_iat_std": [
        "flowiatstd",
        "flow iat std",
    ],
    "pkt_len_mean": [
        "pktlenmean",
        "packetlengthmean",
        "pkt len mean",
    ],
    "pkt_len_std": [
        "pktlenstd",
        "packetlengthstd",
        "pkt len std",
    ],
    "pkt_len_max": [
        "pktlenmax",
        "packetlengthmax",
        "pkt len max",
    ],
    "pkt_len_min": [
        "pktlenmin",
        "packetlengthmin",
        "pkt len min",
    ],
    "active_mean": [
        "activemean",
        "active mean",
    ],
    "idle_mean": [
        "idlemean",
        "idle mean",
    ],
    "syn": [
        "synflagcnt",
        "syn flag cnt",
        "synflagcount",
    ],
    "ack": [
        "ackflagcnt",
        "ack flag cnt",
        "ackflagcount",
    ],
    "rst": [
        "rstflagcnt",
        "rst flag cnt",
        "rstflagcount",
    ],
    "fin": [
        "finflagcnt",
        "fin flag cnt",
        "finflagcount",
    ],
    "psh": [
        "pshflagcnt",
        "psh flag cnt",
        "pshflagcount",
    ],
    "urg": [
        "urgflagcnt",
        "urg flag cnt",
        "urgflagcount",
    ],
}


def find_col(columns, key):
    norm_to_original = {_norm_name(c): c for c in columns}

    for alias in ALIASES[key]:
        n = _norm_name(alias)
        if n in norm_to_original:
            return norm_to_original[n]

    return None


# ============================================================
# UTILS
# ============================================================

def safe_num(s, default=0.0):
    if s is None:
        return default

    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(default)
    return out


def safe_log1p_arr(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    return np.log1p(x)


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()

    if total <= 0:
        return 0.0

    p = counts / total
    p = p[p > 0]

    if len(p) == 0:
        return 0.0

    return float(-(p * np.log2(p)).sum())


def entropy_series(s):
    if len(s) == 0:
        return 0.0

    vc = pd.Series(s).value_counts(dropna=False)
    return entropy_from_counts(vc.values)


def is_private_ip(ip):
    try:
        return float(ipaddress.ip_address(str(ip)).is_private)
    except Exception:
        return 0.0


def is_attempted_label(label):
    return "attempted" in str(label).lower()


def is_benign_label(label):
    return str(label).strip().upper() == "BENIGN"


def parse_timestamp_series(s):
    # Improved CIC-IDS2017 / CSE-CIC-IDS2018 use ISO timestamps (YYYY-MM-DD HH:MM:SS.ffffff).
    # An earlier version parsed them with dayfirst=True, which swapped day and month whenever both were <= 12
    # (e.g. 2018-03-02 -> 3 Feb); window contents were unaffected, but windows after midnight were mis-ordered.
    ts = pd.to_datetime(s, errors="coerce", format="ISO8601")
    bad = ts.isna()
    if bad.any():
        ts.loc[bad] = pd.to_datetime(s[bad], errors="coerce", dayfirst=False)
    return ts


def service_flags(dst_port):
    p = pd.to_numeric(dst_port, errors="coerce").fillna(-1).astype(int)

    is_web = p.isin([80, 443, 8080, 8443]).astype(float)
    is_dns = p.isin([53]).astype(float)
    is_ssh = p.isin([22]).astype(float)
    is_ftp = p.isin([20, 21]).astype(float)

    return is_web, is_dns, is_ssh, is_ftp


def clean_float_matrix(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -1e6, 1e6)
    return arr.astype(np.float32)


def clip_series(s, lo=-1e12, hi=1e12):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s.clip(lo, hi)


# ============================================================
# CHUNK CANONICALIZATION
# ============================================================

def canonicalize_chunk(chunk: pd.DataFrame, source_file: str) -> pd.DataFrame:
    cols = chunk.columns

    c_timestamp = find_col(cols, "timestamp")
    c_src_ip = find_col(cols, "src_ip")
    c_dst_ip = find_col(cols, "dst_ip")
    c_src_port = find_col(cols, "src_port")
    c_dst_port = find_col(cols, "dst_port")
    c_protocol = find_col(cols, "protocol")
    c_label = find_col(cols, "label")

    required = {
        "timestamp": c_timestamp,
        "src_ip": c_src_ip,
        "dst_ip": c_dst_ip,
        "src_port": c_src_port,
        "dst_port": c_dst_port,
        "protocol": c_protocol,
        "label": c_label,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns in {source_file}: {missing}. Columns={list(cols)[:50]}")

    def col_or_zero(key):
        c = find_col(cols, key)
        if c is None:
            return pd.Series(np.zeros(len(chunk)), index=chunk.index)
        return safe_num(chunk[c], 0.0)

    out = pd.DataFrame(index=chunk.index)

    out["timestamp"] = parse_timestamp_series(chunk[c_timestamp])
    out["src_ip"] = chunk[c_src_ip].astype(str).str.strip()
    out["dst_ip"] = chunk[c_dst_ip].astype(str).str.strip()
    out["src_port"] = safe_num(chunk[c_src_port], 0).astype(int)
    out["dst_port"] = safe_num(chunk[c_dst_port], 0).astype(int)
    out["protocol"] = safe_num(chunk[c_protocol], 0).astype(int)
    out["label"] = chunk[c_label].astype(str).str.strip()

    out["flow_duration"] = col_or_zero("flow_duration")
    out["fwd_pkts"] = col_or_zero("tot_fwd_pkts")
    out["bwd_pkts"] = col_or_zero("tot_bwd_pkts")
    out["fwd_bytes"] = col_or_zero("totlen_fwd_pkts")
    out["bwd_bytes"] = col_or_zero("totlen_bwd_pkts")
    out["flow_iat_mean"] = col_or_zero("flow_iat_mean")
    out["flow_iat_std"] = col_or_zero("flow_iat_std")
    out["pkt_len_mean"] = col_or_zero("pkt_len_mean")
    out["pkt_len_std"] = col_or_zero("pkt_len_std")
    out["pkt_len_max"] = col_or_zero("pkt_len_max")
    out["pkt_len_min"] = col_or_zero("pkt_len_min")
    out["active_mean"] = col_or_zero("active_mean")
    out["idle_mean"] = col_or_zero("idle_mean")

    out["syn"] = col_or_zero("syn")
    out["ack"] = col_or_zero("ack")
    out["rst"] = col_or_zero("rst")
    out["fin"] = col_or_zero("fin")
    out["psh"] = col_or_zero("psh")
    out["urg"] = col_or_zero("urg")

    out = out.dropna(subset=["timestamp"])
    out = out[(out["src_ip"] != "") & (out["dst_ip"] != "")]
    out = out[out["label"].notna()]

    # Basic derived features
    out["bytes_total"] = clip_series(out["fwd_bytes"] + out["bwd_bytes"])
    out["pkts_total"] = clip_series(out["fwd_pkts"] + out["bwd_pkts"])

    # Flow duration in CIC is usually microseconds.
    out["duration_sec"] = clip_series(out["flow_duration"] / 1_000_000.0)
    out["duration_sec"] = out["duration_sec"].clip(lower=0.0)

    out["bytes_per_pkt"] = out["bytes_total"] / out["pkts_total"].replace(0, np.nan)
    out["bytes_per_pkt"] = clip_series(out["bytes_per_pkt"])

    out["rate_bytes"] = out["bytes_total"] / out["duration_sec"].replace(0, np.nan)
    out["rate_pkts"] = out["pkts_total"] / out["duration_sec"].replace(0, np.nan)
    out["rate_bytes"] = clip_series(out["rate_bytes"])
    out["rate_pkts"] = clip_series(out["rate_pkts"])

    out["is_short_flow"] = (
        (out["duration_sec"] <= 1.0) |
        (out["pkts_total"] <= 3)
    ).astype(float)

    is_web, is_dns, is_ssh, is_ftp = service_flags(out["dst_port"])
    out["is_web"] = is_web
    out["is_dns"] = is_dns
    out["is_ssh"] = is_ssh
    out["is_ftp"] = is_ftp

    return out.reset_index(drop=True)


# ============================================================
# LABEL META
# ============================================================

def compute_label_meta(df: pd.DataFrame):
    labels = df["label"].astype(str)

    benign_mask = labels.map(is_benign_label)
    attempted_mask = labels.map(is_attempted_label)

    any_nonbenign_mask = ~benign_mask
    strict_attack_mask = any_nonbenign_mask & (~attempted_mask)

    raw_any_nonbenign_flows = int(any_nonbenign_mask.sum())
    raw_strict_attack_flows = int(strict_attack_mask.sum())
    raw_attempted_flows = int((any_nonbenign_mask & attempted_mask).sum())

    n = max(len(df), 1)

    raw_any_nonbenign_share = raw_any_nonbenign_flows / n
    raw_strict_attack_share = raw_strict_attack_flows / n
    raw_attempted_share = raw_attempted_flows / n

    has_attack_any = int(raw_any_nonbenign_flows > 0)
    has_attack_strict = int(raw_strict_attack_flows > 0)

    if raw_strict_attack_flows > 0:
        attack_type = labels[strict_attack_mask].value_counts().index[0]
    elif raw_attempted_flows > 0:
        attack_type = labels[any_nonbenign_mask & attempted_mask].value_counts().index[0]
    else:
        attack_type = "BENIGN"

    return {
        "has_attack_any": has_attack_any,
        "has_attack": has_attack_strict,
        "has_attack_strict": has_attack_strict,
        "raw_any_nonbenign_flows": raw_any_nonbenign_flows,
        "raw_strict_attack_flows": raw_strict_attack_flows,
        "raw_attempted_flows": raw_attempted_flows,
        "raw_any_nonbenign_share": raw_any_nonbenign_share,
        "raw_strict_attack_share": raw_strict_attack_share,
        "raw_attempted_share": raw_attempted_share,
        "attack_type": str(attack_type),
    }


# ============================================================
# EDGE FEATURES
# ============================================================

def build_edge_features(df: pd.DataFrame, history_edges: set, history_services: set):
    df = df.copy()

    df["edge_key"] = list(zip(df["src_ip"], df["dst_ip"]))
    df["service_key"] = list(zip(df["src_ip"], df["dst_ip"], df["dst_port"], df["protocol"]))

    df["is_new_edge"] = [1.0 if k not in history_edges else 0.0 for k in df["edge_key"]]
    df["is_new_service"] = [1.0 if k not in history_services else 0.0 for k in df["service_key"]]

    grp = df.groupby(["src_ip", "dst_ip"], sort=False, observed=True)

    edge_df = grp.agg(
        cnt=("src_ip", "size"),
        bytes_sum=("bytes_total", "sum"),
        pkts_sum=("pkts_total", "sum"),
        duration_sum=("duration_sec", "sum"),
        duration_mean=("duration_sec", "mean"),
        duration_std=("duration_sec", "std"),
        bytes_mean=("bytes_total", "mean"),
        pkts_mean=("pkts_total", "mean"),
        bpp_mean=("bytes_per_pkt", "mean"),
        rate_bytes_mean=("rate_bytes", "mean"),
        rate_pkts_mean=("rate_pkts", "mean"),
        src_ports_nunique=("src_port", "nunique"),
        dst_ports_nunique=("dst_port", "nunique"),
        proto_nunique=("protocol", "nunique"),
        dst_port_min=("dst_port", "min"),
        dst_port_max=("dst_port", "max"),
        syn_sum=("syn", "sum"),
        ack_sum=("ack", "sum"),
        rst_sum=("rst", "sum"),
        fin_sum=("fin", "sum"),
        psh_sum=("psh", "sum"),
        urg_sum=("urg", "sum"),
        short_flow_share=("is_short_flow", "mean"),
        web_share=("is_web", "mean"),
        dns_share=("is_dns", "mean"),
        ssh_share=("is_ssh", "mean"),
        ftp_share=("is_ftp", "mean"),
        fwd_bytes_sum=("fwd_bytes", "sum"),
        bwd_bytes_sum=("bwd_bytes", "sum"),
        fwd_pkts_sum=("fwd_pkts", "sum"),
        bwd_pkts_sum=("bwd_pkts", "sum"),
        flow_iat_mean_mean=("flow_iat_mean", "mean"),
        flow_iat_std_mean=("flow_iat_std", "mean"),
        pkt_len_mean_mean=("pkt_len_mean", "mean"),
        pkt_len_std_mean=("pkt_len_std", "mean"),
        pkt_len_max_max=("pkt_len_max", "max"),
        pkt_len_min_min=("pkt_len_min", "min"),
        active_mean_mean=("active_mean", "mean"),
        idle_mean_mean=("idle_mean", "mean"),
        new_edge_flag=("is_new_edge", "max"),
        new_service_share=("is_new_service", "mean"),
    ).reset_index()

    # Entropies per edge
    port_ent = grp["dst_port"].agg(entropy_series).rename("dst_port_entropy").reset_index()
    proto_ent = grp["protocol"].agg(entropy_series).rename("proto_entropy").reset_index()

    edge_df = edge_df.merge(port_ent, on=["src_ip", "dst_ip"], how="left")
    edge_df = edge_df.merge(proto_ent, on=["src_ip", "dst_ip"], how="left")

    edge_df["duration_std"] = edge_df["duration_std"].fillna(0.0)

    denom = edge_df["cnt"].replace(0, np.nan)
    edge_df["syn_share"] = edge_df["syn_sum"] / denom
    edge_df["rst_share"] = edge_df["rst_sum"] / denom
    edge_df["fin_share"] = edge_df["fin_sum"] / denom

    edge_df = edge_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Raw edge count before cap
    n_edges_full = len(edge_df)

    # Apply deterministic edge cap AFTER aggregation
    if MAX_EDGES_PER_GRAPH is not None and len(edge_df) > MAX_EDGES_PER_GRAPH:
        cap = int(MAX_EDGES_PER_GRAPH)

        n_activity = int(cap * 0.60)
        n_novelty = int(cap * 0.25)
        n_rare = cap - n_activity - n_novelty

        common_ports = {20, 21, 22, 25, 53, 80, 110, 123, 143, 389, 443, 445, 993, 995, 3306, 3389, 5432, 8080, 8443}

        tmp = edge_df.copy()
        tmp["activity_score"] = (
            np.log1p(tmp["cnt"])
            + np.log1p(tmp["bytes_sum"])
            + np.log1p(tmp["pkts_sum"])
        )

        tmp["rare_port_flag"] = (~tmp["dst_port_min"].astype(int).isin(common_ports)).astype(float)

        tmp["novelty_score"] = (
            3.0 * tmp["new_edge_flag"]
            + 2.0 * tmp["new_service_share"]
            + 1.0 * tmp["dst_port_entropy"]
            + 0.5 * tmp["proto_entropy"]
        )

        tmp["rare_score"] = (
            2.0 * tmp["rare_port_flag"]
            + 1.0 * tmp["short_flow_share"]
            + 1.0 / (1.0 + tmp["cnt"])
            + 1.0 * tmp["new_edge_flag"]
        )

        idx_activity = tmp.sort_values("activity_score", ascending=False).head(n_activity).index
        idx_novelty = tmp.sort_values("novelty_score", ascending=False).head(n_novelty).index
        idx_rare = tmp.sort_values("rare_score", ascending=False).head(n_rare).index

        keep_idx = pd.Index(idx_activity).union(idx_novelty).union(idx_rare)

        if len(keep_idx) < cap:
            rest = tmp.drop(index=keep_idx, errors="ignore")
            extra = rest.sort_values("activity_score", ascending=False).head(cap - len(keep_idx)).index
            keep_idx = keep_idx.union(extra)

        edge_df = edge_df.loc[keep_idx].reset_index(drop=True)

    # Log-transform heavy-tailed edge features
    log_cols = [
        "cnt",
        "bytes_sum",
        "pkts_sum",
        "duration_sum",
        "duration_mean",
        "duration_std",
        "bytes_mean",
        "pkts_mean",
        "bpp_mean",
        "rate_bytes_mean",
        "rate_pkts_mean",
        "src_ports_nunique",
        "dst_ports_nunique",
        "proto_nunique",
        "dst_port_min",
        "dst_port_max",
        "syn_sum",
        "ack_sum",
        "rst_sum",
        "fin_sum",
        "psh_sum",
        "urg_sum",
        "fwd_bytes_sum",
        "bwd_bytes_sum",
        "fwd_pkts_sum",
        "bwd_pkts_sum",
        "flow_iat_mean_mean",
        "flow_iat_std_mean",
        "pkt_len_mean_mean",
        "pkt_len_std_mean",
        "pkt_len_max_max",
        "pkt_len_min_min",
        "active_mean_mean",
        "idle_mean_mean",
    ]

    for c in log_cols:
        edge_df[c] = safe_log1p_arr(edge_df[c].values)

    edge_df = edge_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return edge_df, n_edges_full, df


# ============================================================
# NODE FEATURES
# ============================================================

def build_node_features(df: pd.DataFrame, edge_df: pd.DataFrame):
    nodes = sorted(set(edge_df["src_ip"].astype(str)).union(set(edge_df["dst_ip"].astype(str))))

    node_df = pd.DataFrame(index=nodes)
    node_df.index.name = "ip"

    # Outgoing
    out_grp = df.groupby("src_ip", sort=False, observed=True)

    out_basic = out_grp.agg(
        out_flow_count=("src_ip", "size"),
        out_bytes=("bytes_total", "sum"),
        out_packets=("pkts_total", "sum"),
        unique_dst_ports=("dst_port", "nunique"),
        unique_protocols_out=("protocol", "nunique"),
        syn_out=("syn", "sum"),
        ack_out=("ack", "sum"),
        rst_out=("rst", "sum"),
        fin_out=("fin", "sum"),
        short_flow_out_share=("is_short_flow", "mean"),
        web_out_share=("is_web", "mean"),
        new_edge_out_share=("is_new_edge", "mean"),
        bytes_per_pkt_out=("bytes_per_pkt", "mean"),
    )

    # Incoming
    in_grp = df.groupby("dst_ip", sort=False, observed=True)

    in_basic = in_grp.agg(
        in_flow_count=("dst_ip", "size"),
        in_bytes=("bytes_total", "sum"),
        in_packets=("pkts_total", "sum"),
        unique_src_ports=("src_port", "nunique"),
        unique_protocols_in=("protocol", "nunique"),
        syn_in=("syn", "sum"),
        ack_in=("ack", "sum"),
        rst_in=("rst", "sum"),
        fin_in=("fin", "sum"),
        short_flow_in_share=("is_short_flow", "mean"),
        web_in_share=("is_web", "mean"),
        new_edge_in_share=("is_new_edge", "mean"),
        bytes_per_pkt_in=("bytes_per_pkt", "mean"),
    )

    node_df = node_df.join(out_basic, how="left")
    node_df = node_df.join(in_basic, how="left")

    out_deg = edge_df.groupby("src_ip").size().rename("out_degree")
    in_deg = edge_df.groupby("dst_ip").size().rename("in_degree")

    node_df = node_df.join(out_deg, how="left")
    node_df = node_df.join(in_deg, how="left")

    node_df = node_df.fillna(0.0)

    node_df["total_degree"] = node_df["out_degree"] + node_df["in_degree"]
    node_df["is_private_ip"] = [is_private_ip(ip) for ip in node_df.index]

    # Log-transform heavy-tailed node features
    log_cols = [
        "out_flow_count",
        "in_flow_count",
        "out_bytes",
        "in_bytes",
        "out_packets",
        "in_packets",
        "out_degree",
        "in_degree",
        "total_degree",
        "unique_dst_ports",
        "unique_src_ports",
        "unique_protocols_out",
        "unique_protocols_in",
        "syn_out",
        "syn_in",
        "ack_out",
        "ack_in",
        "rst_out",
        "rst_in",
        "fin_out",
        "fin_in",
        "bytes_per_pkt_out",
        "bytes_per_pkt_in",
    ]

    for c in log_cols:
        node_df[c] = safe_log1p_arr(node_df[c].values)

    node_df = node_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    node_df = node_df[NODE_COLS]

    return node_df.reset_index()


# ============================================================
# GRAPH-LEVEL FEATURES
# ============================================================

def max_burst_count(df: pd.DataFrame, window_seconds: int):
    if len(df) == 0:
        return 0.0

    t = df["timestamp"].astype("int64") // 10**9
    t0 = int(t.min())

    bucket = ((t - t0) // window_seconds).astype(int)
    return float(bucket.value_counts().max())


def build_graph_base_features(df: pd.DataFrame, edge_df_full_for_stats: pd.DataFrame, history_edges: set, history_services: set):
    n_flows = len(df)

    src_counts = df["src_ip"].value_counts()
    dst_counts = df["dst_ip"].value_counts()
    edge_counts = df.groupby(["src_ip", "dst_ip"]).size()

    n_src_ips = int(df["src_ip"].nunique())
    n_dst_ips = int(df["dst_ip"].nunique())
    all_nodes = set(df["src_ip"].astype(str)).union(set(df["dst_ip"].astype(str)))

    n_nodes_full = len(all_nodes)
    n_edges_full = int(edge_counts.shape[0])

    bytes_total = float(df["bytes_total"].sum())
    pkts_total = float(df["pkts_total"].sum())
    duration_total = float(df["duration_sec"].sum())

    denom_flows = max(n_flows, 1)
    denom_pkts = max(pkts_total, 1.0)

    out_deg = edge_df_full_for_stats.groupby("src_ip").size()
    in_deg = edge_df_full_for_stats.groupby("dst_ip").size()

    if n_nodes_full > 1:
        edge_density = n_edges_full / (n_nodes_full * (n_nodes_full - 1))
    else:
        edge_density = 0.0

    new_edge_share = float(df["is_new_edge"].mean()) if "is_new_edge" in df.columns and len(df) else 0.0
    new_service_share = float(df["is_new_service"].mean()) if "is_new_service" in df.columns and len(df) else 0.0

    base = {
        "n_flows": n_flows,
        "n_nodes_full": n_nodes_full,
        "n_edges_full": n_edges_full,
        "n_src_ips": n_src_ips,
        "n_dst_ips": n_dst_ips,
        "n_src_ports": int(df["src_port"].nunique()),
        "n_dst_ports": int(df["dst_port"].nunique()),
        "n_protocols": int(df["protocol"].nunique()),

        "bytes_total": bytes_total,
        "pkts_total": pkts_total,
        "duration_total": duration_total,
        "duration_mean": float(df["duration_sec"].mean()),
        "bytes_per_flow": bytes_total / denom_flows,
        "pkts_per_flow": pkts_total / denom_flows,
        "bytes_per_pkt": bytes_total / denom_pkts,

        "edge_density": edge_density,
        "avg_out_degree": float(out_deg.mean()) if len(out_deg) else 0.0,
        "avg_in_degree": float(in_deg.mean()) if len(in_deg) else 0.0,
        "max_out_degree": float(out_deg.max()) if len(out_deg) else 0.0,
        "max_in_degree": float(in_deg.max()) if len(in_deg) else 0.0,

        "top_src_share": float(src_counts.max() / denom_flows) if len(src_counts) else 0.0,
        "top_dst_share": float(dst_counts.max() / denom_flows) if len(dst_counts) else 0.0,
        "top_edge_share": float(edge_counts.max() / denom_flows) if len(edge_counts) else 0.0,

        "src_entropy": entropy_from_counts(src_counts.values),
        "dst_entropy": entropy_from_counts(dst_counts.values),
        "edge_entropy": entropy_from_counts(edge_counts.values),
        "dst_port_entropy": entropy_series(df["dst_port"]),
        "protocol_entropy": entropy_series(df["protocol"]),

        "syn_total": float(df["syn"].sum()),
        "ack_total": float(df["ack"].sum()),
        "rst_total": float(df["rst"].sum()),
        "fin_total": float(df["fin"].sum()),
        "psh_total": float(df["psh"].sum()),
        "urg_total": float(df["urg"].sum()),

        "syn_share": float(df["syn"].sum() / denom_flows),
        "rst_share": float(df["rst"].sum() / denom_flows),
        "fin_share": float(df["fin"].sum() / denom_flows),

        "web_share": float(df["is_web"].mean()),
        "dns_share": float(df["is_dns"].mean()),
        "ssh_share": float(df["is_ssh"].mean()),
        "ftp_share": float(df["is_ftp"].mean()),
        "short_flow_share": float(df["is_short_flow"].mean()),

        "new_edge_share": new_edge_share,
        "new_service_share": new_service_share,

        "burst_10s_max": max_burst_count(df, 10),
        "burst_30s_max": max_burst_count(df, 30),
        "burst_60s_max": max_burst_count(df, 60),
    }

    out = pd.Series(base, dtype=float)[GRAPH_BASE_COLS]

    # Log-transform heavy-tailed graph features
    log_cols = [
        "n_flows",
        "n_nodes_full",
        "n_edges_full",
        "n_src_ips",
        "n_dst_ips",
        "n_src_ports",
        "n_dst_ports",
        "n_protocols",
        "bytes_total",
        "pkts_total",
        "duration_total",
        "duration_mean",
        "bytes_per_flow",
        "pkts_per_flow",
        "bytes_per_pkt",
        "avg_out_degree",
        "avg_in_degree",
        "max_out_degree",
        "max_in_degree",
        "syn_total",
        "ack_total",
        "rst_total",
        "fin_total",
        "psh_total",
        "urg_total",
        "burst_10s_max",
        "burst_30s_max",
        "burst_60s_max",
    ]

    for c in log_cols:
        out[c] = math.log1p(max(float(out[c]), 0.0))

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out.astype(float)


def build_temporal_graph_attr(base_vec: np.ndarray, graph_history: deque):
    base_vec = np.asarray(base_vec, dtype=np.float64)

    if len(graph_history) == 0:
        delta = np.zeros_like(base_vec)
        z = np.zeros_like(base_vec)
        rel = np.zeros_like(base_vec)
    else:
        prev = np.asarray(graph_history[-1], dtype=np.float64)
        hist = np.stack(list(graph_history), axis=0)

        mean = hist.mean(axis=0)
        std = hist.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)

        delta = base_vec - prev
        z = (base_vec - mean) / std
        rel = (base_vec - mean) / (np.abs(mean) + 1e-6)

    graph_attr = np.concatenate([base_vec, delta, z, rel], axis=0)
    graph_attr = clean_float_matrix(graph_attr)

    assert graph_attr.shape[0] == len(GRAPH_BASE_COLS) * 4

    return graph_attr


# ============================================================
# GRAPH BUILDING
# ============================================================

def make_graph_from_window(
    df_raw: pd.DataFrame,
    source_file: str,
    win_start,
    graph_global_id: int,
    history_edges: set,
    history_services: set,
    graph_history: deque,
):
    n_flows_raw_before_sampling = int(len(df_raw))

    # no random flow sampling before graph construction
    df = df_raw.copy()
    n_flows_used_for_features = int(len(df))

    label_meta = compute_label_meta(df)

    edge_df, n_edges_full, df_with_novelty = build_edge_features(
        df=df,
        history_edges=history_edges,
        history_services=history_services,
    )

    if n_edges_full < MIN_EDGES:
        return None, None, None

    node_df = build_node_features(df_with_novelty, edge_df)

    if len(node_df) < MIN_NODES:
        return None, None, None

    # Make node index mapping
    node_ips = node_df["ip"].astype(str).tolist()
    ip_to_idx = {ip: i for i, ip in enumerate(node_ips)}

    # Keep only edges whose endpoints are in node list
    edge_df = edge_df[
        edge_df["src_ip"].astype(str).isin(ip_to_idx) &
        edge_df["dst_ip"].astype(str).isin(ip_to_idx)
    ].reset_index(drop=True)

    if len(edge_df) < MIN_EDGES:
        return None, None, None

    src_idx = edge_df["src_ip"].astype(str).map(ip_to_idx).astype(int).values
    dst_idx = edge_df["dst_ip"].astype(str).map(ip_to_idx).astype(int).values

    edge_index = torch.tensor(
        np.vstack([src_idx, dst_idx]),
        dtype=torch.long,
    )

    x_np = clean_float_matrix(node_df[NODE_COLS].values)
    e_np = clean_float_matrix(edge_df[EDGE_COLS].values)

    graph_base = build_graph_base_features(
        df=df_with_novelty,
        edge_df_full_for_stats=edge_df,
        history_edges=history_edges,
        history_services=history_services,
    )

    graph_attr_np = build_temporal_graph_attr(
        base_vec=graph_base.values,
        graph_history=graph_history,
    )

    y = int(label_meta["has_attack"])

    data = Data(
        x=torch.tensor(x_np, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(e_np, dtype=torch.float32),
        graph_attr=torch.tensor(graph_attr_np, dtype=torch.float32).view(1, -1),
        y=torch.tensor([y], dtype=torch.long),
    )

    # Update history AFTER graph construction
    window_edges = set(zip(df_with_novelty["src_ip"].astype(str), df_with_novelty["dst_ip"].astype(str)))
    window_services = set(
        zip(
            df_with_novelty["src_ip"].astype(str),
            df_with_novelty["dst_ip"].astype(str),
            df_with_novelty["dst_port"].astype(int),
            df_with_novelty["protocol"].astype(int),
        )
    )

    history_edges.update(window_edges)
    history_services.update(window_services)
    graph_history.append(graph_base.values.astype(np.float64))

    meta = {
        "graph_id": int(graph_global_id),
        "source_file": source_file,
        "win_start": pd.Timestamp(win_start),
        "win_end": pd.Timestamp(win_start) + pd.Timedelta(WINDOW),

        "has_attack": int(label_meta["has_attack"]),
        "has_attack_any": int(label_meta["has_attack_any"]),
        "has_attack_strict": int(label_meta["has_attack_strict"]),
        "attack_type": str(label_meta["attack_type"]),

        "raw_any_nonbenign_flows": int(label_meta["raw_any_nonbenign_flows"]),
        "raw_strict_attack_flows": int(label_meta["raw_strict_attack_flows"]),
        "raw_attempted_flows": int(label_meta["raw_attempted_flows"]),
        "raw_any_nonbenign_share": float(label_meta["raw_any_nonbenign_share"]),
        "raw_strict_attack_share": float(label_meta["raw_strict_attack_share"]),
        "raw_attempted_share": float(label_meta["raw_attempted_share"]),

        "n_flows_raw": int(n_flows_raw_before_sampling),
        "n_flows_used_for_features": int(n_flows_used_for_features),
        "n_flows_sampled": int(n_flows_used_for_features),
        "flow_sampling_applied": False,

        "n_nodes": int(data.num_nodes),
        "n_edges": int(data.num_edges),
        "n_edges_full_before_cap": int(n_edges_full),
        "edge_cap_applied": bool(n_edges_full != data.num_edges),

        "node_dim": int(data.x.size(1)),
        "edge_dim": int(data.edge_attr.size(1)),
        "graph_attr_dim": int(data.graph_attr.size(1)),

        "n_src_ips": int(df_with_novelty["src_ip"].nunique()),
        "n_dst_ips": int(df_with_novelty["dst_ip"].nunique()),
        "n_src_ports": int(df_with_novelty["src_port"].nunique()),
        "n_dst_ports": int(df_with_novelty["dst_port"].nunique()),
        "n_protocols": int(df_with_novelty["protocol"].nunique()),

        "bytes_total_raw": float(df_with_novelty["bytes_total"].sum()),
        "pkts_total_raw": float(df_with_novelty["pkts_total"].sum()),
        "duration_total_raw": float(df_with_novelty["duration_sec"].sum()),

        "top_src_share_raw": float(df_with_novelty["src_ip"].value_counts().max() / max(len(df_with_novelty), 1)),
        "top_dst_share_raw": float(df_with_novelty["dst_ip"].value_counts().max() / max(len(df_with_novelty), 1)),
        "new_edge_share_raw": float(df_with_novelty["is_new_edge"].mean()),
        "new_service_share_raw": float(df_with_novelty["is_new_service"].mean()),

        "version": VERSION,
    }

    mapping = {
        "graph_id": int(graph_global_id),
        "source_file": source_file,
        "win_start": str(pd.Timestamp(win_start)),
        "node_idx_to_ip": {int(i): ip for i, ip in enumerate(node_ips)},
        "ip_to_node_idx": {ip: int(i) for i, ip in enumerate(node_ips)},
        "edge_idx_to_pair": {
            int(i): (
                str(edge_df.loc[i, "src_ip"]),
                str(edge_df.loc[i, "dst_ip"]),
            )
            for i in range(len(edge_df))
        },
        "node_cols": NODE_COLS,
        "edge_cols": EDGE_COLS,
        "graph_base_cols": GRAPH_BASE_COLS,
    }

    return data, meta, mapping


# ============================================================
# MAIN BUILD LOOP
# ============================================================

