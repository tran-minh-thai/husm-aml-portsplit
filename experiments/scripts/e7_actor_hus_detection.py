"""E7 - do the MINED high-utility subgraphs help detect illicit wallets?

The paper argues the value of subgraph mining sits at the structural/actor layer rather than
at the transaction-scoring layer. Plain actor-graph structure (degrees, counterparties)
already carries a large signal there, so the question this experiment has to answer is
sharper and less comfortable: does the mined high-utility pattern set add anything OVER that
cheap structure? If it does not, the mining is not earning its place in an actor-level
detector, and the paper should say so.

Design (leakage-free by construction):
- utility buckets are fit on the training window (Time step <= 34) and frozen (as everywhere);
- the pattern VOCABULARY is mined on the train-window subgraph only, then frozen: test-period
  edges never influence which patterns exist, exactly as they never influence the buckets;
- features are CAUSAL: a wallet observed at step s is described only by snapshots t <= s.
  Matching is done per snapshot, which is cheap (a snapshot is a few thousand vertices) and
  mirrors the per-snapshot segment decomposition the miner itself is built on;
- the test set drops wallets that also occur in the train window, so no wallet crosses.

Feature families compared:
  base    the 55 per-wallet monetary/activity features shipped with Elliptic++
  deg     causal degree structure (in/out, distinct counterparties, fan ratio, repetition)
  hus     causal participation in the frozen high-utility pattern vocabulary

The degree family is computed twice, on the hub-capped graph the miner runs on and on the
raw actor graph, because the hub cap exists only to make mining tractable and it is worth
knowing what it costs: a laundering hub is by definition high degree, so a cap on degree may
be discarding the very structure a detector would want.

Run:  python scripts/e7_actor_hus_detection.py [--hub-cap 50] [--min-util 100]
      python scripts/e7_actor_hus_detection.py --max-steps 8      # quick pipeline check
Output: results/e7_actor_hus_detection.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml import miner_bridge  # noqa: E402
from inchusm_aml.data.ellipticpp import (  # noqa: E402
    TRAIN_MAX_STEP,
    load_ellipticpp,
    prefilter_actor_graph,
)
from inchusm_aml.utility.monetary import (  # noqa: E402
    assign_bucket,
    fit_monetary_utility,
    monetary_signal,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = ROOT / "datasets" / "ellipticpp"
JAR = ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
SEEDS = [0, 1, 2, 3, 4]

DEG_COLS = ["out_deg", "in_deg", "uniq_out", "uniq_in", "deg_total", "fan_ratio",
            "rep_out", "rep_in"]
DEG = [f"m_{c}" for c in DEG_COLS]      # degree on the MINING graph (hub-capped)
DEGF = [f"f_{c}" for c in DEG_COLS]     # degree on the FULL graph (no prefilter)
HUS = ["hus_patterns", "hus_hits", "hus_max_util", "hus_sum_util", "hus_max_size", "hus_roles"]


def per_seed_f1(X_tr, y_tr, X_te, y_te):
    f1s, ps, rs = [], [], []
    for seed in SEEDS:
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
        rf.fit(X_tr, y_tr)
        p = rf.predict(X_te)
        f1s.append(f1_score(y_te, p, zero_division=0))
        ps.append(precision_score(y_te, p, zero_division=0))
        rs.append(recall_score(y_te, p, zero_division=0))
    return np.array(f1s), float(np.mean(ps)), float(np.mean(rs))


def causal_degree(edge_frame: pd.DataFrame, keys: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Causal degree structure per key row: a wallet at step s is described only by the
    edges of steps <= s. Returned columns are prefixed so the same construction can be run
    on the hub-capped mining graph and on the full graph and the two compared."""
    out = {}
    for name, addr_col, other in (("out", "src", "dst"), ("in", "dst", "src")):
        cnt = edge_frame.groupby([addr_col, "t"]).size().reset_index(name="c")
        cnt = cnt.rename(columns={addr_col: "address"}).sort_values(["address", "t"])
        cnt["cum"] = cnt.groupby("address")["c"].cumsum()
        m = pd.merge_asof(keys.sort_values("t"), cnt[["address", "t", "cum"]].sort_values("t"),
                          on="t", by="address", direction="backward").sort_values("_ord")
        out[f"{name}_deg"] = m["cum"].fillna(0.0).to_numpy()

        uq = edge_frame.drop_duplicates([addr_col, other]).groupby([addr_col, "t"]).size()
        uq = uq.reset_index(name="c").rename(columns={addr_col: "address"})
        uq = uq.sort_values(["address", "t"])
        uq["cum"] = uq.groupby("address")["c"].cumsum()
        m = pd.merge_asof(keys.sort_values("t"), uq[["address", "t", "cum"]].sort_values("t"),
                          on="t", by="address", direction="backward").sort_values("_ord")
        out[f"uniq_{name}"] = m["cum"].fillna(0.0).to_numpy()

    d = pd.DataFrame(out)
    d["deg_total"] = d.out_deg + d.in_deg
    d["fan_ratio"] = (d.out_deg - d.in_deg) / (d.deg_total + 1.0)
    d["rep_out"] = d.uniq_out / (d.out_deg + 1.0)
    d["rep_in"] = d.uniq_in / (d.in_deg + 1.0)
    return d.rename(columns={c: f"{prefix}_{c}" for c in d.columns})


def causal_cumulative(per_step: pd.DataFrame, keys: pd.DataFrame, cols_sum, cols_max):
    """Aggregate per-(address, step) values causally: for each key row (address, t), combine
    every step <= t. Sums accumulate, maxima take a running maximum."""
    p = per_step.sort_values(["address", "t"]).copy()
    for c in cols_sum:
        p[c] = p.groupby("address")[c].cumsum()
    for c in cols_max:
        p[c] = p.groupby("address")[c].cummax()
    out = pd.merge_asof(
        keys.sort_values("t"), p.sort_values("t"),
        on="t", by="address", direction="backward")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-buckets", type=int, default=4)
    ap.add_argument("--hub-cap", type=int, default=50)
    ap.add_argument("--min-util", type=float, default=100.0,
                    help="utility threshold for the frozen train vocabulary")
    ap.add_argument("--mer", type=int, default=1)
    ap.add_argument("--max-edges", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=49)
    args = ap.parse_args()

    if not JAR.exists():
        raise FileNotFoundError(f"missing {JAR} -- run: mvn -f inchusm-miner/pom.xml package")
    work = RESULTS / "e7_work"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- data, frozen monetary utility ------------------------------------
    print("loading Elliptic++ actor graph ...", flush=True)
    data = load_ellipticpp()
    train_mask = data.wallets["step"] <= TRAIN_MAX_STEP
    buckets = fit_monetary_utility(data.wallets, train_mask, n_buckets=args.n_buckets)
    signal = monetary_signal(data.wallets)
    labels = assign_bucket(signal, buckets.edges)
    node_label = pd.Series(labels.to_numpy(), index=data.wallets["address"])
    node_label = node_label[~node_label.index.duplicated(keep="first")]
    util_tsv = work / "utilities.tsv"
    miner_bridge.export_utilities(buckets.label_utility, util_tsv)

    step_of = data.wallets.groupby("address")["step"].min()

    def with_time(e: pd.DataFrame) -> pd.DataFrame:
        s = pd.concat([e["src"].map(step_of), e["dst"].map(step_of)], axis=1).max(axis=1)
        e = e[s.notna()].copy()
        e["t"] = s[s.notna()].astype(int)
        return e[e["t"] <= args.max_steps]

    # Two graphs on purpose. The miner cannot run on the raw actor graph, so a hub cap is
    # applied first; that cap is part of the mining pipeline, and comparing degree structure
    # on both graphs shows what the cap costs.
    full = with_time(data.edges)
    edges = with_time(prefilter_actor_graph(data.edges, set(node_label.index), args.hub_cap))
    print(f"  {len(node_label):,} labelled wallets | full graph {len(full):,} edges | "
          f"mining graph {len(edges):,} edges after hub-cap {args.hub_cap} "
          f"({100*(1-len(edges)/max(1,len(full))):.0f}% of edges dropped)", flush=True)

    # ---- frozen vocabulary: mine the TRAIN subgraph only -------------------
    tr_edges = edges[edges["t"] <= TRAIN_MAX_STEP]
    tr_nodes = pd.Index(pd.concat([tr_edges["src"], tr_edges["dst"]]).unique())
    tr_labels = node_label[node_label.index.isin(tr_nodes)]
    lg = work / "train.lg"
    miner_bridge.export_lg(tr_labels, tr_edges[["src", "dst"]], lg)
    print(f"  mining frozen vocabulary on train ({len(tr_labels):,} wallets, "
          f"{len(tr_edges):,} edges, minutil={args.min_util}) ...", flush=True)
    vocab, _ = miner_bridge.run_husm(lg, util_tsv, min_util=args.min_util,
                                     bound="ceu", max_edges=args.max_edges)
    if not len(vocab):
        print("!! no high-utility patterns on the train window; lower --min-util")
        return
    codes_path = work / "codes.txt"
    codes_path.write_text("\n".join(vocab["code"]) + "\n", encoding="ascii")
    util_of = dict(zip(vocab["code"], vocab["utility"]))
    size_of = dict(zip(vocab["code"], vocab["vertices"]))
    print(f"  frozen vocabulary: {len(vocab)} patterns", flush=True)

    # ---- match the frozen vocabulary snapshot by snapshot ------------------
    rows = []
    for t in sorted(edges["t"].unique()):
        e_t = edges[edges["t"] == t]
        nodes = pd.Index(pd.concat([e_t["src"], e_t["dst"]]).unique())
        lab_t = node_label[node_label.index.isin(nodes)]
        lg_t = work / f"snap_{t:02d}.lg"
        ids = miner_bridge.export_lg(lab_t, e_t[["src", "dst"]], lg_t)
        inv = {v: k for k, v in ids.items()}
        out_t = work / f"img_{t:02d}.csv"
        r = subprocess.run(
            ["java", "-jar", str(JAR), "match", "--graph", str(lg_t),
             "--util", str(util_tsv), "--patterns", str(codes_path),
             "--mer", str(args.mer), "--out", str(out_t)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"match failed at t={t}:\n{r.stderr}")
        img = pd.read_csv(out_t)
        if not len(img):
            continue
        img["address"] = img["vertex"].map(inv)
        img["util"] = img["pattern"].map(util_of).astype(float)
        img["size"] = img["pattern"].map(size_of).astype(float)
        g = img.groupby("address")
        rows.append(pd.DataFrame({
            "address": g.size().index,
            "t": t,
            "hus_patterns": g["pattern"].nunique().to_numpy(),
            "hus_hits": g.size().to_numpy(),
            "hus_sum_util": g["util"].sum().to_numpy(),
            "hus_max_util": g["util"].max().to_numpy(),
            "hus_max_size": g["size"].max().to_numpy(),
            "hus_roles": g["role"].nunique().to_numpy(),
        }))
        print(f"  t={t:2d}: {len(lab_t):6,} wallets, {len(e_t):7,} edges -> "
              f"{img['address'].nunique():6,} wallets in patterns", flush=True)
    per_step = pd.concat(rows, ignore_index=True)

    # ---- labels + causal features -----------------------------------------
    feats = pd.read_csv(DATA / "wallets_features.csv")
    classes = pd.read_csv(DATA / "wallets_classes.csv")
    df = feats.merge(classes, on="address", how="inner")
    df = df[df["class"].isin([1, 2])].copy()
    df["y"] = (df["class"] == 1).astype(np.int8)
    df = df.rename(columns={"Time step": "t"})
    df = df[df["t"] <= args.max_steps]

    keys = df[["address", "t"]].copy()
    keys["_ord"] = np.arange(len(keys))
    hus_feat = causal_cumulative(
        per_step, keys,
        cols_sum=["hus_patterns", "hus_hits", "hus_sum_util"],
        cols_max=["hus_max_util", "hus_max_size", "hus_roles"])
    hus_feat = hus_feat.sort_values("_ord")
    for c in HUS:
        df[c] = hus_feat[c].fillna(0.0).to_numpy()

    # causal degree structure on each graph: the one the miner sees, and the raw one
    for frame, prefix in ((edges[["src", "dst", "t"]], "m"), (full[["src", "dst", "t"]], "f")):
        d = causal_degree(frame, keys, prefix)
        for c in d.columns:
            df[c] = d[c].to_numpy()

    # ---- wallet-disjoint strict split -------------------------------------
    tr_m = df["t"] <= TRAIN_MAX_STEP
    both = set(df.loc[tr_m, "address"]) & set(df.loc[~tr_m, "address"])
    tr = tr_m
    te = (~tr_m) & (~df["address"].isin(both))
    print(f"\nsplit: train={int(tr.sum()):,} (illicit {int(df.loc[tr,'y'].sum()):,}) | "
          f"test={int(te.sum()):,} (illicit {int(df.loc[te,'y'].sum()):,})")
    nz = float((df.loc[te, HUS].abs().sum(axis=1) > 0).mean())
    print(f"test rows with any frozen-pattern participation: {100*nz:.1f}%\n", flush=True)
    if df.loc[te, "y"].sum() == 0:
        print("!! no illicit wallets in the test window")
        return

    drop = ["address", "class", "y", "t"]
    base = [c for c in df.columns
            if c not in drop and c not in DEG and c not in DEGF and c not in HUS
            and pd.api.types.is_numeric_dtype(df[c])]

    y_tr, y_te = df.loc[tr, "y"].to_numpy(), df.loc[te, "y"].to_numpy()
    rows_out = []

    def run(cols, tag):
        f1, p, r = per_seed_f1(df.loc[tr, cols].to_numpy(np.float32), y_tr,
                               df.loc[te, cols].to_numpy(np.float32), y_te)
        print(f"  {tag:34s} F1 = {f1.mean():.4f} +/- {f1.std():.4f}  (P={p:.3f} R={r:.3f})",
              flush=True)
        rows_out.append(dict(tag=tag, f1=f1.mean(), f1_std=f1.std(), precision=p, recall=r))
        return f1

    print("on the MINING graph (hub-capped), which is the only graph the miner can use:")
    f_base = run(base, "base")
    f_deg = run(base + DEG, "base + degree")
    run(base + HUS, "base + HUS")
    f_all = run(base + DEG + HUS, "base + degree + HUS")
    run(HUS, "HUS only")

    print("\non the FULL graph, to price the hub cap:")
    f_degf = run(base + DEGF, "base + degree (full graph)")
    run(DEGF, "degree only (full graph)")

    # The question the experiment exists to answer.
    t_stat, p_val = ttest_rel(f_all, f_deg)
    print(f"\n  (base+deg+HUS) - (base+deg) = {f_all.mean()-f_deg.mean():+.4f}"
          f"  <-- does mining add over cheap structure?")
    print(f"      paired t-test over {len(SEEDS)} seeds: t={t_stat:.3f} p={p_val:.4f}")
    rows_out.append(dict(tag="delta_hus_over_deg", f1=f_all.mean() - f_deg.mean(),
                         f1_std=np.nan, precision=t_stat, recall=p_val))

    # What the hub cap costs, measured on the same split with the same construction.
    tf, pf = ttest_rel(f_degf, f_base)
    print(f"\n  (base+deg FULL) - base      = {f_degf.mean()-f_base.mean():+.4f}"
          f"  (t={tf:.2f} p={pf:.4f})")
    print(f"  (base+deg MINING) - base    = {f_deg.mean()-f_base.mean():+.4f}"
          f"  <-- same features, graph after the hub cap")
    rows_out.append(dict(tag="delta_degfull_over_base", f1=f_degf.mean() - f_base.mean(),
                         f1_std=np.nan, precision=tf, recall=pf))

    out = RESULTS / "e7_actor_hus_detection.csv"
    pd.DataFrame(rows_out).to_csv(out, index=False)
    print(f"\ndone ({time.time()-t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
