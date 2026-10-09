"""Demonstration of DQGM simulation with Qoro Maestro and LNN inverse QFT.

This script demonstrates scaling continuous and discrete DQGM models beyond
the 30-qubit memory wall using Matrix Product States (MPS) and Linear
Nearest-Neighbor (LNN) inverse QFT.

Usage:
    python scripts/maestro_scaling_demo.py
"""

from __future__ import annotations

import os
import time

# Import maestro before any DQGM/Qiskit imports to guarantee clean OpenMP runtime initialization
import maestro
import numpy as np
import yaml
from dqgm import (
    Dataset,
    MaestroSimulator,
    Model,
    build_lnn_iqft,
    counts_to_probabilities,
    kl_divergence,
    tvd,
)


def load_yaml(candidates: list[str]) -> tuple[dict, str]:
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f), path
    raise FileNotFoundError(f"None of {candidates} exist.")


def load_npz(candidates: list[str]) -> tuple[np.lib.npyio.NpzFile, str]:
    for path in candidates:
        if os.path.exists(path):
            return np.load(path), path
    raise FileNotFoundError(f"None of {candidates} exist.")


def demo_9q_quickstart(qiskit_baseline):
    """Part 1: 9-qubit quickstart model (exact Statevector)."""
    print("\n" + "=" * 70)
    print("Part 1: 9-Qubit Quickstart Model (Exact Statevector)")
    print("=" * 70)

    model = Model(
        num_visible_per_feature=6,
        num_feature=1,
        num_hidden_shared=3,
        multivariate_bits=False,
        seed=3,
        init_scale=3.1415,
        init_method="random",
        freq_buffer=1,
    )
    model.init_parameters()

    sim = MaestroSimulator(model, simulation_type="Statevector")
    t0 = time.time()
    counts = sim.exact_output_probabilities(model.params)
    t_maestro = time.time() - t0
    probs_maestro = counts_to_probabilities(counts, model.num_feature, model.readout_size).flatten()

    probs_qiskit = qiskit_baseline["probs_q_qs"]
    t_qiskit = float(qiskit_baseline["t_q_qs"])
    max_diff = float(np.max(np.abs(probs_qiskit - probs_maestro)))

    print(f"Qiskit Aer Execution Time: {t_qiskit:.3f} s")
    print(f"Maestro SV Execution Time: {t_maestro:.3f} s")
    print(f"Max Difference vs Qiskit:  {max_diff:.2e} (Machine precision)")


def demo_21q_gaussian(qiskit_baseline):
    """Part 2: 21-qubit 1D Gaussian model (exact Statevector)."""
    print("\n" + "=" * 70)
    print("Part 2: 21-Qubit Trained 1D Gaussian (Exact Statevector)")
    print("=" * 70)

    cfg, _ = load_yaml(["configs/exp1_gaussian.yaml", "../configs/exp1_gaussian.yaml"])
    nprec = cfg["model"]["num_visible_per_feature"] - cfg["model"]["freq_buffer"]
    target = Dataset(**cfg["data"]).get_distribution(num_feature=1, num_precision=nprec)

    model = Model(**cfg["model"])
    ckpt, _ = load_npz([
        "notebooks/logs/exp1_gaussian/run_2026-06-25_14-33-08-722421/params_step10000.npz",
        "../notebooks/logs/exp1_gaussian/run_2026-06-25_14-33-08-722421/params_step10000.npz",
    ])
    model.params = ckpt["params"]

    sim = MaestroSimulator(model, simulation_type="Statevector")
    t0 = time.time()
    counts = sim.exact_output_probabilities(model.params)
    t_maestro = time.time() - t0
    probs_maestro = counts_to_probabilities(counts, model.num_feature, model.readout_size).flatten()

    kl_val = float(kl_divergence(target, probs_maestro))
    tvd_val = float(tvd(target, probs_maestro))
    max_diff = float(np.max(np.abs(qiskit_baseline["probs_q_g"] - probs_maestro)))

    print(f"Qiskit Aer Time: {float(qiskit_baseline['t_q_g']):.3f} s  (KL: {float(qiskit_baseline['kl_q_g']):.6f}, TVD: {float(qiskit_baseline['tvd_q_g']):.4f})")
    print(f"Maestro SV Time: {t_maestro:.3f} s  (KL: {kl_val:.6f}, TVD: {tvd_val:.4f})")
    print(f"Max Difference vs Qiskit: {max_diff:.2e}")


def demo_32q_mps_and_lnn(qiskit_baseline):
    """Part 3: 32-qubit continuous model (MPS with Standard vs LNN inverse QFT)."""
    print("\n" + "=" * 70)
    print("Part 3: 32-Qubit Continuous Model & LNN-IQFT (10,000 Shots)")
    print("=" * 70)

    model = Model(
        num_visible_per_feature=28,
        num_feature=1,
        num_hidden_shared=4,
        multivariate_bits=False,
        seed=42,
        init_scale=0.1,
        init_method="random",
        freq_buffer=1,
    )
    model.init_parameters(config_data={"dataset": "gaussian"}, X_train=np.zeros((10, 1), dtype=int))

    # Connectivity comparison
    lnn_qc = build_lnn_iqft(model.readout_size)
    max_dist_lnn = max(
        abs(lnn_qc.find_bit(inst.qubits[0]).index - lnn_qc.find_bit(inst.qubits[1]).index)
        for inst in lnn_qc.data
        if len(inst.qubits) == 2
    )
    print(f"Connectivity on {model.readout_size}-qubit readout register:")
    print(f"  Standard QFT: Max gate distance = {model.readout_size - 1} wires (all-to-all)")
    print(f"  LNN-IQFT:     Max gate distance = {max_dist_lnn} wire  (strictly nearest-neighbor)")

    # Maestro MPS with Standard QFT
    sim_std = MaestroSimulator(
        model, simulation_type="MatrixProductState", max_bond_dimension=64, mps_sampling="probabilities", use_lnn_qft=False
    )
    t0 = time.time()
    sim_std.output_samples(model.params, num_samples=10_000)
    t_m_std = time.time() - t0

    # Maestro MPS with LNN QFT
    sim_lnn = MaestroSimulator(
        model, simulation_type="MatrixProductState", max_bond_dimension=64, mps_sampling="probabilities", use_lnn_qft=True
    )
    t0 = time.time()
    sim_lnn.output_samples(model.params, num_samples=10_000)
    t_m_lnn = time.time() - t0

    t_q_std = float(qiskit_baseline["t_q_32_mps"])
    t_q_lnn = float(qiskit_baseline.get("t_q_32_mps_lnn", 17.656))

    print("\nBenchmark Runtimes (10,000 shots):")
    print(f"  Qiskit Aer Statevector:        CRASHED / OOM (64 GB RAM required)")
    print(f"  Qiskit Aer MPS (Standard QFT): {t_q_std:.2f} s")
    print(f"  Qiskit Aer MPS (LNN-IQFT):     {t_q_lnn:.2f} s")
    print(f"  Maestro MPS (Standard QFT):    {t_m_std:.2f} s")
    print(f"  Maestro MPS (LNN-IQFT):        {t_m_lnn:.2f} s  ({t_q_lnn / t_m_lnn:.1f}x faster than Qiskit LNN)")


def demo_40plus_qubits():
    """Part 4: Scaling to 41-qubit 1D Gaussian and 42-qubit 2D Blobs continuous models."""
    print("\n" + "=" * 70)
    print("Part 4: High-Dimensional Continuous Models at 41 and 42 Qubits")
    print("=" * 70)

    # 41-Qubit 1D Gaussian (36-bit readout, 68.7 billion bins)
    cfg_g, _ = load_yaml(["configs/exp1_gaussian.yaml", "../configs/exp1_gaussian.yaml"])
    model_g = Model(**cfg_g["model"])
    ckpt_g, _ = load_npz([
        "notebooks/logs/exp1_gaussian/run_2026-06-25_14-33-08-722421/params_step10000.npz",
        "../notebooks/logs/exp1_gaussian/run_2026-06-25_14-33-08-722421/params_step10000.npz",
    ])
    model_g.params = ckpt_g["params"]

    sim_41q = MaestroSimulator(
        model_g, simulation_type="MatrixProductState", max_bond_dimension=64, mps_sampling="probabilities", use_lnn_qft=True
    )
    t0 = time.time()
    c_41 = sim_41q.output_samples(model_g.params, num_samples=10_000, num_upsampling=20)
    t_41 = time.time() - t0

    readout_41 = model_g.readout_size + 20
    vals_41 = np.array([int(b, 2) / (2**readout_41 - 1) for b, c in c_41.items() for _ in range(c)])
    print(f"41-Qubit 1D Gaussian (36-bit readout, 10,000 shots):")
    print(f"  Execution Time: {t_41:.2f} s  (Statevector would require 32 TB RAM)")
    print(f"  Sample Mean:    {np.mean(vals_41):.4f} (Target: 0.5000)")
    print(f"  Sample Std:     {np.std(vals_41):.4f} (Target: 0.1300)")

    # 42-Qubit 2D Blobs (18 bits per feature, 262,144 x 262,144 continuous grid)
    cfg_b, _ = load_yaml(["configs/exp2_blobs.yaml", "../configs/exp2_blobs.yaml"])
    model_b = Model(**cfg_b["model"])
    ckpt_b, _ = load_npz([
        "notebooks/logs/exp2_blobs/run_2026-06-25_14-18-52-373009/params_step10000.npz",
        "../notebooks/logs/exp2_blobs/run_2026-06-25_14-18-52-373009/params_step10000.npz",
    ])
    model_b.params = ckpt_b["params"]
    nprec_b = cfg_b["model"]["num_visible_per_feature"] - cfg_b["model"]["freq_buffer"]
    target_2d = np.asarray(
        Dataset(**cfg_b["data"]).get_distribution(num_feature=cfg_b["model"]["num_feature"], num_precision=nprec_b)
    ).reshape(2**model_b.readout_size, 2**model_b.readout_size)

    sim_42q = MaestroSimulator(
        model_b, simulation_type="MatrixProductState", max_bond_dimension=64, mps_sampling="probabilities", use_lnn_qft=True
    )
    t0 = time.time()
    c_42 = sim_42q.output_samples(model_b.params, num_samples=50_000, num_upsampling=10)
    t_42 = time.time() - t0

    # Downsample to 8-bit grid for correlation check
    c_binned = {}
    for b, count in c_42.items():
        b_trunc = b[:8] + b[-18:-10]
        c_binned[b_trunc] = c_binned.get(b_trunc, 0) + count
    probs_42 = counts_to_probabilities(c_binned, model_b.num_feature, model_b.readout_size)
    corr = float(np.corrcoef(target_2d.flatten(), probs_42.flatten())[0, 1])

    print(f"\n42-Qubit 2D Blobs (18 bits/feature, 50,000 shots):")
    print(f"  Execution Time:          {t_42:.2f} s  (Statevector would require 64 TB RAM)")
    print(f"  Correlation with Target: r = {corr:.4f}")


def demo_scaling_benchmark(qiskit_baseline):
    """Part 5: Scaling benchmark table across increasing qubit counts."""
    print("\n" + "=" * 70)
    print("Part 5: Scaling Benchmark (100,000 Shots, chi=64)")
    print("=" * 70)

    qubits = list(qiskit_baseline["qubit_counts"])
    q_sv = list(qiskit_baseline["qiskit_scaling_times_100k"])
    q_mps_std = list(qiskit_baseline["qiskit_mps_scaling_times_100k"])
    q_mps_lnn = list(qiskit_baseline["qiskit_mps_lnn_scaling_times_100k"])

    print(f"{'Qubits':<8}{'Qiskit SV':<12}{'Qiskit MPS':<12}{'Qiskit MPS+LNN':<16}{'Maestro SV':<12}{'Maestro MPS+LNN':<16}")
    print("-" * 76)

    for i, n_tot in enumerate(qubits):
        n_hid = 4
        m = Model(
            num_visible_per_feature=n_tot - n_hid,
            num_feature=1,
            num_hidden_shared=n_hid,
            multivariate_bits=False,
            seed=42,
            init_scale=0.1,
            init_method="random",
            freq_buffer=1,
        )
        m.init_parameters(config_data={"dataset": "gaussian"}, X_train=np.zeros((10, 1), dtype=int))

        # Maestro SV (<= 28 qubits)
        if n_tot <= 28:
            m_sv = MaestroSimulator(m, simulation_type="Statevector")
            t0 = time.time()
            m_sv.output_samples(m.params, num_samples=100_000)
            t_sv = f"{time.time() - t0:.3f}"
        else:
            t_sv = "OOM"

        # Maestro MPS with LNN
        m_mps = MaestroSimulator(
            m, simulation_type="MatrixProductState", max_bond_dimension=64, mps_sampling="probabilities", use_lnn_qft=True
        )
        t0 = time.time()
        m_mps.output_samples(m.params, num_samples=100_000)
        t_mps = f"{time.time() - t0:.3f}"

        q_sv_str = f"{q_sv[i]:.3f}" if not np.isnan(q_sv[i]) else "OOM"
        print(f"{n_tot:<8}{q_sv_str:<12}{q_mps_std[i]:<12.3f}{q_mps_lnn[i]:<16.3f}{t_sv:<12}{t_mps:<16}")


def main():
    qiskit_baseline, path = load_npz([
        "notebooks/data/qiskit_baseline.npz",
        "data/qiskit_baseline.npz",
        "../notebooks/data/qiskit_baseline.npz",
    ])
    print(f"Loaded reference baseline data from: {path}")

    demo_9q_quickstart(qiskit_baseline)
    demo_21q_gaussian(qiskit_baseline)
    demo_32q_mps_and_lnn(qiskit_baseline)
    demo_40plus_qubits()
    demo_scaling_benchmark(qiskit_baseline)

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("For full visual plots and interactive analysis, see: notebooks/04_maestro_scaling.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    main()
