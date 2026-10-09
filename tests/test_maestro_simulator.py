"""Tests for MaestroSimulator."""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import yaml
import numpy as np
import pytest
from dqgm import (
    Model,
    QiskitSimulator,
    MaestroSimulator,
    counts_to_probabilities,
)


@pytest.fixture
def exp1_model():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "configs", "exp1_gaussian.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = Model(**cfg["model"])
    checkpoint_path = os.path.join(
        repo_root,
        "notebooks",
        "logs",
        "exp1_gaussian",
        "run_2026-06-25_14-33-08-722421",
        "params_step10000.npz",
    )
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path)
        model.params = data["params"]
    else:
        model.init_parameters()
    return model


def test_maestro_statevector_exact_equivalence(exp1_model):
    """Verify that Maestro exact statevector matches Qiskit Aer to machine precision."""
    qiskit_sim = QiskitSimulator(exp1_model)
    maestro_sim = MaestroSimulator(exp1_model, simulation_type="Statevector")

    p_qiskit = qiskit_sim.exact_output_probabilities(exp1_model.params)
    p_maestro = maestro_sim.exact_output_probabilities(
        exp1_model.params, exact_statevector=True
    )

    assert set(p_qiskit.keys()) == set(p_maestro.keys())
    max_diff = max(abs(p_qiskit[k] - p_maestro[k]) for k in p_qiskit)
    assert max_diff < 1e-12


def test_maestro_statevector_sampling(exp1_model):
    """Verify that Maestro sampled counts correlate with exact distribution."""
    qiskit_sim = QiskitSimulator(exp1_model)
    maestro_sim = MaestroSimulator(exp1_model, simulation_type="Statevector")

    probs_exact = counts_to_probabilities(
        qiskit_sim.exact_output_probabilities(exp1_model.params),
        exp1_model.num_feature,
        exp1_model.readout_size,
    ).flatten()

    counts_sampled = maestro_sim.output_samples(exp1_model.params, num_samples=100_000)
    probs_sampled = counts_to_probabilities(
        counts_sampled, exp1_model.num_feature, exp1_model.readout_size
    ).flatten()

    corr = np.corrcoef(probs_exact, probs_sampled)[0, 1]
    assert corr > 0.80


def test_maestro_mps_scaling_34_qubits():
    """Verify that Maestro MPS simulates 34 qubits with full inverse QFT."""
    model = Model(
        num_visible_per_feature=30,
        num_feature=1,
        num_hidden_shared=4,
        multivariate_bits=False,
        seed=42,
        init_scale=0.1,
        init_method="random",
        freq_buffer=1,
    )
    X_dummy = np.zeros((10, 1), dtype=int)
    model.init_parameters(config_data={"dataset": "gaussian"}, X_train=X_dummy)

    assert model.num_qubits == 34
    assert model.readout_size == 29

    mps_sim = MaestroSimulator(
        model,
        simulation_type="MatrixProductState",
        max_bond_dimension=64,
        mps_sampling="apply_measure",
    )
    counts = mps_sim.output_samples(model.params, num_samples=20)
    assert sum(counts.values()) == 20
    # Every bitstring should have readout length (29 bits)
    for b in counts.keys():
        assert len(b) == 29


def test_maestro_mps_multivariate_bits():
    """Verify that Maestro MPS simulates 50 qubits in multi-bit discrete mode."""
    model = Model(
        num_visible_per_feature=45,
        num_feature=1,
        num_hidden_shared=5,
        multivariate_bits=True,
        seed=42,
        init_scale=0.1,
        init_method="random",
        freq_buffer=0,
        custom_connectivity={"hidden_locality": 6},
    )
    X_dummy = np.zeros((10, 45), dtype=int)
    model.init_parameters(config_data={"dataset": "blobs"}, X_train=X_dummy)

    assert model.num_qubits == 50

    mps_sim = MaestroSimulator(
        model,
        simulation_type="MatrixProductState",
        max_bond_dimension=64,
        mps_sampling="apply_measure",
    )
    counts = mps_sim.output_samples(model.params, num_samples=50)
    assert sum(counts.values()) == 50
    for b in counts.keys():
        assert len(b) == 45
