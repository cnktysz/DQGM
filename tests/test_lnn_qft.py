"""Tests for Linear Nearest-Neighbor (LNN) Inverse QFT implementation."""

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
from dqgm.qiskit_simulator import build_lnn_iqft


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


def test_build_lnn_iqft_connectivity():
    """Verify that build_lnn_iqft has strictly linear nearest-neighbor 2-qubit gates."""
    for n in [2, 3, 5, 8]:
        qc = build_lnn_iqft(n)
        assert qc.num_qubits == n
        for instruction in qc.data:
            qubits = [qc.find_bit(q).index for q in instruction.qubits]
            if len(qubits) == 2:
                q1, q2 = qubits
                assert abs(q1 - q2) == 1, f"Gate {instruction.operation.name} on non-neighbor qubits {q1}, {q2}"


def test_lnn_iqft_exact_distribution_equivalence(exp1_model):
    """Verify that use_lnn_qft=True produces identical exact distribution to standard QFT."""
    sim_standard = QiskitSimulator(exp1_model, use_lnn_qft=False)
    sim_lnn = QiskitSimulator(exp1_model, use_lnn_qft=True)

    probs_standard = sim_standard.exact_output_probabilities(exp1_model.params)
    probs_lnn = sim_lnn.exact_output_probabilities(exp1_model.params)

    assert set(probs_standard.keys()) == set(probs_lnn.keys())
    max_diff = max(abs(probs_standard[k] - probs_lnn[k]) for k in probs_standard)
    assert max_diff < 1e-10, f"Max probability difference: {max_diff}"


def test_lnn_iqft_maestro_statevector_equivalence(exp1_model):
    """Verify MaestroSimulator with use_lnn_qft=True matches standard exact probabilities."""
    sim_qiskit = QiskitSimulator(exp1_model, use_lnn_qft=False)
    sim_maestro_lnn = MaestroSimulator(exp1_model, simulation_type="Statevector", use_lnn_qft=True)

    probs_qiskit = sim_qiskit.exact_output_probabilities(exp1_model.params)
    probs_maestro = sim_maestro_lnn.exact_output_probabilities(
        exp1_model.params, exact_statevector=True
    )

    assert set(probs_qiskit.keys()) == set(probs_maestro.keys())
    max_diff = max(abs(probs_qiskit[k] - probs_maestro[k]) for k in probs_qiskit)
    assert max_diff < 1e-10, f"Max probability difference: {max_diff}"


def test_lnn_iqft_maestro_mps_sampling(exp1_model):
    """Verify Maestro MPS runs successfully with use_lnn_qft=True and samples valid bitstrings."""
    sim_mps_lnn = MaestroSimulator(
        exp1_model,
        simulation_type="MatrixProductState",
        max_bond_dimension=64,
        mps_sampling="probabilities",
        use_lnn_qft=True,
    )
    counts = sim_mps_lnn.output_samples(exp1_model.params, num_samples=1000)
    assert sum(counts.values()) == 1000
    for b in counts.keys():
        assert len(b) == exp1_model.readout_size
