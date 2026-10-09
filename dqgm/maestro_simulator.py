"""Qoro Maestro-based DQGM circuit sampler.

Drop-in replacement for QiskitSimulator using qoro-maestro (maestro).
Provides ultra-fast Statevector simulation (over 20x faster than Qiskit Aer)
and Matrix Product State (MPS) simulation for scaling beyond 30 qubits.

Circuit structure
-----------------
  1. Hadamard layer on all qubits.
  2. First RZ/RZZ (diagonal) layer — parametrised by row 1 of self.params.
  3. Ry(-π/2) on visible qubits.
  4. Second RZ/RZZ (diagonal) layer — parametrised by row 0 of self.params.
  5. Hadamard on hidden qubits.
  6. Inverse QFT on visible qubits (one block per feature).
  7. Measurement of visible qubits.
"""

from __future__ import annotations

from typing import Union

try:
    import maestro
    _MAESTRO_AVAILABLE = True
except ImportError:
    maestro = None
    _MAESTRO_AVAILABLE = False

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Statevector

from .qiskit_simulator import QiskitSimulator


class MaestroSimulator(QiskitSimulator):
    """Sample-based and exact DQGM circuit simulator built on Qoro Maestro.

    Parameters
    ----------
    model : Model instance — provides qubit counts and gate connectivity.
    simulation_type : str or maestro.SimulationType, default "Statevector".
        "Statevector" for exact simulation (recommended for <= 28 qubits).
        "MatrixProductState" for tensor-network simulation scaling beyond 30 qubits.
    max_bond_dimension : int, default 128 (used when simulation_type is MPS).
    mps_sampling : str, default "apply_measure".
    seed : int or None, default 42.
    """

    def __init__(
        self,
        model,
        simulation_type: Union[str, "maestro.SimulationType"] = "Statevector",
        max_bond_dimension: int | None = 128,
        mps_sampling: str = "probabilities",
        seed: int | None = 42,
        use_lnn_qft: bool = False,
        *args,
        **kwargs,
    ):
        if not _MAESTRO_AVAILABLE:
            raise ImportError(
                "MaestroSimulator requires the 'qoro-maestro' package. "
                "Install it with: pip install qoro-maestro (or pip install dqgm[maestro])"
            )
        super().__init__(model, use_lnn_qft=use_lnn_qft, *args, **kwargs)
        if isinstance(simulation_type, str):
            simulation_type = getattr(maestro.SimulationType, simulation_type)
        self.simulation_type = simulation_type
        self.max_bond_dimension = max_bond_dimension
        self.mps_sampling = mps_sampling
        self.seed = seed

    def _get_simulator_config(
        self, simulation_type: maestro.SimulationType | None = None
    ) -> maestro.SimulatorConfig:
        sim_type = simulation_type or self.simulation_type
        return maestro.SimulatorConfig(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=sim_type,
            max_bond_dimension=self.max_bond_dimension,
            mps_sampling=self.mps_sampling,
            seed=self.seed,
        )

    def output_samples(
        self, params, num_samples: int = 1000, num_upsampling: int = 0
    ) -> dict:
        """Run the circuit and return a counts dictionary matching Qiskit format.

        Parameters
        ----------
        params         : (2, num_gates) parameter array (JAX or NumPy)
        num_samples    : number of shots
        num_upsampling : extra precision bits per feature (default 0 = no upsampling).

        Returns
        -------
        dict mapping bitstring -> count (Qiskit format).
        """
        circuit = self._build_circuit(
            np.array(params), num_upsampling=num_upsampling, measure=True
        )
        decomposed = circuit.decompose()
        qasm_str = qasm2.dumps(decomposed)

        sim_config = self._get_simulator_config()
        raw = maestro.simple_execute(qasm_str, config=sim_config, shots=num_samples)

        if self.model.multivariate_bits:
            n_measure = self.model.num_visible
        else:
            bits_per_feat = self.model.readout_size + num_upsampling
            n_measure = bits_per_feat * self.model.num_feature

        counts = {}
        for b, c in raw["counts"].items():
            # Maestro returns bitstrings over all quantum registers with q0 leftmost.
            # Slice the measured qubits and reverse to match Qiskit's little-endian convention.
            meas = b[:n_measure][::-1]
            counts[meas] = counts.get(meas, 0) + c

        return counts

    def exact_output_probabilities(
        self,
        params,
        num_samples: int = 100_000,
        num_upsampling: int = 0,
        exact_statevector: bool | None = None,
    ) -> dict:
        """Return readout probabilities.

        For < 25 qubits with Statevector simulation, computes exact mathematical
        statevector amplitudes via vectorized marginalisation for sharp, exact results.
        For larger circuits (>= 25 qubits) or MPS simulation, approximates the PMF
        via sampling (default 100,000 shots).
        """
        if exact_statevector is None:
            # Default to exact statevector for < 25 qubits when using Statevector
            use_statevector = (
                self.model.num_qubits < 25
                and self.simulation_type == maestro.SimulationType.Statevector
            )
        else:
            use_statevector = (
                exact_statevector
                and self.model.num_qubits <= 28
                and self.simulation_type == maestro.SimulationType.Statevector
            )

        if use_statevector:
            circuit = self._build_circuit(
                np.array(params), num_upsampling=num_upsampling, measure=False
            )
            decomposed = circuit.decompose()
            qasm_str = qasm2.dumps(decomposed)

            circ = maestro.QasmToCirc().parse_and_translate(qasm_str)
            sim_config = self._get_simulator_config(
                simulation_type=maestro.SimulationType.Statevector
            )
            sv_raw = np.asarray(maestro.get_statevector(circ, config=sim_config))
            n_tot = self.model.num_qubits + (
                self.model.num_feature * num_upsampling
                if not self.model.multivariate_bits
                else 0
            )
            probs_tensor = (np.abs(sv_raw) ** 2).reshape([2] * n_tot)
            qargs = self._measurement_qubits(num_upsampling=num_upsampling)
            kept_axes = [n_tot - 1 - q for q in reversed(qargs)]
            unmeasured_axes = [n_tot - 1 - q for q in range(n_tot) if q not in qargs]
            transposed = np.transpose(probs_tensor, kept_axes + unmeasured_axes)
            if unmeasured_axes:
                sum_axes = tuple(range(len(kept_axes), n_tot))
                marg = transposed.sum(axis=sum_axes)
            else:
                marg = transposed

            num_bits = len(qargs)
            fmt = f"0{num_bits}b"
            marg_flat = marg.flatten()
            probs = {format(i, fmt): float(marg_flat[i]) for i in range(len(marg_flat))}
            if self.model.multivariate_bits:
                probs = {bitstring[::-1]: p for bitstring, p in probs.items()}
            return probs

        # For larger circuits (>= 25 qubits) or MPS, approximate PMF via high-shot sampling
        counts = self.output_samples(
            params, num_samples=num_samples, num_upsampling=num_upsampling
        )
        total_shots = sum(counts.values())
        return {b: float(c) / total_shots for b, c in counts.items()}
