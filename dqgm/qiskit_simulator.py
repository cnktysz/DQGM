"""Qiskit-based DQGM circuit sampler using Aer.

This simulator builds the full quantum circuit and samples from it.  It is used
at evaluation time to obtain samples from the model distribution (e.g. for
computing KL divergence against the training set).  For training, the
ClassicalSimulator is used instead — it computes gradients analytically without
running a circuit.

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

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from qiskit import transpile
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, RZGate, RZZGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def build_lnn_iqft(num_qubits: int) -> QuantumCircuit:
    """Build an inverse Quantum Fourier Transform using strictly linear nearest-neighbor gates.

    Implements the Fowler et al. (2004) / Woolfe et al. (2014) LNN QFT scheme.
    Each non-local controlled-phase gate CP(θ) is interleaved with a SWAP gate,
    rippling the active qubit across adjacent physical wires.
    This eliminates all long-range entangling gates, making the readout stage
    optimal for 1D Matrix Product State (MPS) tensor network contractions.

    Note that the final physical qubit positions are reversed relative to the input
    logical wires. Physical wire `k` holds logical qubit `num_qubits - 1 - k`.
    """
    qc = QuantumCircuit(num_qubits, name="lnn_iqft")
    phys_to_log = list(range(num_qubits))

    for j in range(num_qubits):
        if j == 0:
            qc.h(0)
        else:
            for step in range(j, 0, -1):
                p_act = step
                p_nbr = step - 1
                log_nbr = phys_to_log[p_nbr]
                angle = -np.pi / (2 ** (j - log_nbr))
                qc.cp(angle, p_act, p_nbr)
                qc.swap(p_act, p_nbr)
                phys_to_log[p_act], phys_to_log[p_nbr] = phys_to_log[p_nbr], phys_to_log[p_act]
            qc.h(0)

    return qc


class QiskitSimulator:
    """Sample-based DQGM circuit simulator built on Qiskit Aer.

    Parameters
    ----------
    model : Model instance — provides qubit counts and gate connectivity.
    use_lnn_qft : bool, default False.
        If True, synthesizes the inverse QFT with linear nearest-neighbor (LNN)
        connectivity using interleaved SWAP gates (Woolfe et al. / Fowler et al.).
    """

    def __init__(self, model, use_lnn_qft: bool = False, *args, **kwargs):
        self.model = model
        self.use_lnn_qft = use_lnn_qft

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _measurement_qubits(self, num_upsampling: int = 0) -> list[int]:
        """Qubit indices that are measured, in the order they map to classical bits.

        Matches the ``circuit.measure`` order in ``_build_circuit`` so a
        statevector marginalised on these qargs produces the same bitstring
        convention as Aer's ``get_counts()``.
        """
        if self.model.multivariate_bits:
            return list(range(self.model.num_visible))
        n_qubits = self.model.num_qubits
        qargs: list[int] = []
        for i in range(self.model.num_feature):
            vis_start = i * self.model.num_visible_per_feature
            ancilla_start = n_qubits + i * num_upsampling
            wires = [
                *range(ancilla_start, ancilla_start + num_upsampling),
                *range(vis_start, vis_start + self.model.readout_size),
            ]
            if self.use_lnn_qft:
                qargs.extend([wires[len(wires) - 1 - k] for k in range(len(wires))])
            else:
                qargs.extend(wires)
        return qargs

    def _build_circuit(
        self, params: np.ndarray, num_upsampling: int = 0, measure: bool = True,
    ) -> QuantumCircuit:
        """Construct the IQP sampling circuit for the given parameters.

        Parameters
        ----------
        params         : (2, num_gates) float array — row 0 is the first gate layer,
                         row 1 is the second.
        num_upsampling : if >0, sample the trained model at higher precision by adding
                         ``num_upsampling`` ancilla qubits per feature, kept in |0⟩
                         (no Hadamard, no trained gates) and placed at the MSB end of
                         the per-feature inverse QFT.  Equivalent to zero-padding the
                         high-frequency Fourier coefficients, which gives an
                         interpolated (higher-resolution) position-domain distribution.
                         Not supported with ``multivariate_bits=True``.
        """
        if num_upsampling < 0:
            raise ValueError("num_upsampling must be >= 0.")
        if num_upsampling > 0 and self.model.multivariate_bits:
            raise ValueError(
                "num_upsampling > 0 is not supported with multivariate_bits=True "
                "(no Fourier basis to upsample)."
            )

        n_qubits        = self.model.num_qubits
        total_ancilla   = num_upsampling * self.model.num_feature
        bits_per_feat   = self.model.readout_size + num_upsampling
        n_measure       = bits_per_feat * self.model.num_feature

        q_reg = QuantumRegister(n_qubits + total_ancilla)
        c_reg = ClassicalRegister(n_measure, name="readout")
        circuit = QuantumCircuit(q_reg, c_reg, name="DQGM-IQP")

        # 1. Initial Hadamard on trained qubits only; ancilla stay in |0⟩ so their
        #    frequency amplitudes are zero (zero-padding the high-frequency bins).
        for i in range(n_qubits):
            circuit.h(i)

        # 2. First IQP layer (single-qubit RZ and two-qubit RZZ gates).
        for idg, wires in enumerate(self.model.gate_list):
            if len(wires) == 1:
                circuit.compose(RZGate(-params[1, idg]), qubits=wires, inplace=True)
            else:
                v, h = wires[0], wires[1] + self.model.num_visible
                circuit.compose(RZZGate(-params[1, idg]), qubits=(v, h), inplace=True)

        # 3. Ry(π/2) rotation on visible qubits (mixes computational basis).
        for i in range(self.model.num_visible):
            circuit.ry(-np.pi / 2, i)

        # 4. Second IQP layer.
        for idg, wires in enumerate(self.model.gate_list):
            if len(wires) == 1:
                circuit.compose(RZGate(-params[0, idg]), qubits=wires, inplace=True)
            else:
                v, h = wires[0], wires[1] + self.model.num_visible
                circuit.compose(RZZGate(-params[0, idg]), qubits=(v, h), inplace=True)

        # 5. Hadamard on hidden qubits.
        for i in range(self.model.num_visible, n_qubits):
            circuit.h(i)

        # 6 & 7. Readout: Fourier decoding + measurement (mode-dependent).
        if self.model.multivariate_bits:
            # Each visible qubit is an independent bit — the "inverse QFT"
            # reduces to a single Hadamard per qubit (X-basis measurement).
            for i in range(self.model.num_visible):
                circuit.h(i)
            if measure:
                for i in range(self.model.num_visible):
                    circuit.measure(i, i)
        else:
            iqft_size = self.model.readout_size + num_upsampling
            if self.use_lnn_qft:
                inv_qft = build_lnn_iqft(iqft_size)
            else:
                inv_qft = QFT(
                    num_qubits=iqft_size,
                    inverse=True,
                    do_swaps=False,
                    insert_barriers=False,
                )
            for i in range(self.model.num_feature):
                vis_start     = i * self.model.num_visible_per_feature
                ancilla_start = n_qubits + i * num_upsampling
                # Ancilla (|0⟩) at the MSB end of the IQFT, trained visibles after.
                wires = [
                    *range(ancilla_start, ancilla_start + num_upsampling),
                    *range(vis_start, vis_start + self.model.readout_size),
                ]
                circuit.compose(inv_qft, qubits=wires, inplace=True)

            if measure:
                counter = 0
                for i in range(self.model.num_feature):
                    vis_start     = i * self.model.num_visible_per_feature
                    ancilla_start = n_qubits + i * num_upsampling
                    wires = [
                        *range(ancilla_start, ancilla_start + num_upsampling),
                        *range(vis_start, vis_start + self.model.readout_size),
                    ]
                    for k in range(len(wires)):
                        q = wires[len(wires) - 1 - k] if self.use_lnn_qft else wires[k]
                        circuit.measure(q, counter)
                        counter += 1

        return circuit

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def output_samples(
        self, params, num_samples: int = 1000, num_upsampling: int = 0
    ) -> dict:
        """Run the circuit and return a Qiskit counts dictionary.

        Parameters
        ----------
        params         : (2, num_gates) parameter array (JAX or NumPy)
        num_samples    : number of shots
        num_upsampling : extra precision bits per feature (default 0 = no upsampling).
                         Adds ``num_upsampling`` ancilla qubits per feature that stay
                         in |0⟩ and are appended at the MSB end of each per-feature
                         inverse QFT.  The returned bitstrings then carry
                         ``readout_size + num_upsampling`` bits per feature.  Pass the
                         extended per-feature bit count to ``counts_to_probabilities``
                         when converting counts to probability arrays.  Not supported
                         with ``multivariate_bits=True``.

        Returns
        -------
        dict mapping bitstring → count (Qiskit format).  In multivariate_bits
        mode the bitstrings are reversed so that bit f*V+j of the returned
        bitstring corresponds to the (f, j)-th data bit, matching the
        convention of ``counts_to_probabilities``.
        """
        circuit = self._build_circuit(np.array(params), num_upsampling=num_upsampling)
        simulator = AerSimulator()
        transpiled = transpile(circuit, backend=simulator, optimization_level=0)
        result = simulator.run(transpiled, shots=num_samples, seed_simulator=42).result()
        counts = result.get_counts()
        if self.model.multivariate_bits:
            counts = {bitstring[::-1]: c for bitstring, c in counts.items()}
        return counts

    def exact_output_probabilities(
        self, params, num_upsampling: int = 0
    ) -> dict:
        """Return the exact readout PMF — no shot noise.

        Builds the same circuit as ``output_samples`` but without the final
        measurement, evolves it with Aer's statevector backend, and
        marginalises the resulting state onto the measured qubits.  Output
        format mirrors ``output_samples``: a dict keyed by readout bitstring
        (same convention as Aer's ``get_counts``) with values being exact
        probabilities.  Pipe through ``counts_to_probabilities`` exactly like
        sampled counts.

        Use this instead of ``output_samples`` whenever the target
        distribution is known analytically (so the comparison is exact on
        both sides).  Cost is dominated by the 2^num_qubits-amplitude state
        evolution; on CPU this is ~1 s at 25 qubits and grows 2× per qubit.
        """
        circuit = self._build_circuit(
            np.array(params), num_upsampling=num_upsampling, measure=False,
        )
        circuit.save_statevector()
        simulator = AerSimulator(method="statevector")
        transpiled = transpile(circuit, backend=simulator, optimization_level=0)
        sv = simulator.run(transpiled).result().get_statevector()
        state = Statevector(np.asarray(sv))
        qargs = self._measurement_qubits(num_upsampling=num_upsampling)
        probs = state.probabilities_dict(qargs=qargs)
        probs = {str(b): float(p) for b, p in probs.items()}
        if self.model.multivariate_bits:
            probs = {bitstring[::-1]: p for bitstring, p in probs.items()}
        return probs
