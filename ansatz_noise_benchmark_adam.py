"""
ansatz_noise_benchmark_adam.py
=========================
Utilities for comparing symmetry-reduced CD ansatzes, Adaptive-CD, SAE-UCCSD
and a resource-matched hardware-efficient ansatz (HEA) with MindQuantum.

Intended workflow
-----------------
1. Keep molecular integrals / symmetry encoding in PySCF, Qiskit Nature and
   quantumsymmetry.
2. Build variational circuits in MindQuantum.
3. Optimize ideal circuits with mqvector.
4. Freeze the optimized parameters and evaluate noisy circuits with mqmatrix.
5. Save long-format CSV files for plotting robustness figures.

Reproducibility notes
---------------------
* The numerical implementations and public function signatures are retained
  from the research version; please record your exact dependency versions.
* Noise benchmarks use ideal optimized parameters (no noisy retraining).
* Circuit counts refer to the constructed MindQuantum gates; they are NOT
  automatically equivalent to counts after IBM ISA transpilation.
* ``CD-three-body`` means Pauli weight <= 3 at the generator-selection stage.
  It does NOT mean the hardware circuit contains native three-qubit gates.
  After Pauli-evolution decomposition, the circuit can still be counted in
  one- and two-qubit gates (typically a CNOT ladder for a weight-w string).
* For SAE-UCCSD, UCCSD excitation operators are generated in the ACTIVE SPACE
  specified by CAS=(n_electrons, n_spatial_orbitals), then mapped by the
  quantumsymmetry SAE mapper. This avoids mixing the full molecular orbital
  space with a symmetry-reduced CAS Hamiltonian.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.sparse.linalg import eigsh

# MindQuantum
from mindquantum import Simulator
from mindquantum.core.circuit import Circuit
try:
    from mindquantum.core.circuit import decompose_single_term_time_evolution
except ImportError:  # compatibility with releases exporting it at top level
    from mindquantum import decompose_single_term_time_evolution
from mindquantum.core.gates import RY, RZ, X, DepolarizingChannel
from mindquantum.core.operators import Hamiltonian, QubitOperator, commutator, TimeEvolution
from mindquantum.core.parameterresolver import ParameterResolver

# Noise adders. These imports are stable in recent MindQuantum releases, but
# are kept isolated here because older releases may have slightly different APIs.
try:
    from mindquantum.core.circuit import (
        MixerAdder,
        SequentialAdder,
        NoiseChannelAdder,
        DepolarizingChannelAdder,
        QubitNumberConstrain,
        NoiseExcluder,
    )
    _HAS_MQ_NOISE_ADDERS = True
except Exception:
    _HAS_MQ_NOISE_ADDERS = False

# Optional chemistry stack. Imported lazily when SAE-UCCSD is requested.


# -----------------------------------------------------------------------------
# Generic conversion utilities
# -----------------------------------------------------------------------------

def qiskit_pauli_to_mq(qubit_op: Any, tol: float = 1e-12) -> QubitOperator:
    """Convert a Qiskit SparsePauliOp-like object to MindQuantum QubitOperator.

    Parameters
    ----------
    qubit_op
        Object exposing ``paulis`` and ``coeffs`` (e.g. SparsePauliOp).
    tol
        Drop coefficients below this magnitude.
    """
    out = QubitOperator()
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        coeff = complex(coeff)
        if abs(coeff) < tol:
            continue
        label = pauli.to_label()
        # Qiskit labels are q_{n-1}...q_0; MindQuantum uses explicit indices.
        terms = [f"{p}{q}" for q, p in enumerate(label[::-1]) if p != "I"]
        out += QubitOperator(" ".join(terms), coeff)
    out.compress()
    return out


def generate_hamiltonian(qubit_op: Any) -> QubitOperator:
    """Alias for converting a Qiskit qubit Hamiltonian to MindQuantum."""
    return qiskit_pauli_to_mq(qubit_op)


# Backward-compatible spelling used in the accompanying notebook.
generate_hamitonian = generate_hamiltonian


def qiskit_to_mq_circuit(qiskit_circuit: Any) -> Circuit:
    """Convert a *non-parameterized* Qiskit circuit to MindQuantum via QASM2.

    This is used for the SAE Hartree-Fock reference circuit. Variational UCCSD
    generators are constructed directly in MindQuantum rather than passed
    through QASM.
    """
    from qiskit import qasm2
    from mindquantum.io import OpenQASM

    if getattr(qiskit_circuit, "num_parameters", 0):
        raise ValueError(
            "qiskit_to_mq_circuit is intended for non-parameterized circuits. "
            "Bind parameters before conversion."
        )
    return OpenQASM().from_string(qasm2.dumps(qiskit_circuit))


# -----------------------------------------------------------------------------
# Exact reference solution
# -----------------------------------------------------------------------------

def get_exact_ground_state(qubit_hamiltonian: Any) -> Tuple[float, np.ndarray]:
    """Return exact ground-state energy and normalized statevector."""
    h_sparse = qubit_hamiltonian.to_matrix(sparse=True)
    dim = h_sparse.shape[0]

    if dim <= 4:
        vals, vecs = np.linalg.eigh(h_sparse.toarray())
        idx = int(np.argmin(vals))
        e0 = float(np.real(vals[idx]))
        psi0 = np.asarray(vecs[:, idx], dtype=complex)
    else:
        vals, vecs = eigsh(h_sparse, k=1, which="SA")
        e0 = float(np.real(vals[0]))
        psi0 = np.asarray(vecs[:, 0], dtype=complex)

    psi0 /= np.linalg.norm(psi0)
    return e0, psi0


# -----------------------------------------------------------------------------
# Circuit metrics
# -----------------------------------------------------------------------------

def _gate_qubits(gate: Any) -> List[int]:
    obj = list(getattr(gate, "obj_qubits", []))
    ctrl = list(getattr(gate, "ctrl_qubits", []))
    # preserve order but remove duplicates
    return list(dict.fromkeys(obj + ctrl))


def get_circuit_metrics(circuit: Circuit) -> Dict[str, int]:
    """Count parameters, gates, two-qubit gates, CX-like gates and ASAP depth.

    ``n_2q`` is the most implementation-independent metric. ``n_cx`` detects
    controlled-X gates in MindQuantum's gate representation. For a CD Pauli
    evolution that has already been decomposed, these counts are hardware-level
    circuit counts and are independent of whether the original Pauli generator
    had weight 1, 2, 3, ... .
    """
    n_gates = n_1q = n_2q = n_multiq = n_cx = 0
    qdepth = [0] * circuit.n_qubits

    for gate in circuit:
        qubits = _gate_qubits(gate)
        if not qubits:
            continue

        n_gates += 1
        nq = len(qubits)
        if nq == 1:
            n_1q += 1
        elif nq == 2:
            n_2q += 1
        else:
            n_multiq += 1

        # In MindQuantum, CNOT is normally represented as X.on(target, control).
        name = gate.__class__.__name__.lower()
        obj = list(getattr(gate, "obj_qubits", []))
        ctrl = list(getattr(gate, "ctrl_qubits", []))
        if len(obj) == 1 and len(ctrl) == 1 and ("xgate" in name or "cnot" in name):
            n_cx += 1

        d = max(qdepth[q] for q in qubits) + 1
        for q in qubits:
            qdepth[q] = d

    return {
        "n_qubits": int(circuit.n_qubits),
        "n_params": int(len(circuit.params_name)),
        "n_gates": int(n_gates),
        "n_1q": int(n_1q),
        "n_2q": int(n_2q),
        "n_cx": int(n_cx),
        "n_multiq": int(n_multiq),
        "depth": int(max(qdepth) if qdepth else 0),
    }


# -----------------------------------------------------------------------------
# VQE optimization
# -----------------------------------------------------------------------------

def optimize_mq_vqe(
    ham: Hamiltonian,
    circuit: Circuit,
    method: str = "BFGS",
    maxiter: int = 500,
    theta0: Optional[Sequence[float]] = None,
    tol: Optional[float] = None,
    n_qubits: Optional[int] = None,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
) -> Tuple[Dict[str, float], Any, List[float]]:
    """Optimize a MindQuantum circuit with analytic gradients.

    ``method="Adam"`` uses a small NumPy implementation of Adam, so the same
    function works without adding a MindSpore training-layer dependency.  All
    other method names are forwarded to :func:`scipy.optimize.minimize`.
    """
    if int(maxiter) < 1:
        raise ValueError("maxiter must be at least 1.")
    # MindQuantum Circuit.n_qubits is inferred from qubits touched by gates.
    # SAE circuits can have idle logical qubits, so the simulator width must
    # follow the Hamiltonian/register width rather than circuit.n_qubits.
    if n_qubits is None:
        n_qubits = max(int(circuit.n_qubits), int(ham.n_qubits))
    sim = Simulator("mqvector", int(n_qubits))
    grad_ops = sim.get_expectation_with_grad(ham, circuit)
    n_params = len(circuit.params_name)

    if theta0 is None:
        theta0 = np.zeros(n_params, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)
    if theta0.size != n_params:
        raise ValueError(f"theta0 has {theta0.size} entries; circuit has {n_params} parameters.")

    history: List[float] = []

    def objective(theta: np.ndarray):
        energy, grad = grad_ops(theta)
        energy_val = float(np.real(np.asarray(energy).reshape(-1)[0]))
        grad_val = np.real(np.asarray(grad).reshape(-1, n_params)[0]).astype(float)
        history.append(energy_val)
        return energy_val, grad_val

    if str(method).upper() == "ADAM":
        theta = theta0.copy()
        first_moment = np.zeros_like(theta)
        second_moment = np.zeros_like(theta)
        final_grad = np.zeros_like(theta)
        success = True
        message = "Adam completed the requested number of iterations."

        for step in range(1, int(maxiter) + 1):
            _, grad = objective(theta)
            final_grad = grad
            grad_norm = float(np.linalg.norm(grad))
            if tol is not None and grad_norm <= float(tol):
                message = f"Adam gradient norm reached tolerance {tol}."
                break

            first_moment = beta1 * first_moment + (1.0 - beta1) * grad
            second_moment = beta2 * second_moment + (1.0 - beta2) * (grad * grad)
            m_hat = first_moment / (1.0 - beta1 ** step)
            v_hat = second_moment / (1.0 - beta2 ** step)
            theta -= float(learning_rate) * m_hat / (np.sqrt(v_hat) + float(adam_epsilon))
        else:
            step = int(maxiter)

        # Store the post-update energy.  This is important for checkpointing:
        # it keeps result.fun consistent with result.x.
        final_energy, final_grad = objective(theta)
        result = OptimizeResult(
            x=theta,
            fun=float(final_energy),
            jac=final_grad,
            nit=int(step),
            nfev=len(history),
            success=success,
            message=message,
        )
    else:
        result = minimize(
            objective,
            theta0,
            jac=True,
            method=method,
            tol=tol,
            options={"maxiter": int(maxiter)},
        )
    pr = {name: float(val) for name, val in zip(circuit.params_name, result.x)}
    return pr, result, history


# -----------------------------------------------------------------------------
# SAE-UCCSD
# -----------------------------------------------------------------------------

def active_space_particles(cas: Sequence[int], spin: int = 0) -> Tuple[int, int]:
    """Return (N_alpha, N_beta) from CAS=(N_e, N_orb) and PySCF/Qiskit spin.

    ``spin`` follows the usual convention N_alpha - N_beta = 2S.
    """
    n_e = int(cas[0])
    spin = int(spin)
    if (n_e + spin) % 2 != 0:
        raise ValueError("CAS electron number and spin are incompatible.")
    n_alpha = (n_e + spin) // 2
    n_beta = n_e - n_alpha
    if n_alpha < 0 or n_beta < 0:
        raise ValueError("Invalid particle numbers inferred from CAS and spin.")
    return n_alpha, n_beta

def format_atom_for_pyscf_driver(atom):
    """Convert nested molecular geometry to PySCFDriver format."""

    if isinstance(atom, str):
        return atom

    formatted = []

    for idx, entry in enumerate(atom):
        if isinstance(entry, str):
            formatted.append(entry)
            continue

        if len(entry) == 2:
            symbol, coordinates = entry
            x, y, z = coordinates

        elif len(entry) == 4:
            symbol, x, y, z = entry

        else:
            raise ValueError(
                f"atom[{idx}] must be "
                "(symbol, (x, y, z)) or (symbol, x, y, z)."
            )

        formatted.append(
            f"{symbol} {float(x):.16g} "
            f"{float(y):.16g} {float(z):.16g}"
        )

    return "; ".join(formatted)

def get_sae_uccsd_generators(
    atom: Any,
    basis: str,
    charge: int = 0,
    spin: int = 0,
    CAS: Optional[Sequence[int]] = None,
    encoding_type: str = "SAE",
) -> Tuple[List[Any], Any, Any, Dict[str, Any]]:
    """Generate active-space UCCSD excitation generators and map them.

    For SAE the quantumsymmetry mapper and HF circuit are retained, while the
    expensive VQE execution is NOT done in Qiskit.
    """
    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from quantumsymmetry import Encoding

    # IMPORTANT: generate the UCCSD excitation pool in the SAME fermionic
    # space as the original Qiskit implementation.  For SAE-CAS the
    # quantumsymmetry mapper itself performs the CAS/symmetry reduction.
    # Building the excitation pool directly in the reduced CAS orbital
    # indexing can be incompatible with that mapper and may collapse the
    # mapped pool to too few logical qubits.
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.units import DistanceUnit
    driver_atom = format_atom_for_pyscf_driver(atom)

    driver = PySCFDriver(
        atom=driver_atom,
        unit=DistanceUnit.ANGSTROM,
        charge=charge,
        spin=spin,
        basis=basis,
    )
    problem = driver.run()
    n_spatial = int(problem.num_spatial_orbitals)
    particles = tuple(int(x) for x in problem.num_particles)

    encoding_type = encoding_type.upper()

    if encoding_type == "SAE":
        if CAS is None:
            raise ValueError("SAE-UCCSD requires CAS=(n_electrons, n_spatial_orbitals).")
        encoding = Encoding(
            atom=atom,
            basis=basis,
            charge=charge,
            spin=spin,
            CAS=list(CAS),
            output_format="qiskit",
        )
        mapper = encoding.qiskit_mapper
        initial_state = encoding.HF_circuit
    elif encoding_type == "JW":
        mapper = JordanWignerMapper()
        initial_state = HartreeFock(
            num_spatial_orbitals=n_spatial,
            num_particles=particles,
            qubit_mapper=mapper,
        )
    else:
        raise ValueError("encoding_type must be 'SAE' or 'JW'.")

    # UCCSD is used ONLY to generate the full-space excitation operators;
    # the SAE-CAS mapper performs the reduction, matching the reference Qiskit workflow.
    ucc = UCCSD(
        num_spatial_orbitals=n_spatial,
        num_particles=particles,
        qubit_mapper=mapper,
    )
    fermionic_generators = ucc.excitation_ops()

    mapped = []
    for op in fermionic_generators:
        qop = mapper.map(op)
        if hasattr(qop, "simplify"):
            qop = qop.simplify()
        mapped.append(qop)

    meta = {
        "n_active_electrons": int(sum(particles)),
        "n_active_spatial_orbitals": int(n_spatial),
        "num_particles": tuple(particles),
        "n_excitations": int(len(mapped)),
        "encoding_type": encoding_type,
    }
    return mapped, initial_state, mapper, meta


def _hermitian_from_ucc_generator(mapped_generator: Any, tol: float = 1e-9) -> QubitOperator:
    """Convert anti-Hermitian K=T-T^dagger into Hermitian H=iK.

    The target unitary is exp(theta*K) = exp(-i*theta*H).
    """
    K = qiskit_pauli_to_mq(mapped_generator)
    H = 1j * K
    H.compress()

    bad = [(term, coeff) for term, coeff in H.terms.items() if abs(complex(coeff).imag) > tol]
    if bad:
        warnings.warn(
            "Mapped UCC generator did not become manifestly Hermitian after H=iK. "
            "Check the Qiskit Nature excitation_ops() convention for your installed version. "
            f"First offending term: {bad[0]}",
            RuntimeWarning,
        )
    return H


def build_sae_uccsd_mq(
    atom: Any,
    basis: str,
    charge: int = 0,
    spin: int = 0,
    CAS: Optional[Sequence[int]] = None,
    encoding_type: str = "SAE",
) -> Tuple[Circuit, Dict[str, Any]]:
    """Build an SAE/JW active-space UCCSD circuit as a MindQuantum Circuit."""
    mapped, initial_qiskit, mapper, meta = get_sae_uccsd_generators(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        CAS=CAS,
        encoding_type=encoding_type,
    )

    hf = qiskit_to_mq_circuit(initial_qiskit)
    ucc_circuit = Circuit()
    generator_meta = []

    for k, mapped_generator in enumerate(mapped):
        Hk = _hermitian_from_ucc_generator(mapped_generator)
        pname = f"ucc_{k}"
        evo = TimeEvolution(Hk, time=ParameterResolver(pname))
        ucc_circuit += evo.circuit
        generator_meta.append({
            "parameter": pname,
            "n_pauli_terms": len(Hk.terms),
        })

    full = hf + ucc_circuit
    meta = dict(meta)
    meta.update({
        "mapper": mapper,
        "hf_circuit_mq": hf,
        "n_qubits_touched": int(full.n_qubits),
        "n_parameters": int(len(full.params_name)),
        "generators": generator_meta,
    })
    return full, meta


def batch_sae_uccsd(
    atom_record: Sequence[Any],
    basis: str,
    CAS: Sequence[int],
    qubit_hamiltonian_record: Sequence[Any],
    charge: int = 0,
    spin: int = 0,
    encoding_type: str = "SAE",
    method: str = "BFGS",
    maxiter: int = 500,
    learning_rate: float = 0.01,
    tol: Optional[float] = None,
    theta0: Optional[Sequence[float]] = None,
    checkpoint_file: Optional[str] = "uccsd_ideal_results.csv",
    verbose: bool = True,
):
    """Build and optimize SAE-UCCSD for a geometry/Hamiltonian scan."""
    if len(atom_record) != len(qubit_hamiltonian_record):
        raise ValueError("atom_record and qubit_hamiltonian_record must have equal length.")

    circuits, params, energies, metadata, rows = [], [], [], [], []

    for idx, (atom, qham) in enumerate(zip(atom_record, qubit_hamiltonian_record)):
        if verbose:
            print(f"\n[UCCSD] {idx + 1}/{len(atom_record)}")

        circuit, meta = build_sae_uccsd_mq(
            atom=atom,
            basis=basis,
            charge=charge,
            spin=spin,
            CAS=CAS,
            encoding_type=encoding_type,
        )
        ham = Hamiltonian(generate_hamiltonian(qham))

        this_theta0 = theta0
        if theta0 is not None and len(theta0) != len(circuit.params_name):
            this_theta0 = None

        pr, result, history = optimize_mq_vqe(
                    ham=ham,
                    circuit=circuit,
                    method=method,
                    maxiter=maxiter,
                    theta0=this_theta0,
                    learning_rate=learning_rate,
                    tol=tol,
                    n_qubits=int(qham.num_qubits),
                )

        circuits.append(circuit)
        params.append(pr)
        energies.append(float(result.fun))
        metadata.append(meta)

        m = get_circuit_metrics(circuit)
        row = {
            "ham_idx": idx,
            "energy": float(result.fun),
            "success": bool(result.success),
            "nit": int(getattr(result, "nit", -1)),
            **m,
        }
        rows.append(row)

        if checkpoint_file:
            pd.DataFrame(rows).to_csv(checkpoint_file, index=False)

        if verbose:
            print(
                f"  qubits={m['n_qubits']}, params={m['n_params']}, "
                f"2q={m['n_2q']}, CX={m['n_cx']}, depth={m['depth']}, "
                f"E={result.fun:.12f}, success={result.success}"
            )

    return circuits, params, energies, metadata


# -----------------------------------------------------------------------------
# Hardware-efficient ansatz
# -----------------------------------------------------------------------------

def choose_hea_depth(n_qubits: int, target_n_params: int, rotations_per_qubit: int = 2) -> int:
    """Choose HEA layers to approximately match a target parameter count."""
    per_layer = max(1, rotations_per_qubit * int(n_qubits))
    return max(1, int(np.ceil(int(target_n_params) / per_layer)))


def build_hea(
    hf_circuit: Circuit,
    n_qubits: int,
    depth: int = 2,
    prefix: str = "hea",
    bidirectional_entanglers: bool = False,
) -> Circuit:
    """RY-RZ hardware-efficient ansatz with linear CX entanglers."""
    circuit = hf_circuit.copy()

    for layer in range(int(depth)):
        for q in range(int(n_qubits)):
            circuit += RY(f"{prefix}_ry_{layer}_{q}").on(q)
            circuit += RZ(f"{prefix}_rz_{layer}_{q}").on(q)

        for q in range(int(n_qubits) - 1):
            circuit += X.on(q + 1, q)

        if bidirectional_entanglers and layer % 2 == 1:
            for q in reversed(range(1, int(n_qubits))):
                circuit += X.on(q - 1, q)

    return circuit


def batch_hea(
    hf_circuit_record: Sequence[Circuit],
    qubit_hamiltonian_record: Sequence[Any],
    target_circuit_record: Sequence[Circuit],
    method: str = "Adam",
    maxiter: int = 200,
    theta0: Optional[Sequence[float]] = None,
    learning_rate: float = 0.01,
    tol: Optional[float] = None,
    checkpoint_file: Optional[str] = "hea_ideal_results.csv",
    verbose: bool = True,
):
    """Build resource-matched HEAs and optimize them with MindQuantum.

    The default now reproduces the reference MindSpore training convention:
    Adam with learning rate 0.01 for 200 parameter updates.  Set
    ``method="BFGS"`` to recover the previous behavior.
    """
    n = len(qubit_hamiltonian_record)
    if len(hf_circuit_record) != n or len(target_circuit_record) != n:
        raise ValueError("HF, Hamiltonian and target-circuit records must have equal length.")

    circuits, params, energies, depths, rows = [], [], [], [], []

    for idx, (hf, qham, target) in enumerate(
        zip(hf_circuit_record, qubit_hamiltonian_record, target_circuit_record)
    ):
        nq = target.n_qubits
        target_np = len(target.params_name)
        depth = choose_hea_depth(nq, target_np)
        circuit = build_hea(hf, nq, depth=depth, prefix=f"hea{idx}")
        ham = Hamiltonian(generate_hamiltonian(qham))

        this_theta0 = theta0
        if theta0 is not None and len(theta0) != len(circuit.params_name):
            this_theta0 = None

        pr, result, history = optimize_mq_vqe(
            ham=ham,
            circuit=circuit,
            method=method,
            maxiter=maxiter,
            theta0=this_theta0,
            learning_rate=learning_rate,
            tol=tol,
            n_qubits=int(qham.num_qubits),
        )

        circuits.append(circuit)
        params.append(pr)
        energies.append(float(result.fun))
        depths.append(depth)

        m = get_circuit_metrics(circuit)
        row = {
            "ham_idx": idx,
            "hea_layers": depth,
            "target_n_params": target_np,
            "energy": float(result.fun),
            "success": bool(result.success),
            **m,
        }
        rows.append(row)
        if checkpoint_file:
            pd.DataFrame(rows).to_csv(checkpoint_file, index=False)

        if verbose:
            print(
                f"[HEA] {idx + 1}/{n}: layers={depth}, params={m['n_params']} "
                f"(target={target_np}), 2q={m['n_2q']}, CX={m['n_cx']}, "
                f"E={result.fun:.12f}"
            )

    return circuits, params, energies, depths


# -----------------------------------------------------------------------------
# Adaptive-CD (ADAPT-VQE selection rule with a CD Pauli pool)
# -----------------------------------------------------------------------------

PauliTerm = Tuple[Tuple[int, str], ...]


def _is_pauli_term(obj: Any) -> bool:
    """Return True for a MindQuantum Pauli-word key such as ((0, 'X'), ...)."""
    if not isinstance(obj, tuple):
        return False
    return all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], (int, np.integer))
        and str(item[1]).upper() in {"X", "Y", "Z"}
        for item in obj
    )


def normalize_cd_pool(
    cd_pool: Any,
    max_pauli_weight: Optional[int] = None,
) -> List[PauliTerm]:
    """Normalize the notebook's CD-pool formats to unique Pauli-word keys.

    Accepted inputs are:

    * the ``OrderedDict`` returned by ``nonzero_CD_operator``;
    * a MindQuantum ``QubitOperator``;
    * a sequence of Pauli-word keys or single/multi-term ``QubitOperator`` s.

    Coefficients are intentionally discarded, matching the companion notebook:
    each retained CD Pauli word receives its own variational angle.
    """
    if isinstance(cd_pool, QubitOperator):
        raw_terms = list(cd_pool.terms)
    elif isinstance(cd_pool, Mapping):
        raw_terms = list(cd_pool.keys())
    else:
        raw_terms = []
        for item in cd_pool:
            if isinstance(item, QubitOperator):
                raw_terms.extend(item.terms)
            elif _is_pauli_term(item):
                raw_terms.append(item)
            else:
                raise TypeError(
                    "Each CD-pool entry must be a Pauli-word key or QubitOperator; "
                    f"received {type(item).__name__}."
                )

    unique: List[PauliTerm] = []
    seen = set()
    for raw in raw_terms:
        if not _is_pauli_term(raw):
            raise TypeError(f"Invalid MindQuantum Pauli-word key: {raw!r}")
        term = tuple((int(q), str(p).upper()) for q, p in raw)
        if not term:  # identity cannot generate a variational direction
            continue
        if max_pauli_weight is not None and len(term) > int(max_pauli_weight):
            continue
        if term not in seen:
            seen.add(term)
            unique.append(term)

    if not unique:
        raise ValueError("The normalized CD operator pool is empty.")
    return unique


def build_cd_pool(
    qubit_hamiltonian: Any,
    interaction_order: Optional[int] = None,
    z_coefficients: Optional[Sequence[float]] = None,
    coefficient_tol: float = 1e-4,
) -> List[PauliTerm]:
    """Build the same first-commutator CD pool used in the companion notebook.

    The reference operator is ``Z0 - Z1 - ...`` unless ``z_coefficients`` is
    supplied.  Pauli words are extracted from ``[H_ref, H]`` and filtered by
    Pauli weight, while their commutator coefficients are not used as fixed
    circuit amplitudes.
    """
    if isinstance(qubit_hamiltonian, QubitOperator):
        ham_mq = qubit_hamiltonian
    else:
        ham_mq = generate_hamiltonian(qubit_hamiltonian)

    if hasattr(qubit_hamiltonian, "num_qubits"):
        n_qubits = int(qubit_hamiltonian.num_qubits)
    elif hasattr(ham_mq, "count_qubits"):
        n_qubits = int(ham_mq.count_qubits())
    else:
        n_qubits = 1 + max((q for term in ham_mq.terms for q, _ in term), default=-1)
    if n_qubits < 1:
        raise ValueError("Cannot build a CD pool for a zero-qubit Hamiltonian.")
    if z_coefficients is None:
        z_coefficients = [1.0] + [-1.0] * max(0, n_qubits - 1)
    if len(z_coefficients) != n_qubits:
        raise ValueError("z_coefficients must contain one value per qubit.")

    reference = QubitOperator()
    for q, coeff in enumerate(z_coefficients):
        reference += QubitOperator(f"Z{q}", float(coeff))

    cd_commutator = commutator(reference, ham_mq)
    retained_terms: List[PauliTerm] = []
    above_tolerance = 0
    for term, coeff in cd_commutator.terms.items():
        if abs(complex(coeff)) <= float(coefficient_tol):
            continue
        above_tolerance += 1
        if interaction_order is not None and len(term) > int(interaction_order):
            continue
        # Keep the internal Pauli key directly.  Reconstructing a
        # QubitOperator from an internal tuple is version-dependent in
        # MindQuantum and can silently produce an empty operator.
        retained_terms.append(term)

    if not retained_terms:
        raise ValueError(
            "The CD operator pool is empty: "
            f"commutator_terms={len(cd_commutator.terms)}, "
            f"above_coefficient_tol={above_tolerance}, "
            f"interaction_order={interaction_order}, "
            f"coefficient_tol={coefficient_tol}. "
            "If commutator_terms=0, the chosen Z reference commutes with H; "
            "pass the previously generated cd_pool_record instead. If only "
            "the interaction-order filter removes the terms, increase "
            "interaction_order or set it to None."
        )
    return normalize_cd_pool(retained_terms)


def _pauli_term_label(term: PauliTerm) -> str:
    return " ".join(f"{p}{q}" for q, p in term)


def _append_pauli_evolution(circuit: Circuit, term: PauliTerm, parameter: str) -> Circuit:
    out = circuit.copy()
    out += decompose_single_term_time_evolution(term, {parameter: 1.0})
    return out


def _energy_at_parameters(
    ham: Hamiltonian,
    circuit: Circuit,
    parameters: Mapping[str, float],
    n_qubits: int,
) -> float:
    sim = Simulator("mqvector", int(n_qubits))
    sim.apply_circuit(circuit.apply_value(dict(parameters)))
    return float(np.real(sim.get_expectation(ham)))


def adaptive_cd_pool_gradients(
    ham: Hamiltonian,
    circuit: Circuit,
    parameters: Mapping[str, float],
    cd_pool: Any,
    n_qubits: Optional[int] = None,
    candidate_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Evaluate the ADAPT energy gradient for every eligible CD operator.

    Each candidate is appended with a zero-valued probe parameter, and
    MindQuantum's analytic circuit derivative supplies
    ``d<E>/d theta_mu |_(theta_mu=0)``.  This is equivalent to measuring the
    Hamiltonian/operator commutator used in ADAPT-VQE, while avoiding fragile
    sign and generator-normalization assumptions.
    """
    pool = normalize_cd_pool(cd_pool)
    if candidate_indices is None:
        indices = list(range(len(pool)))
    else:
        indices = [int(i) for i in candidate_indices]
    if not indices:
        return np.asarray([], dtype=float), []

    if n_qubits is None:
        n_qubits = max(int(circuit.n_qubits), int(ham.n_qubits))
    gradients: List[float] = []

    for pool_index in indices:
        if pool_index < 0 or pool_index >= len(pool):
            raise IndexError(f"CD-pool index {pool_index} is out of range.")
        probe_name = "__adaptive_cd_probe__"
        while probe_name in circuit.params_name:
            probe_name += "_"
        trial = _append_pauli_evolution(circuit, pool[pool_index], probe_name)
        theta = np.asarray(
            [float(parameters.get(name, 0.0)) for name in trial.params_name],
            dtype=float,
        )
        sim = Simulator("mqvector", int(n_qubits))
        grad_ops = sim.get_expectation_with_grad(ham, trial)
        _, grad = grad_ops(theta)
        grad_vec = np.real(np.asarray(grad).reshape(-1, len(trial.params_name))[0])
        gradients.append(float(grad_vec[trial.params_name.index(probe_name)]))

    return np.asarray(gradients, dtype=float), indices


def adaptive_cd_vqe(
    hf_circuit: Circuit,
    qubit_hamiltonian: Any,
    cd_pool: Optional[Any] = None,
    interaction_order: Optional[int] = None,
    gradient_threshold: float = 1e-3,
    max_adapt_iterations: int = 50,
    optimizer: str = "Adam",
    optimizer_maxiter: int = 200,
    optimizer_tol: Optional[float] = None,
    learning_rate: float = 0.01,
    allow_repeated_operators: bool = True,
    verbose: bool = True,
) -> Tuple[Circuit, Dict[str, float], float, Dict[str, Any]]:
    """Grow and optimize an Adaptive-CD ansatz.

    This follows Grimsley *et al.* (2019): compute the complete pool-gradient
    vector, stop when its L2 norm is below ``gradient_threshold``, append the
    largest-absolute-gradient operator, and re-optimize every ansatz parameter.
    By default the pool is not drained, so a Pauli word may be selected again.
    """
    ham_mq_op = (
        qubit_hamiltonian
        if isinstance(qubit_hamiltonian, QubitOperator)
        else generate_hamiltonian(qubit_hamiltonian)
    )
    ham = Hamiltonian(ham_mq_op)
    n_qubits = max(int(hf_circuit.n_qubits), int(ham.n_qubits))
    pool = (
        build_cd_pool(
            qubit_hamiltonian,
            interaction_order=interaction_order,
        )
        if cd_pool is None
        else normalize_cd_pool(cd_pool, max_pauli_weight=interaction_order)
    )

    circuit = hf_circuit.copy()
    parameters: Dict[str, float] = {name: 0.0 for name in circuit.params_name}
    if circuit.params_name:
        parameters, initial_result, _ = optimize_mq_vqe(
            ham,
            circuit,
            method=optimizer,
            maxiter=optimizer_maxiter,
            theta0=[parameters[name] for name in circuit.params_name],
            tol=optimizer_tol,
            n_qubits=n_qubits,
            learning_rate=learning_rate,
        )
        energy = float(initial_result.fun)
    else:
        initial_result = None
        energy = _energy_at_parameters(ham, circuit, parameters, n_qubits)

    selected_indices: List[int] = []
    selected_terms: List[str] = []
    selected_gradients: List[float] = []
    gradient_norms: List[float] = []
    energies: List[float] = [energy]
    optimizer_success: List[bool] = []
    converged = False
    eligible = list(range(len(pool)))

    for adapt_iteration in range(int(max_adapt_iterations)):
        gradients, evaluated_indices = adaptive_cd_pool_gradients(
            ham=ham,
            circuit=circuit,
            parameters=parameters,
            cd_pool=pool,
            n_qubits=n_qubits,
            candidate_indices=eligible,
        )
        gradient_norm = float(np.linalg.norm(gradients))
        gradient_norms.append(gradient_norm)

        if verbose:
            print(
                f"[Adaptive-CD] iter={adapt_iteration:02d}, "
                f"pool_grad_norm={gradient_norm:.6e}, E={energy:.12f}"
            )
        if gradient_norm <= float(gradient_threshold):
            converged = True
            break
        if gradients.size == 0:
            break

        local_choice = int(np.argmax(np.abs(gradients)))
        pool_index = int(evaluated_indices[local_choice])
        chosen_gradient = float(gradients[local_choice])
        parameter_name = f"adaptive_cd_{adapt_iteration}"
        circuit = _append_pauli_evolution(circuit, pool[pool_index], parameter_name)

        theta0 = [float(parameters.get(name, 0.0)) for name in circuit.params_name]
        parameters, result, _ = optimize_mq_vqe(
            ham=ham,
            circuit=circuit,
            method=optimizer,
            maxiter=optimizer_maxiter,
            theta0=theta0,
            tol=optimizer_tol,
            n_qubits=n_qubits,
            learning_rate=learning_rate,
        )
        energy = float(result.fun)
        selected_indices.append(pool_index)
        selected_terms.append(_pauli_term_label(pool[pool_index]))
        selected_gradients.append(chosen_gradient)
        energies.append(energy)
        optimizer_success.append(bool(result.success))

        if not allow_repeated_operators:
            eligible.remove(pool_index)

    metadata = {
        "converged": bool(converged),
        "gradient_threshold": float(gradient_threshold),
        "final_gradient_norm": gradient_norms[-1] if gradient_norms else np.nan,
        "n_adapt_iterations": len(selected_indices),
        "pool_size": len(pool),
        "allow_repeated_operators": bool(allow_repeated_operators),
        "selected_pool_indices": selected_indices,
        "selected_terms": selected_terms,
        "selected_gradients": selected_gradients,
        "gradient_norms": gradient_norms,
        "energies": energies,
        "optimizer_success": optimizer_success,
    }
    return circuit, parameters, float(energy), metadata


def batch_adaptive_cd(
    hf_circuit_record: Sequence[Circuit],
    qubit_hamiltonian_record: Sequence[Any],
    cd_pool_record: Optional[Sequence[Any]] = None,
    interaction_order: Optional[int] = None,
    gradient_threshold: float = 1e-3,
    max_adapt_iterations: int = 50,
    optimizer: str = "Adam",
    optimizer_maxiter: int = 200,
    optimizer_tol: Optional[float] = None,
    learning_rate: float = 0.01,
    allow_repeated_operators: bool = True,
    checkpoint_file: Optional[str] = "adaptive_cd_ideal_results.csv",
    verbose: bool = True,
):
    """Run Adaptive-CD over a molecular geometry/Hamiltonian scan."""
    n = len(qubit_hamiltonian_record)
    if len(hf_circuit_record) != n:
        raise ValueError("HF and Hamiltonian records must have equal length.")
    if cd_pool_record is not None and len(cd_pool_record) != n:
        raise ValueError("cd_pool_record must match the Hamiltonian record length.")

    circuits, params, energies, metadata_record, rows = [], [], [], [], []
    for idx, (hf, qham) in enumerate(zip(hf_circuit_record, qubit_hamiltonian_record)):
        pool = None if cd_pool_record is None else cd_pool_record[idx]
        circuit, pr, energy, meta = adaptive_cd_vqe(
            hf_circuit=hf,
            qubit_hamiltonian=qham,
            cd_pool=pool,
            interaction_order=interaction_order,
            gradient_threshold=gradient_threshold,
            max_adapt_iterations=max_adapt_iterations,
            optimizer=optimizer,
            optimizer_maxiter=optimizer_maxiter,
            optimizer_tol=optimizer_tol,
            learning_rate=learning_rate,
            allow_repeated_operators=allow_repeated_operators,
            verbose=verbose,
        )
        circuits.append(circuit)
        params.append(pr)
        energies.append(energy)
        metadata_record.append(meta)

        metrics = get_circuit_metrics(circuit)
        rows.append({
            "ham_idx": idx,
            "energy": energy,
            "success": bool(meta["converged"]),
            "n_adapt_iterations": int(meta["n_adapt_iterations"]),
            "pool_size": int(meta["pool_size"]),
            "final_gradient_norm": float(meta["final_gradient_norm"]),
            "selected_pool_indices": json.dumps(meta["selected_pool_indices"]),
            "selected_terms": json.dumps(meta["selected_terms"]),
            **metrics,
        })
        if checkpoint_file:
            pd.DataFrame(rows).to_csv(checkpoint_file, index=False)

    return circuits, params, energies, metadata_record


# -----------------------------------------------------------------------------
# Depolarizing noise and noisy observables
# -----------------------------------------------------------------------------

def build_depolarizing_noise_model(p2: float, p1_ratio: float = 0.1):
    """Construct gate-dependent depolarizing noise: p1=p1_ratio*p2."""
    if not _HAS_MQ_NOISE_ADDERS:
        raise ImportError(
            "This MindQuantum version does not expose the circuit noise-adder API used here. "
            "Upgrade MindQuantum or adapt build_depolarizing_noise_model to your version."
        )

    p2 = float(p2)
    p1 = float(p1_ratio) * p2

    one_qubit_noise = MixerAdder([
        QubitNumberConstrain(1),
        NoiseExcluder(),
        NoiseChannelAdder(DepolarizingChannel(p1)),
    ])

    two_qubit_noise = MixerAdder([
        QubitNumberConstrain(2),
        NoiseExcluder(),
        DepolarizingChannelAdder(p2, n_qubits=2),
    ])

    return SequentialAdder([one_qubit_noise, two_qubit_noise])


def evaluate_ideal_circuit(
    ham_mq: Hamiltonian,
    circuit: Circuit,
    parameters: Mapping[str, float],
    n_qubits: Optional[int] = None,
) -> Tuple[float, np.ndarray, Circuit]:
    """Return ideal energy, statevector and parameter-bound circuit."""
    bound = circuit.apply_value(dict(parameters))
    if n_qubits is None:
        n_qubits = max(int(circuit.n_qubits), int(ham_mq.n_qubits))
    sim = Simulator("mqvector", int(n_qubits))
    sim.apply_circuit(bound)
    energy = float(np.real(sim.get_expectation(ham_mq)))
    psi = np.asarray(sim.get_qs(), dtype=complex).reshape(-1)
    psi /= np.linalg.norm(psi)
    return energy, psi, bound


def evaluate_noisy_circuit(
    ham_mq: Hamiltonian,
    optimized_circuit: Circuit,
    psi_exact: np.ndarray,
    p2: float,
    p1_ratio: float = 0.1,
    n_qubits: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Return noisy energy, exact-state fidelity and purity using mqmatrix.

    Fidelity is <psi_exact|rho_noisy|psi_exact>, where psi_exact is the
    encoded-Hamiltonian ground state. This is distinct from fidelity to the
    ideal variational state.
    """
    noise_model = build_depolarizing_noise_model(p2=p2, p1_ratio=p1_ratio)
    noisy_circuit = noise_model(optimized_circuit)

    if n_qubits is None:
        n_qubits = max(int(optimized_circuit.n_qubits), int(ham_mq.n_qubits))
    sim = Simulator("mqmatrix", int(n_qubits))
    sim.apply_circuit(noisy_circuit)
    e_noisy = float(np.real(sim.get_expectation(ham_mq)))

    dim = 2 ** int(n_qubits)
    rho = np.asarray(sim.get_qs(), dtype=complex).reshape(dim, dim)
    fidelity = float(np.real(np.vdot(psi_exact, rho @ psi_exact)))
    purity = float(np.real(np.trace(rho @ rho)))

    # numerical roundoff protection only
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    purity = float(np.clip(purity, 0.0, 1.0))
    return e_noisy, fidelity, purity


# -----------------------------------------------------------------------------
# Batch robustness benchmark
# -----------------------------------------------------------------------------

def benchmark_ansatz_noise(
    qubit_hamiltonian_record: Sequence[Any],
    ansatz_records: Mapping[str, Mapping[str, Sequence[Any]]],
    p2_list: Optional[Sequence[float]] = None,
    x_values: Optional[Sequence[float]] = None,
    p1_ratio: float = 0.1,
    output_file: str = "ansatz_noise_benchmark.csv",
    circuit_file: str = "ansatz_circuit_metrics.csv",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark all Hamiltonians/ansatzes/noise strengths and checkpoint CSVs.

    Observables include absolute deviation from encoded exact energy,
    noise-induced energy deviation, exact-state fidelity, and purity.
    The two output CSVs contain one row per noise point / circuit respectively.

    ``ansatz_records`` format::

        {
            "CD-full": {"circuits": cd_circuit_record_full, "params": cd_pr},
            "CD-three-body": {"circuits": circuit_cd_record, "params": pr_record},
            "SAE-UCCSD": {"circuits": uccsd_circuit_record, "params": uccsd_pr_record},
            "HEA": {"circuits": hea_circuit_record, "params": hea_pr_record},
        }
    """
    if p2_list is None:
        p2_list = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    p2_list = np.asarray(p2_list, dtype=float)

    n_ham = len(qubit_hamiltonian_record)
    if x_values is None:
        x_values = np.arange(n_ham)
    if len(x_values) != n_ham:
        raise ValueError("len(x_values) must equal len(qubit_hamiltonian_record).")

    for name, data in ansatz_records.items():
        if len(data["circuits"]) != n_ham or len(data["params"]) != n_ham:
            raise ValueError(f"{name}: circuits/params length does not match Hamiltonian record.")

    noise_rows: List[Dict[str, Any]] = []
    circuit_rows: List[Dict[str, Any]] = []

    for ham_idx, qham in enumerate(qubit_hamiltonian_record):
        e_exact, psi_exact = get_exact_ground_state(qham)
        ham_mq = Hamiltonian(generate_hamiltonian(qham))

        if verbose:
            print(f"\n[Noise] Hamiltonian {ham_idx + 1}/{n_ham}, x={x_values[ham_idx]}, E0={e_exact:.12f}")

        for name, data in ansatz_records.items():
            circuit = data["circuits"][ham_idx]
            pr = data["params"][ham_idx]
            metrics = get_circuit_metrics(circuit)

            e_ideal, psi_ideal, bound = evaluate_ideal_circuit(ham_mq, circuit, pr)
            fidelity_ideal = float(abs(np.vdot(psi_exact, psi_ideal)) ** 2)
            dE_ideal = float(abs(e_ideal - e_exact))

            circuit_rows.append({
                "ham_idx": ham_idx,
                "x": x_values[ham_idx],
                "ansatz": name,
                "E_exact": e_exact,
                "E_ideal": e_ideal,
                "delta_E_ideal": dE_ideal,
                "fidelity_ideal": fidelity_ideal,
                **metrics,
            })

            for p2 in p2_list:
                p1 = p1_ratio * float(p2)
                e_noisy, fidelity, purity = evaluate_noisy_circuit(
                    ham_mq=ham_mq,
                    optimized_circuit=bound,
                    psi_exact=psi_exact,
                    p2=float(p2),
                    p1_ratio=p1_ratio,
                    n_qubits=int(ham_mq.n_qubits),
                )

                noise_rows.append({
                    "ham_idx": ham_idx,
                    "x": x_values[ham_idx],
                    "ansatz": name,
                    "p1": p1,
                    "p2": float(p2),
                    "E_exact": e_exact,
                    "E_ideal": e_ideal,
                    "E_noisy": e_noisy,
                    "delta_E_exact": abs(e_noisy - e_exact),
                    "delta_E_ideal": dE_ideal,
                    "delta_E_noise": abs(e_noisy - e_ideal),
                    "fidelity": fidelity,
                    "purity": purity,
                    **metrics,
                })

                if verbose:
                    print(
                        f"  {name:16s} p2={p2:.1e}: "
                        f"dE_exact={abs(e_noisy-e_exact):.3e}, "
                        f"dE_noise={abs(e_noisy-e_ideal):.3e}, "
                        f"F={fidelity:.6f}, purity={purity:.6f}, CX={metrics['n_cx']}"
                    )

        # checkpoint after every Hamiltonian
        pd.DataFrame(noise_rows).to_csv(output_file, index=False)
        pd.DataFrame(circuit_rows).to_csv(circuit_file, index=False)

    df_noise = pd.DataFrame(noise_rows)
    df_circuit = pd.DataFrame(circuit_rows)
    return df_noise, df_circuit


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------

def make_sae_hf_record(
    atom_record: Sequence[Any],
    basis: str,
    CAS: Sequence[int],
    charge: int = 0,
    spin: int = 0,
) -> List[Circuit]:
    """Construct the same symmetry-encoded HF reference state for HEA/UCCSD."""
    from quantumsymmetry import Encoding

    record = []
    for atom in atom_record:
        enc = Encoding(
            atom=atom,
            basis=basis,
            charge=charge,
            spin=spin,
            CAS=list(CAS),
            output_format="qiskit",
        )
        record.append(qiskit_to_mq_circuit(enc.HF_circuit))
    return record


def make_ansatz_records(
    cd_full_circuits: Sequence[Circuit],
    cd_full_params: Sequence[Mapping[str, float]],
    cd_truncated_circuits: Sequence[Circuit],
    cd_truncated_params: Sequence[Mapping[str, float]],
    uccsd_circuits: Optional[Sequence[Circuit]] = None,
    uccsd_params: Optional[Sequence[Mapping[str, float]]] = None,
    hea_circuits: Optional[Sequence[Circuit]] = None,
    hea_params: Optional[Sequence[Mapping[str, float]]] = None,
    adaptive_cd_circuits: Optional[Sequence[Circuit]] = None,
    adaptive_cd_params: Optional[Sequence[Mapping[str, float]]] = None,
    truncated_label: str = "CD-three-body",
    adaptive_cd_label: str = "Adaptive-CD",
) -> Dict[str, Dict[str, Sequence[Any]]]:
    """Create the dictionary expected by :func:`benchmark_ansatz_noise`."""
    out: Dict[str, Dict[str, Sequence[Any]]] = {
        "CD-full": {"circuits": cd_full_circuits, "params": cd_full_params},
        truncated_label: {"circuits": cd_truncated_circuits, "params": cd_truncated_params},
    }
    if uccsd_circuits is not None and uccsd_params is not None:
        out["SAE-UCCSD"] = {"circuits": uccsd_circuits, "params": uccsd_params}
    if hea_circuits is not None and hea_params is not None:
        out["HEA"] = {"circuits": hea_circuits, "params": hea_params}
    if adaptive_cd_circuits is not None and adaptive_cd_params is not None:
        out[adaptive_cd_label] = {
            "circuits": adaptive_cd_circuits,
            "params": adaptive_cd_params,
        }
    return out


__all__ = [
    "generate_hamiltonian",
    "generate_hamitonian",
    "qiskit_pauli_to_mq",
    "qiskit_to_mq_circuit",
    "get_exact_ground_state",
    "get_circuit_metrics",
    "optimize_mq_vqe",
    "active_space_particles",
    "get_sae_uccsd_generators",
    "build_sae_uccsd_mq",
    "batch_sae_uccsd",
    "choose_hea_depth",
    "build_hea",
    "batch_hea",
    "normalize_cd_pool",
    "build_cd_pool",
    "adaptive_cd_pool_gradients",
    "adaptive_cd_vqe",
    "batch_adaptive_cd",
    "build_depolarizing_noise_model",
    "evaluate_ideal_circuit",
    "evaluate_noisy_circuit",
    "benchmark_ansatz_noise",
    "make_sae_hf_record",
    "make_ansatz_records",
]
