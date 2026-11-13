# Resource-Efficient Counterdiabatic VQE in Symmetry-Reduced Active Spaces

This repository contains the source code and numerical implementations associated with the research article:

**"Resource-Efficient Counterdiabatic VQE in Symmetry-Reduced Active Spaces"**

**Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(to be updated)*

## Overview

This work presents a resource-efficient framework for quantum molecular ground-state simulations by integrating molecular symmetry encoding, complete active-space (CAS) selection, and counterdiabatic-inspired variational quantum eigensolvers (VQE).

The framework combines three key components:

1. **Symmetry-reduced Hamiltonian encoding:** Exploiting molecular point-group symmetries to reduce the number of qubits required to represent electronic Hamiltonians.
2. **Active-space selection:** Employing chemically motivated CAS approximations to reduce computational resources while retaining essential electronic correlations.
3. **Counterdiabatic-inspired variational ansätze:** Constructing compact quantum circuits using nested-commutator expansions and adaptive operator selection.

Numerical benchmarks are performed on representative molecular systems, including benzene (C6H6), ethylene (C2H4), and molecular nitrogen (N2).

The repository provides implementations for molecular Hamiltonian construction, symmetry-adapted encoding, variational optimization, and quantum circuit analysis.

## Software Dependencies

The project is implemented in Python and uses the following scientific computing and quantum simulation libraries.

| Software | Version |
|---|---|
| Python | To be specified |
| Qiskit | To be specified |
| Qiskit Aer | To be specified |
| Qiskit Algorithms | To be specified |
| Qiskit IBM Runtime | To be specified |
| PySCF | To be specified |
| QuantumSymmetry | To be specified |
| NumPy | To be specified |
| SciPy | To be specified |
| Matplotlib | To be specified |
| pandas | To be specified |
| OpenFermion | To be specified |
| OpenFermion-PySCF | To be specified |

The molecular symmetry encoding is based on the open-source [QuantumSymmetry](https://github.com/dariopicozzi/quantumsymmetry) package.

Exact package versions and environment specifications will be provided for reproducibility.

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Citation

If you use this code in your research, please cite the associated paper:

```bibtex
@article{CounterdiabaticVQE2026,
  title = {Resource-Efficient Counterdiabatic VQE
           in Symmetry-Reduced Active Spaces},
  author = {To be updated},
  journal = {To be updated},
  year = {2026},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv}
}
```

The citation information will be updated following publication.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
