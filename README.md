# L-SING
Learning local neighborhoods of non-Gaussian graphical models  
**Accepted at AAAI'25**

Arxiv: https://arxiv.org/abs/2503.13899

AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/34059/36214

## Overview
L-SING (Localized Sparsity Identification for Non-Gaussian Distributions) solves a graph recovery problem: given $n$ i.i.d. from an (unspecified and possibly non-Gaussian) probability distribution, L-SING recovers the local neighborhood structure (local Markov properties) of each variable in the graph.

## Authors
Sarah Liaw, Rebecca Morrison, Youssef Marzouk, Ricardo Baptista  
Correspondence to: [sliaw@caltech.edu](mailto:sliaw@caltech.edu)

---

## Experiments
This repo includes the codebase and experiment scripts used for our AAAI'25 submission. Note that unit tests are still being worked on right now.

We conducted three primary experiments to evaluate the performance of L-SING:

1. **Butterfly Distribution (Non-Gaussian)**  
   - A synthetic experiment to show L-SING's ability to learn complex distributions with localized dependencies. We explain how the butterfly distribution is generated in the arxiv submission. 

2. **Gaussian Distribution**
   - Demonstrates L-SING's compatibility with Gaussian distributions for benchmarking purposes.

3. **Ovarian Cancer Dataset**  
   - From the `curatedOvarianPackage` in R.
   - The dataset was pre-processed following the methodology outlined by Shutta et al. (2022) for direct comparison between GLASSO and localized L-SING methods.

---

## Usage

This repo includes examples of differently parameterized UMNNs to replicate results described in the AAAI'25 submission. For detailed experiment configurations (e.g., regularization values, specific UMNN architectures) in our paper, refer to the **technical appendix** in arxiv.


### Running an Experiment
To execute an experiment, follow these steps:

1. **Ensure Configuration Files Are Correct**:
   - Modify the paths in the `config.yaml` file to point to the correct dataset files (`training_file`, `validation_file`, `testing_file`) and desired output directory (`results_path`).
   - Each experiment's script (e.g., `run_butterfly.py`) reads the paths directly from `config.yaml`.

2. **Run the Experiment Script**:
   - Navigate to the main project directory.
   - Run the experiment script using:
     ```bash
     python -m experiments.butterfly.run_butterfly
     ```
   - Replace `butterfly` with the appropriate experiment name for other experiments (e.g., `gaussian`, `ovarian`).

3. **Results**:
   - Results, including the precision matrix, model files, and plots, will be saved in a uniquely indexed folder within the `results/` directory (e.g., `results/BF0/` for the first experiment run).
   - The experiment configuration used for the run is also logged in `log.txt` within the results folder.

---

### Changing Configurations
The `config.yaml` file allows you to customize parameters for each experiment:

- **Dataset Paths**:
  - Update `training_file`, `validation_file`, and `testing_file` to point to your generated `.txt` or `.csv` dataset files.

- **UMNN Parameters**:
  - Adjust `hidden_layers`, `num_steps`, and other model-related settings as wanted.

- **Regularization and Training**:
  - Change/Update `regularizations`, `learning_rate`, and `max_epochs` for alternative experimental setups.

---

### Experiment Logs
After running an experiment, the corresponding `log.txt` file in the results directory will include:

- The configuration used for the run.
- Paths to the saved precision matrix, precision matrix, and models.

---

## Citations
If you use this code, please cite the following references:

1. **Wehenkel, A., & Louppe, G. (2019).**  
   *Unconstrained Monotonic Neural Networks.*  
   In Wallach, H.; Larochelle, H.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E.; and Garnett, R., eds., *Advances in Neural Information Processing Systems,* Vol. 32. Curran Associates, Inc.

2. **Shutta, K. H., Vito, R. D., Scholtens, D. M., & Balasubramanian, R. (2022).**  
   *Gaussian Graphical Models with Applications to Omics Analyses.*  
   *Statistics in Medicine,* 41(25), 5150–5187.

---

## Acknowledgments
The UMNN model implementation in this repository is adapted from the official source:  
[https://github.com/AWehenkel/UMNN](https://github.com/AWehenkel/UMNN).

For further inquiries or clarifications, please refer to our **technical appendix** or contact the authors.

## Monotonic KAN & Scale-Modulated KAN Transport Maps

This repository also contains the implementation and research manuscript for **Monotonic Kolmogorov--Arnold Network (KAN) Transport Maps** and **Scale-Modulated KANs (SM-KANs)** for L-SING:

+- **Theoretical Foundations & Manuscript:** `theory/kan_transport.pdf` (LaTeX source: `theory/kan_transport.tex`).
+- **Core Package:** `models/monotonic_kan/` (`kan_map.py`, `layer.py`, `basis.py`).
+- **Empirical Report:** `benchmarks/kan_benchmark_results.md`.

### Key Advantages:
1. **Elimination of Numerical Quadrature:** Replaces 50-node Clenshaw--Curtis numerical integration in UMNNs with closed-form B-spline primitives, delivering $84\times$ to $299\times$ faster generalized precision extraction ($\widehat{\Omega}_{jk}$).
2. **Certified Monotonicity:** Global target monotonicity $\partial_k S^k(x) \ge \epsilon > 0$ across all $\mathbb{R}^d$ certified by construction via cumulative softplus reparameterization of discrete control points and $C^1$ linear tail extrapolation.
3. **Scale Modulation & Hessian Cancellation:** Multiplicative scale modulation $E(x_{-k}) \cdot S^k_{\text{core}}(x) + \alpha(x_{-k})$ resolves the algebraic polarization obstruction on heteroscedastic couplings while proving that scale and offset modulation introduce zero third-order curvature into the precision log-Jacobian.

### Running Unit Tests & Verification:
```bash
# Run all 39 unit tests
PYTHONPATH=.:src pytest tests/

# Run the analytical theory verification suite
python verify_kan_theory.py
```

### Running KAN Benchmarks:
```bash
# Run comprehensive benchmark suite (Gaussian, Butterfly, high-dimensional scaling)
python run_kan_experiments.py

# Benchmark scripts are located in benchmarks/
python benchmarks/run_scale_modulated_butterfly.py
python benchmarks/run_mixed_coupling_benchmark.py
python benchmarks/run_ovarian_analysis.py
```

---

## Citations
If you use L-SING or the Monotonic KAN transport map extensions in an academic paper, please cite:

```bibtex
@misc{liaw2025learninglocalneighborhoodsnongaussian,
      title={Learning local neighborhoods of non-Gaussian graphical models: A measure transport approach}, 
      author={Sarah Liaw and Rebecca Morrison and Youssef Marzouk and Ricardo Baptista},
      year={2025},
      eprint={2503.13899},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.13899}, 
}
```
