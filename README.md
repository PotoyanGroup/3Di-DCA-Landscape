# DCA-3Di

Structure-aware protein landscape modeling with 3Di sequences, ProstT5 translation, variational autoencoders, and direct coupling analysis.

Shukla D, Martin J, Morcos F, Potoyan DA. *A Structure-Aware Generative AI Framework for Revealing Functional Relationships in Proteins Families*. bioRxiv, 2025. DOI: [10.1101/2025.09.18.676787](https://doi.org/10.1101/2025.09.18.676787)

## Pipeline Figure

![Figure 1 from the preprint: ProstT5-3Di and VAE-based structural landscape generation pipeline](https://github.com/Divyanshu2132/DCA-3Di/blob/main/assets/pipeline-figure-1.jpg)

## Project Overview

This method builds structure-informed protein landscapes by combining:

1. Multiple sequence alignments (MSAs) for a protein family
2. ProstT5 translation from amino acid sequences to 3Di tokens
3. A VAE that embeds 3Di sequences into a 2D latent space
4. Decoding across a latent grid to generate maximum-likelihood 3Di sequences
5. Mean-field DCA on the same 3Di MSA to obtain couplings and local fields
6. Hamiltonian scoring of decoded sequences to define an energy landscape


## Implementation

- [`run_vae.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/run_vae.py): trains the VAE on a FASTA alignment
- [`model/`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/model): VAE model definition, sampling layer, and FASTA one-hot encoding utilities
- [`dca/`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/dca): mean-field DCA implementation, Hamiltonian scoring, contact-map helpers, and analysis utilities
- [`Analysis.ipynb`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/Analysis.ipynb): notebook for generating and plotting structural landscapes from saved data
- [`translate.sh`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/translate.sh): SLURM wrapper for an external translation script
- [`train.sh`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/train.sh): SLURM wrapper for VAE training

## End-To-End Workflow

The workflow implied by the paper and partially implemented here is:

1. Collect or build a family MSA in amino acid space.
2. Translate the aligned sequences into 3Di tokens with ProstT5.
3. Save the translated 3Di alignment as FASTA.
4. Train the VAE on the 3Di FASTA alignment.
5. Sample latent coordinates on a 2D grid.
6. Decode a maximum-probability 3Di sequence at each grid point.
7. Fit a DCA Potts model on the same 3Di alignment.
8. Score each decoded sequence with the DCA Hamiltonian.
9. Visualize the resulting structural energy landscape and compare clustering, annotation, contacts, or generated sequences.

## Repository Layout

- [`model/model.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/model/model.py): VAE encoder/decoder and training step
- [`model/layers.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/model/layers.py): reparameterization layer
- [`model/generator.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/model/generator.py): FASTA loading and one-hot encoding
- [`dca/dca_class.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/dca/dca_class.py): main DCA interface
- [`dca/dca_functions.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/dca/dca_functions.py): mfDCA internals and Hamiltonian calculations
- [`dca/dca_analysis.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/dca/dca_analysis.py): mapping and plotting helpers for DI/contact analysis
- [`dca/helper_functions.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/dca/helper_functions.py): PFAM cleaning/filtering and contact-map utilities
- [`data/globin/`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/data/globin): example globin landscape data
- [`data/muticlass_protease/`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/data/muticlass_protease): example peptidase data

## VAE Model

The training script is [`run_vae.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/run_vae.py):

```bash
python run_vae.py <input_fasta> <output_model_path> <log_dir_prefix>
```

Example:

```bash
python run_vae.py data/globin/3Di.fasta outputs/globin.keras logs/
```

The VAE maps each flattened one-hot encoded sequence into a 2D latent space and reconstructs per-position token probabilities with a softmax decoder.

## DCA Utilities

The DCA entrypoint is the [`dca`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/dca/dca_class.py) class in [`dca/dca_class.py`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/dca/dca_class.py).

Minimal example:

```python
from dca.dca_class import dca

model = dca("data/globin/3Di.fasta", stype="protein")
model.mean_field()

print(model.couplings.shape)
print(model.DI[:5])

energies, headers = model.compute_Hamiltonian("data/globin/all_grid.fasta")
```

This supports:

- fitting mean-field DCA on an alignment
- extracting DI scores
- loading couplings or local fields
- scoring sequences with the Potts Hamiltonian

## Example Data And Notebook

The included data and notebook match the structural-landscape workflow described in the paper.

- [`data/globin/3Di.fasta`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/data/globin/3Di.fasta): example 3Di alignment
- [`data/globin/all_grid.fasta`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/data/globin/all_grid.fasta): decoded or grid-sampled sequences used for landscape scoring
- [`data/globin/coord.pkl`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/data/globin/coord.pkl): latent coordinate metadata
- [`data/globin/entropy_map.npy`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/data/globin/entropy_map.npy): precomputed decoder entropy map
- [`Analysis.ipynb`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/Analysis.ipynb): notebook for computing Hamiltonians on a grid and visualizing the landscape

Minimal install:

```bash
pip install tensorflow numpy biopython scipy numba matplotlib notebook
```

If you want to reproduce the full paper workflow, you will also need access to ProstT5 and any downstream tools used for generation and validation.

## Batch Scripts

[`train.sh`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/train.sh) and [`translate.sh`](https://github.com/Divyanshu2132/DCA-3Di/blob/main/translate.sh) are cluster-oriented templates, not turnkey launch scripts.

- paths are placeholders
- `train.sh` points to a different repository directory name
- `translate.sh` expects an external translation script

## Citation

If you use this repository, cite the preprint:

```text
Shukla D, Martin J, Morcos F, Potoyan DA.
A Structure-Aware Generative AI Framework for Revealing Functional Relationships in Proteins Families.
bioRxiv (2025).
doi:10.1101/2025.09.18.676787
```
