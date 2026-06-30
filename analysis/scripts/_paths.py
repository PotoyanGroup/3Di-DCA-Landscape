"""Central registry of data and output paths used by the analysis scripts.

All path constants derive from three configurable roots:

    LAB_DATA_ROOT       Root of the data tree (family MSAs, trained VAEs,
                        landscape grids, structures, MD trajectories).
                        Default: ``$REPO_ROOT/lab_data``.
    REPO_ROOT           Top of this repository (contains ``model/``, ``dca/``).
                        Default: two levels above this file.
    OUT_DIR             Where to write figure and data outputs.
                        Default: ``$REPO_ROOT/analyses/outputs``.

Set these as environment variables before running any script:

    export LAB_DATA_ROOT=/path/to/your/data
    export REPO_ROOT=/path/to/this/repo
    export OUT_DIR=/path/to/output/folder

Expected layout under LAB_DATA_ROOT (per protein family, per representation):

    LAB_DATA_ROOT/
        mdh_aa/                # MDH amino-acid VAE + grid + MSAs
            saved_model.pb + variables/        # AA-VAE SavedModel
            all_grid.pkl                       # latent-grid points
            hamil_mat.npy                      # Hamiltonian on grid
            remove_repetitive.fasta            # aligned MDH AA MSA
            clean_MDH_MSA.fasta                # gap-stripped MSA
            globin_pfam_training_set.fasta     # Globin AA training MSA
        mdh_3di/               # MDH 3Di-VAE + grid
            saved_model.pb + variables/
            all_grid.pkl
            MDH_3Di.fasta
            alphafold_structures/
            coupling_array_3dis.pkl
            coupling_array_seqs.pkl
        mdh_cohort_clusters/   # psy / T-M cohort coordinate files
            psycrophilic_cluster/cluster_coordinates_{AA,3DI}.txt
            thermo_meso_cluster/cluster_coordinates_{AA,3DI}.txt
        mdh_structures/        # AlphaFold PDBs for MDH cohort sequences
        globin_aa/             # Globin AA-VAE + MSA
            saved_model.pb + variables/
            all_grid.pkl
            globin_pfam_training_set.fasta
        globin_3di/            # Globin 3Di + zed coordinates + AF structures
            all_grid.pkl
            zed_coordinates_with_classes.csv
            alphafold_structures/
        trpm_aa/               # TRPM AA-VAE + MSA
            saved_model.pb + variables/
            all_grid.pkl
            trmp8_training_set.fasta
        trpm_3di/              # TRPM 3Di + zed coordinates
            all_grid.pkl
            zed_coordinates_with_classes.csv
        e1_3di/                # E1 glycoprotein 3Di + tree
            saved_model.keras
            E1_glycoprotein_3Di.fasta
            all_grid.fasta
        e2_3di/                # E2 glycoprotein 3Di + tree
            saved_model.keras
            E2_glycoprotein_3Di.fasta
            all_grid.fasta
        md_simulations/        # MD trajectories per organism
            <species_name>/
            shortest_path_thermo_to_meso/
        species_temperatures.csv  # optional functional / phenotypic labels
"""
import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    """Return Path from environment variable ``name`` (or ``default``)."""
    v = os.environ.get(name)
    return Path(v) if v else default


# ============================================================
# Root locations (override via environment)
# ============================================================
_THIS = Path(__file__).resolve()

REPO_ROOT = _env_path("REPO_ROOT", _THIS.parents[2])
LAB_DATA_ROOT = _env_path("LAB_DATA_ROOT", REPO_ROOT / "lab_data")
OUT_DIR = _env_path("OUT_DIR", REPO_ROOT / "analyses" / "outputs")


# ============================================================
# MDH (Fig 2)
# ============================================================
MDH_AA_DIR = LAB_DATA_ROOT / "mdh_aa"
MDH_GRID_PKL = MDH_AA_DIR / "all_grid.pkl"
MDH_HAMIL = MDH_AA_DIR / "hamil_mat.npy"
MDH_MSA_ALIGNED = MDH_AA_DIR / "remove_repetitive.fasta"
MDH_MSA = MDH_AA_DIR / "clean_MDH_MSA.fasta"
MDH_VAE = MDH_AA_DIR
# Backward-compat alias used by older scripts that called this MDH_DIR
MDH_DIR = MDH_AA_DIR

MDH_3DI_DIR = LAB_DATA_ROOT / "mdh_3di"
MDH_3DI_MSA = MDH_3DI_DIR / "MDH_3Di.fasta"
MDH_3DI_VAE = MDH_3DI_DIR
MDH_3DI_GRID_FA = MDH_3DI_DIR / "all_grid.fasta"
MDH_3DI_GRID_PKL = MDH_3DI_DIR / "all_grid.pkl"
MDH_3DI_AF_DIR = MDH_3DI_DIR / "alphafold_structures"
MDH_3DI_COUPLINGS = MDH_3DI_DIR / "coupling_array_3dis.pkl"
MDH_AA_COUPLINGS = MDH_3DI_DIR / "coupling_array_seqs.pkl"

MDH_CLUSTER_DATA = LAB_DATA_ROOT / "mdh_cohort_clusters"
MDH_PDB_DIR = LAB_DATA_ROOT / "mdh_structures"


# ============================================================
# Globin (Fig 3)
# ============================================================
GLOBIN_3DI_DIR = LAB_DATA_ROOT / "globin_3di"
GLOBIN_ZED_CSV = GLOBIN_3DI_DIR / "zed_coordinates_with_classes.csv"
GLOBIN_3DI_GRID_PKL = GLOBIN_3DI_DIR / "all_grid.pkl"
GLOBIN_AF_DIR = GLOBIN_3DI_DIR / "alphafold_structures"

GLOBIN_AA_DIR = LAB_DATA_ROOT / "globin_aa"
GLOBIN_AA_MSA = GLOBIN_AA_DIR / "globin_pfam_training_set.fasta"
GLOBIN_AA_VAE = GLOBIN_AA_DIR
GLOBIN_AA_GRID_PKL = GLOBIN_AA_DIR / "all_grid.pkl"


# ============================================================
# TRPM (Fig 3)
# ============================================================
TRPM_3DI_DIR = LAB_DATA_ROOT / "trpm_3di"
TRPM_ZED_CSV = TRPM_3DI_DIR / "zed_coordinates_with_classes.csv"
TRPM_3DI_GRID_PKL = TRPM_3DI_DIR / "all_grid.pkl"

TRPM_AA_DIR = LAB_DATA_ROOT / "trpm_aa"
TRPM_AA_MSA = TRPM_AA_DIR / "trmp8_training_set.fasta"
TRPM_AA_VAE = TRPM_AA_DIR
TRPM_AA_GRID_PKL = TRPM_AA_DIR / "all_grid.pkl"


# ============================================================
# Flaviviridae E1 / E2
# ============================================================
E1_DIR = LAB_DATA_ROOT / "e1_3di"
E1_3DI_MSA = E1_DIR / "E1_glycoprotein_3Di.fasta"
E1_VAE = E1_DIR / "saved_model.keras"
E1_GRID_FA = E1_DIR / "all_grid.fasta"
E1_AA_DIR = LAB_DATA_ROOT / "e1_aa"
E1_AA_VAE = E1_AA_DIR / "saved_model.keras"

E2_DIR = LAB_DATA_ROOT / "e2_3di"
E2_3DI_MSA = E2_DIR / "E2_glycoprotein_3Di.fasta"
E2_VAE = E2_DIR / "saved_model.keras"
E2_GRID_FA = E2_DIR / "all_grid.fasta"
E2_AA_DIR = LAB_DATA_ROOT / "e2_aa"
E2_AA_VAE = E2_AA_DIR / "saved_model.keras"


# ============================================================
# MD simulations / functional labels
# ============================================================
MD_DIR = LAB_DATA_ROOT / "md_simulations"
MD_PATH_DIR = MD_DIR / "shortest_path_thermo_to_meso"
TEMPERATURE_CSV = LAB_DATA_ROOT / "species_temperatures.csv"

# Per-species MD trajectory directories (used by dynamics_similarity.py).
# Customize this dict to point at the trajectory roots you have available.
MD_SPECIES = {
    "Adamussium_colbecki":       MD_DIR / "Adamussium_colbecki",       # polar / psychrophilic
    "Echinolittorinamalaccana":  MD_DIR / "Echinolittorinamalaccana",  # heat-adapted
    "Mytiluscalifornianus":      MD_DIR / "Mytiluscalifornianus",
    "Neritayoldii":              MD_DIR / "Neritayoldii",
    "Nipponacmearadula":         MD_DIR / "Nipponacmearadula",
}


# ============================================================
# Outputs
# ============================================================
OUT = OUT_DIR                          # alias used by some scripts
OUT_FIG = OUT_DIR / "figures"
OUT_DATA = OUT_DIR / "data"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)
