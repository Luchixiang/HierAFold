# HierAFold

**HierAFold** is a hierarchical inference framework for large biomolecular complex structure prediction, built on top of the [AlphaFold3 / Protenix](https://github.com/bytedance/Protenix) architecture.

Standard AF3-style models predict all chains simultaneously in a single forward pass.  This works well for small complexes but runs out of memory (or accuracy) on large assemblies with hundreds of chains and thousands of tokens.  HierAFold addresses this by decomposing the problem into two stages:

1. **Pairwise pre-screening** — a lightweight prediction is run for every chain pair to estimate inter-chain confidence (iPTM / PAE).
2. **Focused per-chain prediction** — each query chain is predicted together with a context window of the most relevant partner chains/domains selected using spatial proximity and PAE scores.

The two stages are implemented in `protenix/model/HierAFold.py` and are fully compatible with the standard Protenix checkpoint format.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Preparing Input](#preparing-input)
- [Running Inference](#running-inference)
- [Domain Splitting with InterProScan (Optional)](#domain-splitting-with-interproscan-optional)
- [Output Format](#output-format)
- [Configuration Reference](#configuration-reference)
- [Citation](#citation)

---

## How It Works

```
Input JSON (multi-chain complex)
          │
          ▼
┌─────────────────────────────┐
│  Stage 1: Pairwise screening │   (lightweight, consistency mode)
│  – run AF3 on each chain pair│
│  – extract iPTM + PAE        │
└────────────┬────────────────┘
             │  per-pair PAE maps + ranked scores
             ▼
┌──────────────────────────────────────────────────────┐
│  select_context_tokens  (per query chain)             │ 
│  1. domain detection:                                 │
│     a. GFF3 annotation (InterProScan, if available)  │
│     b. PAE-based domain splitting (fallback)         │
│  2. PAE-guided token selection based on distance an	 │
│		  and PAE                                          │
└────────────┬─────────────────────────────────────────┘
             │  focused context window (≤ 1000 tokens)
             ▼
┌─────────────────────────────────────┐
│  Stage 2: Per-chain full prediction │   (full AF3 forward pass)
│  – Kabsch alignment onto assembly   │
└─────────────────────────────────────┘
             │
             ▼
    Assembled complex coordinates
```

---

## Installation

HierAFold uses the same environment as Protenix.

```bash
# Clone the repository
git clone https://github.com/your-org/HierAFold.git
cd HierAFold

# Install dependencies
pip install -e .
```

For GPU-accelerated kernels (recommended for large complexes):

```bash
# Optional: fast LayerNorm kernel
export LAYERNORM_TYPE=fast_layernorm

# Optional: triangle attention / multiplicative kernels
# See docs/kernels.md for setup
```

---

## Preparing Input

HierAFold uses the same JSON input format as Protenix.  Each entry describes the full complex with sequences, optional MSA paths, and optional constraints.

A minimal example for a two-chain protein complex:

```json
[
  {
    "name": "my_complex",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
          "count": 1
        }
      },
      {
        "proteinChain": {
          "sequence": "GSHMTGGQQMGRGSEFENLYFQGHMHHHHHHHHGSGENLYFQG",
          "count": 1
        }
      }
    ]
  }
]
```

See [`examples/example.json`](examples/example.json) for a full example with MSA paths.

### Convert PDB/CIF to Input JSON

```bash
# Generate CCD cache (first time only)
python scripts/gen_ccd_cache.py -c release_data/ccd_cache/ -n 4

# Convert a PDB file
protenix tojson --input examples/7pzb.pdb --out_dir ./output

# Convert a CIF file
protenix tojson --input examples/7pzb.cif --out_dir ./output
```

### Prepare MSA (Optional but Recommended)

```bash
# Search MSA from JSON input
protenix msa --input examples/example_without_msa.json --out_dir ./output

# Using ColabFold server
export MMSEQS_SERVICE_HOST_URL=https://api.colabfold.com
protenix msa --input examples/example_without_msa.json --out_dir ./output --msa_server_mode colabfold
```

---

## Running Inference

### Via Bash Script (Recommended)

```bash
bash inference_demo.sh
```

### Via Command Line

```bash
python3 runner/inference.py \
    --model_name protenix_base_default_v0.5.0 \
    --seeds 101 \
    --dump_dir ./output \
    --input_json_path examples/example.json \
    --model.N_cycle 10 \
    --sample_diffusion.N_sample 5 \
    --sample_diffusion.N_step 200 \
    --triangle_attention triattention \
    --triangle_multiplicative cuequivariance
```

**What happens at startup:**

1. HierAFold checks whether `nextflow` is available on `PATH`.
2. **If Nextflow is found** — InterProScan is run automatically for every protein chain in the input JSON to produce GFF3 domain annotations (see [Domain Splitting](#domain-splitting-with-interproscan-optional)).  GFF3 files are saved to `<dump_dir>/gff3_domains/`.
3. **If Nextflow is not found** — domain boundaries are estimated from inter-chain PAE maps produced during the pairwise screening stage.  No manual setup is required.

---

## Domain Splitting with InterProScan (Optional)

Accurate domain boundaries improve context selection, especially for large multi-domain proteins.  HierAFold can use [InterProScan](https://www.ebi.ac.uk/interpro/about/interproscan/) (via the CATH-Gene3D application) to obtain sequence-based domain annotations.

### Requirements

| Requirement | Version | Notes |
|---|---|---|
| [Nextflow](https://github.com/ebi-pf-team/interproscan6) | ≥ 23.x | Must be on `PATH` |
| InterProScan data directory | — | Set via `--interproscan_datadir` |

### Install Nextflow

```bash
# Using the official installer
curl -s https://get.nextflow.io | bash
mv nextflow /usr/local/bin/

# Verify
nextflow -version
```

### Run with Domain Splitting

Once Nextflow is on `PATH`, no additional flags are needed — HierAFold detects it automatically:

```bash
python3 runner/inference.py \
    --model_name protenix_base_default_v0.5.0 \
    --seeds 101 \
    --dump_dir ./output \
    --input_json_path examples/example.json \
    --model.N_cycle 10 \
    --sample_diffusion.N_sample 5 \
    --sample_diffusion.N_step 200 \
    --interproscan_datadir /path/to/interproscan/data
```

On startup you will see log output similar to:

```
Nextflow detected — running InterProScan domain annotation.
Running InterProScan for my_complex chain 0 (length 330)
GFF3 saved: ./output/gff3_domains/my_complex_seq_0.fasta.gff3
GFF3 domain annotations available at: ./output/gff3_domains
```

### Domain Splitting Priority

For each context chain, domain boundaries are determined in the following order:

```
1. GFF3 annotation  ──── available?  ──► use GFF3 domains
        │                   no
        ▼
2. PAE-based split  ──── intra-chain PAE > threshold? ──► split with get_domain_splits()
        │                   no
        ▼
3. Single domain (no split)
```

### GFF3 File Naming Convention

HierAFold expects GFF3 files to follow this naming pattern:

```
<sample_name>_seq_<chain_index>.fasta.gff3
```

Where `chain_index` is the 0-based index of the chain in the `sequences` array of the input JSON.  If you have pre-computed GFF3 files from a previous run, point `--interproscan_datadir` to their directory and ensure the names match.

---

## Output Format

Results are saved to `--dump_dir` in the same format as Protenix:

```
output/
├── gff3_domains/             # GFF3 domain annotations (if Nextflow was used)
│   ├── _fasta_tmp/           # Temporary FASTA files (can be deleted)
│   └── <sample>_seq_N.fasta.gff3
├── ERR/                      # Error logs (if any samples failed)
└── <sample_name>/
    ├── seed_101/
    │   ├── sample_1.cif      # Predicted structure (ranked by confidence)
    │   ├── sample_2.cif
    │   └── ...
    └── ...
```

---

## Configuration Reference

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `protenix_base_default_v0.5.0` | Model checkpoint name |
| `--input_json_path` | — | Path to input JSON file or directory |
| `--dump_dir` | `./output` | Directory for saving predictions |
| `--seeds` | `101` | Comma-separated random seeds |
| `--model.N_cycle` | `10` | Number of recycling cycles |
| `--sample_diffusion.N_sample` | `5` | Diffusion samples per seed |
| `--sample_diffusion.N_step` | `200` | Diffusion steps |
| `--dtype` | `bf16` | Precision: `bf16` or `fp32` |
| `--interproscan_datadir` | `/data2/cxlu` | Path to InterProScan data directory |
| `--triangle_attention` | `triattention` | Kernel: `triattention`, `deepspeed`, `torch` |
| `--triangle_multiplicative` | `cuequivariance` | Kernel: `cuequivariance`, `torch` |

---

## Citation

If you use HierAFold in your research, please cite:

```bibtex
@article{lu2026efficient,
  title={Efficient Prediction of Large Protein Complexes via Subunit-Guided Hierarchical Refinement},
  author={Lu, Chixiang and Zhong, Yunhua and Liang, Shikang and Qi, Xiaojuan and Jiang, Haibo},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

HierAFold is built on Protenix — please also cite:

```bibtex
@article{bytedance2025protenix,
  title     = {Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  author    = {ByteDance AML AI4Science Team and Chen, Xinshi and others},
  year      = {2025},
  journal   = {bioRxiv},
  doi       = {10.1101/2025.01.08.631967},
}

@article{abramson2024accurate,
  title     = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  author    = {Abramson, Josh and Adler, Jonas and others},
  journal   = {Nature},
  volume    = {630},
  pages     = {493--500},
  year      = {2024},
}
```

---

## License

Released under the [Apache 2.0 License](LICENSE).

