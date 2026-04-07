# HierAFold

**HierAFold** is a hierarchical inference framework for large biomolecular complex structure prediction, built on top of the [AlphaFold3 / Protenix](https://github.com/bytedance/Protenix) architecture.

Standard AF3-style models predict all chains simultaneously in a single forward pass.  This works well for small complexes but runs out of memory (or accuracy) on large assemblies with hundreds of chains and thousands of tokens.  HierAFold addresses this by decomposing the problem into two stages:

1. **Pairwise pre-screening** — a lightweight prediction is run for every chain pair to estimate inter-chain confidence (iPTM / PAE).
2. **Focused per-chain prediction** — each query chain is predicted together with a context window of the most relevant partner chains/domains, selected using spatial proximity and PAE scores.

The two stages are implemented in `protenix/model/HierAFold.py` and are fully compatible with the standard Protenix checkpoint format.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Preparing Input](#preparing-input)
- [Running Inference](#running-inference)
- [Domain Splitting](#domain-splitting)
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
┌─────────────────────────────────────────────────────┐
│  select_context_tokens  (per query chain)            │
│  1. distance-based token selection (≤ 40 Å cutoff)  │
│  2. PAE-based intra-chain domain detection           │
│  3. PAE-guided additional token selection            │
└────────────┬────────────────────────────────────────┘
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

### Domain Detection from PAE

When no external domain annotations are available, HierAFold detects domain boundaries automatically from the intra-chain PAE map produced during Stage 1:

```
For each context chain:
  if intra-chain PAE.max() > threshold:
      ──► split into domains with get_domain_splits(PAE)
  else:
      ──► treat as a single domain
```

Domains with strong predicted interactions (PAE < 3 Å) or that are within 5 Å of the query chain are always included in the context window.  Among remaining candidate domains only the single nearest one is kept, preventing the context from becoming too large.

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

HierAFold will automatically use PAE-based domain splitting at runtime.  No additional setup is required.

**Example log output:**

```
Loading data from
./examples/example.json
Nextflow not found on PATH — skipping GFF3 domain annotation.
Domain boundaries will be estimated from PAE maps during inference.
[Rank 0 (1/1)] my_complex: N_asym 2, N_token 374, N_atom 2891, N_msa 512
Pairwise screening: chain 1 vs chain 2
Chain 2: 2 domain(s) detected
[Rank 0] my_complex succeeded. Model forward time: 48.3s.
Results saved to ./output
```

---

## Domain Splitting

HierAFold performs automatic domain detection from PAE maps produced during the pairwise screening stage.  No external tools are needed.

### How PAE-Based Domain Splitting Works

For each context chain, the intra-chain PAE block from Stage 1 is used to detect structural domain boundaries:

1. The discretised PAE matrix is computed for the chain.
2. If the maximum intra-chain PAE value exceeds a threshold (bin 10), `get_domain_splits()` segments the chain into domains.
3. Otherwise the chain is treated as a single domain.
4. For each detected domain the cross-chain PAE (query ↔ domain) is evaluated:
   - **PAE < 3** → strong predicted interaction → domain always included
   - **Distance < 5 Å** → spatially very close → domain always included
   - **Distance 5–20 Å** → ambiguous → only the nearest candidate domain is included
   - **Distance > 20 Å** → domain excluded from context window

This selection is computed independently for each query chain, so every chain gets its own focused context.

### Using Pre-Computed GFF3 Domain Annotations (Advanced)

If you have already run InterProScan separately and have GFF3 domain annotation files, you can point HierAFold to the directory containing them:

```bash
python3 runner/inference.py \
    --model_name protenix_base_default_v0.5.0 \
    --seeds 101 \
    --dump_dir ./output \
    --input_json_path examples/example.json \
    --interproscan_datadir /path/to/gff3_files
```

GFF3 files must follow this naming convention:

```
<sample_name>_seq_<chain_index>.fasta.gff3
```

Where `chain_index` is the 0-based index of the chain in the `sequences` array of the input JSON.  When a matching GFF3 file is found, it takes precedence over PAE-based splitting for that chain.

---

## Output Format

Results are saved to `--dump_dir` in the same format as Protenix:

```
output/
├── ERR/                      # Error logs (if any samples failed)
└── <sample_name>/
    ├── seed_101/
    │   ├── sample_1.cif      # Predicted structure (ranked by confidence)
    │   ├── sample_2.cif
    │   └── ...
    └── ...
```

Each `.cif` file contains:
- Predicted atom coordinates for all chains, assembled into the full complex
- Per-atom pLDDT scores
- PAE and pDE matrices (in the `summary_confidence` fields)
- Ranking score (iPTM-based)

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
| `--interproscan_datadir` | — | Path to pre-computed GFF3 files (optional) |
| `--triangle_attention` | `triattention` | Kernel: `triattention`, `deepspeed`, `torch` |
| `--triangle_multiplicative` | `cuequivariance` | Kernel: `cuequivariance`, `torch` |

---

## Citation

If you use HierAFold in your research, please cite:

```bibtex
@article{hierafold2025,
  title   = {HierAFold: Hierarchical Inference for Large Biomolecular Complex Structure Prediction},
  author  = {Lu, Chixiang and others},
  journal = {bioRxiv},
  year    = {2025},
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

