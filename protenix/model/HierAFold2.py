import random
import time
from typing import Any, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from copy import deepcopy
from protenix.model import sample_confidence
from protenix.model.generator import (
    InferenceNoiseScheduler,
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_training,
)
from protenix.model.utils import simple_merge_dict_list
from collections import defaultdict
from .hire_util import *
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.utils.logger import get_logger
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator

from .modules.confidence import ConfidenceHead
from .modules.diffusion import DiffusionModule
from .modules.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from .modules.head import DistogramHead
from .modules.pairformer import MSAModule, PairformerStack, TemplateEmbedder
from .modules.primitives import LinearNoBias

logger = get_logger(__name__)


def update_input_feature_dict(input_feature_dict):
    from protenix.model.modules.transformer import rearrange_qk_to_dense_trunk

    with torch.no_grad():
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        # Compute atom pair feature
        d_lm = (
                q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
                q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        input_feature_dict["d_lm"] = d_lm
        input_feature_dict["v_lm"] = v_lm
        input_feature_dict["pad_info"] = pad_info
        return input_feature_dict


class HierAFold(nn.Module):
    """
    HierAFold: Hierarchical inference wrapper around the AlphaFold3 architecture.

    At inference time, each polymer chain is processed independently.  For every
    query chain the model first runs a lightweight pairwise prediction against
    every other chain to estimate the inter-chain confidence (iPTM / ranking
    score).  The confidence scores are used to select a focused context window
    (nearby domains / subunits) for that query chain via ``select_context_tokens``.
    The query chain together with its selected context is then passed through the
    full model to produce the final structure prediction.  Individual chain
    predictions are assembled into a full-complex coordinate tensor using a
    Kabsch-based rigid alignment.

    Training follows the standard AlphaFold3 training loop unchanged
    (``main_train_loop``).
    """

    def __init__(self, configs) -> None:
        super(HierAFold, self).__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        # Some constants
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0

        # Diffusion scheduler
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        esm_configs = configs.get("esm", {})  # This is used in InputFeatureEmbedder
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **configs.model.constraint_embedder
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        configs.model.diffusion_module['transformer']['n_blocks'] = 12
        self.consistency_contact = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    # ---------------------------------------------------------------------- #
    # Token-selection helpers                                                  #
    # ---------------------------------------------------------------------- #

    def select_context_tokens(
            self,
            query_chain_id: int,
            input_feature_dict: dict[str, Any],
            pred_coordinates: torch.Tensor,
            is_resolved: torch.Tensor,
            top_percent: float = 0.5,
            pae: torch.Tensor = None,
            global_token_indices: torch.Tensor = None,
            gff3_dir: Optional[str] = None,
            pdb_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Select context tokens for a query chain based on spatial proximity and
        predicted inter-chain confidence (PAE).

        For the query chain all of its tokens are kept.  For every other polymer
        chain the method:

        1. Selects up to ``top_percent`` of residues closest to the query chain
           (capped at a hard 40 Å distance cutoff).
        2. Determines structural domains for each context chain using one of
           two sources (in priority order):

           a. **GFF3 annotation** (``gff3_dir`` / ``pdb_id``): domain boundaries
              produced by an external tool (e.g. InterPro) are read from a GFF3
              file named ``<pdb_id>_seq_<chain_id>.fasta.gff3``.
           b. **PAE-based detection** (fallback): when no GFF3 annotation is
              available (or it describes only one domain), intra-chain PAE maps
              are used to detect domain boundaries with :func:`get_domain_splits`.

        3. Applies per-domain filtering when multiple domains are found: domains
           with strong predicted interaction (PAE < 3 Å) or within the hard
           distance cutoff are always kept; among the remaining "soft" candidates
           only the single nearest domain is retained.
        4. Additionally includes residues with low cross-chain PAE quantile
           scores (indicating predicted interaction).

        All ligand tokens are always included.

        Args:
            query_chain_id: Chain ID (``asym_id`` value) of the query chain.
            input_feature_dict: Input feature dictionary containing ``asym_id``,
                ``is_ligand``, ``frame_atom_index``, etc.
            pred_coordinates: Predicted atom coordinates from the pairwise
                pre-screening step, shape ``[N_atom, 3]``.
            is_resolved: Boolean mask indicating which atoms have resolved
                coordinates, shape ``[N_atom]``.
            top_percent: Fraction of nearest tokens to keep from each context
                chain (default: 0.5).
            pae: Predicted aligned error logits from the pairwise screening,
                used to refine context token selection.  Shape
                ``[N_token, N_token, n_bins]`` or ``None``.
            global_token_indices: Mapping from local (pair-subset) token indices
                to global token indices in the full complex.  Used to populate
                ``priority_info`` with global indices for downstream clipping.
            gff3_dir: Optional path to a directory containing GFF3 domain
                annotation files (one file per chain, named
                ``<pdb_id>_seq_<chain_id>.fasta.gff3``).  When provided, GFF3
                annotations are used as the primary domain-splitting source.
            pdb_id: PDB identifier used to construct the GFF3 filename.
                Required when ``gff3_dir`` is set.

        Returns:
            selected_tokens: 1-D tensor of local token indices (sorted,
                deduplicated) to include in the context window.
            priority_info: Dict mapping a descriptive key to a
                ``(priority_score, global_tokens, is_query_chain)`` tuple,
                consumed by :meth:`clip_tokens_to_max`.
        """
        # Distance cutoffs (Å) for domain selection
        HARD_DIST_CUTOFF = 5.0  # always keep domain if closer than this
        SOFT_DIST_CUTOFF = 20.0  # consider domain if closer than this

        device = pred_coordinates.device
        asym_id = input_feature_dict["asym_id"]
        is_ligand = input_feature_dict["is_ligand"].bool()

        # ------------------------------------------------------------------ #
        # Build per-token helper arrays                                        #
        # ------------------------------------------------------------------ #
        frame_atom_index = input_feature_dict["frame_atom_index"]
        N_token = asym_id.shape[0]
        centre_atom_index: List[int] = []
        is_ligand_token_list: List[bool] = []
        is_resolved_token_list: List[bool] = []

        for i in range(N_token):
            assert frame_atom_index[i][1] != -1, (
                f"Token {i} has no valid centre atom (frame_atom_index[i][1] == -1)."
            )
            center_atom = frame_atom_index[i][1]
            centre_atom_index.append(center_atom)
            is_resolved_token_list.append(is_resolved[center_atom])
            is_ligand_token_list.append(is_ligand[center_atom])

        is_ligand_token = torch.tensor(is_ligand_token_list, device=device).bool()
        is_resolved_token = torch.tensor(is_resolved_token_list, device=device).bool()

        # ------------------------------------------------------------------ #
        # 1. Token centre coordinates                                          #
        # ------------------------------------------------------------------ #
        token_centers = pred_coordinates[centre_atom_index]  # [N_token, 3]

        # ------------------------------------------------------------------ #
        # 2. Query-chain tokens                                                #
        # ------------------------------------------------------------------ #
        query_tokens = torch.where(asym_id == query_chain_id)[0]
        query_resolved_tokens = torch.where(
            (asym_id == query_chain_id) & is_resolved_token
        )[0]
        query_center_coords = token_centers[query_resolved_tokens]  # [N_q, 3]

        # ------------------------------------------------------------------ #
        # 3. Other polymer chain IDs                                           #
        # ------------------------------------------------------------------ #
        all_polymer_chain_ids = torch.unique(asym_id[~is_ligand_token])
        other_chain_ids = all_polymer_chain_ids[all_polymer_chain_ids != query_chain_id]

        selected_tokens: List[torch.Tensor] = [query_tokens]
        priority_info: dict = {}
        subunit_counter = 0

        # Query chain gets highest priority (score = 0)
        priority_info[f"target_{query_chain_id}"] = (
            0.0, global_token_indices[query_tokens], True
        )
        subunit_counter += 1

        # Pre-compute discretised PAE once (reused for intra- and cross-chain)
        pae_discrete = torch.argmax(pae, dim=-1) if pae is not None else None

        # ------------------------------------------------------------------ #
        # 4. Context-token selection for each other chain                     #
        # ------------------------------------------------------------------ #
        for other_id in other_chain_ids.tolist():
            other_resolved_tokens = torch.where(
                (asym_id == other_id) & (~is_ligand_token) & is_resolved_token
            )[0]
            if len(other_resolved_tokens) == 0:
                continue

            # Small chains (≤ 40 residues) are always kept in full
            if len(other_resolved_tokens) <= 40:
                priority_info[f"small_chain_{other_id}"] = (
                    1.0, global_token_indices[other_resolved_tokens], False
                )
                selected_tokens.append(other_resolved_tokens)
                subunit_counter += 1
                continue

            # -- Distance-based pre-selection --------------------------------
            other_centers = token_centers[other_resolved_tokens]  # [N_o, 3]
            dists = torch.cdist(query_center_coords, other_centers)  # [N_q, N_o]
            min_dist_per_other_token, _ = dists.min(dim=0)  # [N_o]
            sorted_min_dist, _ = torch.sort(min_dist_per_other_token)

            n_select = max(
                min(50, len(other_resolved_tokens)),
                int(top_percent * len(other_resolved_tokens)),
            )
            # Hard 40 Å distance cutoff
            cutoff_idx = torch.searchsorted(sorted_min_dist, 40).item()
            n_select = min(n_select, cutoff_idx)
            sorted_order = torch.argsort(min_dist_per_other_token)
            distance_selected_tokens = other_resolved_tokens[sorted_order[:n_select]]

            # -- Domain decomposition ----------------------------------------
            # Priority 1: GFF3 annotation from an external domain-splitting tool
            # Priority 2: PAE-based detection (fallback when GFF3 not available
            #             or gives only one domain, i.e. len(subunits) == 1)
            replicate = 1
            subunits: Optional[List[Tuple[int, int]]] = None

            if gff3_dir is not None and pdb_id is not None and os.path.exists(gff3_dir):
                gff3_filename = f"{pdb_id}_seq_{other_id}.fasta.gff3"
                gff3_path = os.path.join(gff3_dir, gff3_filename)
                sequence_length = len(torch.where(asym_id == other_id)[0])
                subunits = parse_gff3_domains(gff3_path, sequence_length, replicate=replicate)
                logger.info(f"GFF3 domains for chain {other_id}: {subunits}")

            # Fallback: use PAE map to detect domains when GFF3 is absent or
            # returned only a single domain (len == replicate == 1)
            if subunits is None or len(subunits) == replicate:
                if pae_discrete is not None:
                    pae_intra = pae_discrete[other_resolved_tokens][:, other_resolved_tokens]
                    if pae_discrete.max() > 10:
                        # Intra-chain PAE suggests multiple domains
                        domain_boundaries = get_domain_splits(pae_intra.cpu().numpy(),
                                                              10)  # adjust if you want fine domain or coarse domain
                        subunits = [
                            (domain_boundaries[i - 1], domain_boundaries[i])
                            for i in range(1, len(domain_boundaries))
                        ]
                    else:
                        subunits = [(0, other_resolved_tokens.shape[0])]

            if subunits is None:
                subunits = [(0, other_resolved_tokens.shape[0])]

            logger.debug(f"Chain {other_id}: {len(subunits)} domain(s) detected")

            # -- Cross-chain PAE slice (query × other, used below) -----------
            pae_cross = (
                pae_discrete[query_resolved_tokens][:, other_resolved_tokens]
                if pae_discrete is not None
                else None
            )

            # -- PAE-guided additional token selection -----------------------
            context_selected_tokens = distance_selected_tokens
            if pae_cross is not None:
                pae_quantile = torch.quantile(pae_cross.float(), 0.2, dim=0)
                pae_selected = other_resolved_tokens[pae_quantile <= 10]
                # Fall back to a stricter threshold if too many tokens pass
                if pae_selected.shape[0] >= other_resolved_tokens.shape[0] * 0.7:
                    pae_quantile = torch.quantile(pae_cross.float(), 0.5, dim=0)
                    pae_selected = other_resolved_tokens[pae_quantile <= 3]
                context_selected_tokens = torch.cat(
                    (distance_selected_tokens, pae_selected)
                )

            # -- Per-domain filtering (only when multiple domains detected) --
            if len(subunits) > 1:
                domain_tokens_to_keep: List[torch.Tensor] = []
                soft_candidates: List[Tuple] = []  # (dist, tokens, span, pae_score, priority)

                for span in subunits:
                    domain_tokens = other_resolved_tokens[span[0]: span[1]]
                    domain_centers = token_centers[domain_tokens]
                    domain_dists = torch.cdist(query_center_coords, domain_centers)
                    min_domain_dist = domain_dists.min().item()

                    # Compute median cross-chain PAE for this domain
                    pae_domain = pae_cross[:, span[0]: span[1]]
                    pae_median_per_domain_token = torch.quantile(
                        pae_domain.float(), 0.5, dim=0
                    )
                    pae_domain_score = torch.quantile(
                        pae_median_per_domain_token, 0.5, dim=0
                    )

                    priority_score = min_domain_dist + 2.0 * pae_domain_score.item()

                    if pae_domain_score < 3:
                        # Strong predicted interaction — always keep
                        domain_tokens_to_keep.append(domain_tokens)
                        priority_info[
                            f"chain_{other_id}_domain_{subunit_counter}_pae"
                        ] = (priority_score, global_token_indices[domain_tokens], False)
                        subunit_counter += 1
                    elif min_domain_dist < HARD_DIST_CUTOFF:
                        # Spatially very close — always keep
                        domain_tokens_to_keep.append(domain_tokens)
                        priority_info[
                            f"chain_{other_id}_domain_{subunit_counter}_close"
                        ] = (priority_score, global_token_indices[domain_tokens], False)
                        subunit_counter += 1
                    elif min_domain_dist < SOFT_DIST_CUTOFF:
                        # Ambiguous — defer decision
                        soft_candidates.append(
                            (min_domain_dist, domain_tokens, span,
                             pae_domain_score, priority_score)
                        )
                    # Domains beyond SOFT_DIST_CUTOFF are dropped

                # Among soft candidates keep only the single nearest domain
                if soft_candidates:
                    soft_candidates.sort(key=lambda x: x[0])
                    nearest_dist, nearest_tokens, _, _, nearest_priority = soft_candidates[0]
                    domain_tokens_to_keep.append(nearest_tokens)
                    priority_info[
                        f"chain_{other_id}_domain_{subunit_counter}_nearest"
                    ] = (nearest_priority, global_token_indices[nearest_tokens], False)
                    subunit_counter += 1

                context_selected_tokens = torch.cat(domain_tokens_to_keep)
            else:
                # Single domain — record priority for the whole chain
                min_dist = dists.min().item()
                pae_score = (
                    torch.quantile(pae_cross.float(), 0.5).item()
                    if pae_cross is not None
                    else 0.0
                )
                priority_score = min_dist + 2.0 * pae_score
                priority_info[f"chain_{other_id}_full"] = (
                    priority_score,
                    global_token_indices[context_selected_tokens],
                    False,
                )

            selected_tokens.append(context_selected_tokens)

            # Also include unresolved tokens from this chain (no coordinates but
            # still part of the sequence context)
            unresolved_tokens = torch.where(
                (asym_id == other_id) & (~is_ligand_token) & (~is_resolved_token)
            )[0]
            if len(unresolved_tokens) > 0:
                priority_info[f"chain_{other_id}_unresolved"] = (
                    1000.0, global_token_indices[unresolved_tokens], False
                )
                selected_tokens.append(unresolved_tokens)

        # ------------------------------------------------------------------ #
        # 5. Always include all ligand tokens                                  #
        # ------------------------------------------------------------------ #
        ligand_tokens = torch.where(is_ligand_token)[0]
        if len(ligand_tokens) > 0:
            priority_info["ligands"] = (0.5, global_token_indices[ligand_tokens], False)
            selected_tokens.append(ligand_tokens)

        # ------------------------------------------------------------------ #
        # 6. Combine, deduplicate, and sort                                    #
        # ------------------------------------------------------------------ #
        selected_tokens_tensor = torch.cat(selected_tokens).unique()
        selected_tokens_tensor, _ = torch.sort(selected_tokens_tensor)
        return selected_tokens_tensor, priority_info

    def clip_tokens_to_max(
            self,
            all_selected_tokens: torch.Tensor,
            all_priority_info: List[dict],
            max_tokens: int = 4000
    ) -> torch.Tensor:
        """
        Clip tokens to max_tokens based on importance (distance and PAE/PDE).
        Tokens with larger distance or PAE/PDE are clipped first (higher priority_score = clip first).

        Args:
            all_selected_tokens: All selected tokens from all chains
            all_priority_info: List of priority_info dicts from each chain
            max_tokens: Maximum number of tokens to keep

        Returns:
            clipped_tokens: Token indices to keep (up to max_tokens)
        """
        if len(all_selected_tokens) <= max_tokens:
            return all_selected_tokens

        print(f"Clipping: {len(all_selected_tokens)} -> {max_tokens} tokens")

        # Merge all priority info
        merged_subunits = []
        for priority_dict in all_priority_info:
            for subunit_id, (priority_score, tokens, is_target) in priority_dict.items():
                merged_subunits.append((priority_score, tokens, is_target, subunit_id))

        # Sort by priority (lower score = higher priority = keep first)
        merged_subunits.sort(key=lambda x: x[0])

        # Greedily add subunits until we reach max_tokens
        selected_tokens_list = []
        current_count = 0
        device = all_selected_tokens.device
        for priority_score, tokens, is_target, subunit_id in merged_subunits:
            tokens_unique = tokens.unique()

            # Remove already selected tokens to avoid double counting
            if len(selected_tokens_list) > 0:
                already_selected = torch.cat(selected_tokens_list)
                new_tokens = tokens_unique[~torch.isin(tokens_unique, already_selected)]
            else:
                new_tokens = tokens_unique

            new_count = len(new_tokens)

            if current_count + new_count <= max_tokens:
                # Add all tokens from this subunit
                if new_count > 0:
                    selected_tokens_list.append(new_tokens)
                    current_count += new_count
                    print(f"  Added {subunit_id}: {new_count} tokens (total: {current_count})")
            elif is_target:
                # Always include target chain even if it exceeds limit
                if new_count > 0:
                    selected_tokens_list.append(new_tokens)
                    current_count += new_count
                    print(f"  Added {subunit_id} (target): {new_count} tokens (total: {current_count})")
            else:
                # Partially add tokens if there's room
                remaining = max_tokens - current_count
                if remaining > 0:
                    selected_tokens_list.append(new_tokens)
                    current_count += len(new_tokens)
                    print(f"  Partially added {subunit_id}: {remaining}/{new_count} tokens (total: {current_count})")
                else:
                    print(f"  Skipped {subunit_id}: {new_count} tokens (would exceed max)")
                break  # Stop after first partial/skip

        if len(selected_tokens_list) == 0:
            print("Warning: No tokens selected, returning first max_tokens")
            return all_selected_tokens[:max_tokens]

        clipped_tokens = torch.cat(selected_tokens_list).unique()
        clipped_tokens, _ = torch.sort(clipped_tokens)

        print(f"Final clipped tokens: {len(clipped_tokens)}")
        return clipped_tokens

    def create_subset_input_feature_dict(
            self,
            input_feature_dict: dict[str, Any],
            selected_tokens: torch.Tensor,
            is_resolved: torch.Tensor = None,
    ) -> dict[str, Any]:
        """
        Build a new ``input_feature_dict`` containing only the features for
        ``selected_tokens`` (and their corresponding atoms).

        All index-based fields (``atom_to_token_idx``, ``frame_atom_index``,
        etc.) are re-mapped so that they are valid within the subset.

        Args:
            input_feature_dict: Full-complex input feature dictionary.
            selected_tokens: 1-D tensor of token indices to include.
            is_resolved: Optional per-atom boolean mask.  When provided, the
                subset mask is returned alongside the feature dict.

        Returns:
            subset_dict: Feature dictionary restricted to the selected tokens.
            is_resolved (only when the argument is not None): Per-atom resolved
                mask for the subset.
        """
        subset_dict = {}
        device = selected_tokens.device
        selected_tokens = selected_tokens.cpu()
        N_selected = len(selected_tokens)

        # Map original token indices → new contiguous indices [0, N_selected)
        token_mapping = {int(t.item()): i for i, t in enumerate(selected_tokens)}

        # Per-token features
        per_token_keys = [
            "residue_index", "asym_id", "entity_id", "sym_id", "restype",
            "has_frame", "profile", "sym_id_custom", "deletion_mean",
        ]
        for key in per_token_keys:
            if key in input_feature_dict:
                subset_dict[key] = input_feature_dict[key][selected_tokens].clone()
        subset_dict["token_index"] = torch.arange(0, N_selected, device=device)

        # Pair features (token × token)
        for key in ["token_bonds"]:
            if key in input_feature_dict:
                subset_dict[key] = (
                    input_feature_dict[key][selected_tokens][:, selected_tokens]
                    .clone()
                    .cuda()
                )

        # Atom-level index remapping
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"].cpu()
        selected_atoms_mask = torch.isin(atom_to_token_idx, selected_tokens)
        selected_atoms = torch.where(selected_atoms_mask)[0].cpu()

        subset_dict["atom_to_token_idx"] = torch.tensor(
            [token_mapping[int(atom_to_token_idx[a].item())] for a in selected_atoms],
            dtype=atom_to_token_idx.dtype,
            device=device,
        )

        if is_resolved is not None:
            is_resolved = is_resolved[selected_atoms]

        atom_mapping = {int(t.item()): i for i, t in enumerate(selected_atoms)}
        atom_mapping[-1] = -1

        frame_atom_index = input_feature_dict["frame_atom_index"]
        try:
            subset_dict["frame_atom_index"] = torch.tensor(
                [
                    [atom_mapping[int(frame_atom_index[a][b].item())] for b in range(3)]
                    for a in selected_tokens
                ],
                dtype=frame_atom_index.dtype,
                device=device,
            )
        except KeyError:
            pass

        # Per-atom features
        atom_keys = [
            "ref_pos", "ref_mask", "ref_element", "atom_to_tokatom_idx",
            "ref_charge", "ref_atom_name_chars", "ref_space_uid",
            "is_protein", "is_dna", "is_rna", "is_ligand",
            "mol_id", "mol_atom_index", "entity_mol_id",
            "pae_rep_atom_mask", "plddt_m_rep_atom_mask",
            "distogram_rep_atom_mask", "modified_res_mask",
        ]
        for key in atom_keys:
            if key in input_feature_dict:
                subset_dict[key] = input_feature_dict[key][selected_atoms].clone().cuda()

        for key in ["ref_atom_name", "ref_atom_name_chars"]:
            if key in input_feature_dict:
                subset_dict[key] = input_feature_dict[key][selected_atoms.cpu().numpy()].cuda()

        # Pair atom features (atom × atom)
        for key in ["bond_mask"]:
            if key in input_feature_dict:
                subset_dict[key] = (
                    input_feature_dict[key][selected_atoms][:, selected_atoms]
                    .clone()
                    .cuda()
                )

        # Scalar features that are kept as-is
        for key in ["resolution"]:
            if key in input_feature_dict:
                subset_dict[key] = input_feature_dict[key].clone()

        # MSA features
        if "msa" in input_feature_dict:
            subset_dict["msa"] = (
                input_feature_dict["msa"][:, selected_tokens.cpu()].clone().cuda()
            )
            subset_dict["has_deletion"] = (
                input_feature_dict["has_deletion"][:, selected_tokens.cpu()].clone().cuda()
            )
            subset_dict["deletion_value"] = (
                input_feature_dict["deletion_value"][:, selected_tokens.cpu()].clone().cuda()
            )

        if "template_input" in input_feature_dict:
            subset_dict["template_input"] = None

        if "coordinate" in input_feature_dict:
            subset_dict["coordinate"] = (
                input_feature_dict["coordinate"][selected_atoms, :].clone()
            )

        # Generate relative position encodings and atom-pair features for the subset
        subset_dict = self.relative_position_encoding.generate_relp(subset_dict)
        subset_dict = update_input_feature_dict(subset_dict)

        if is_resolved is not None:
            return subset_dict, is_resolved
        return subset_dict

    # ---------------------------------------------------------------------- #
    # Pairformer                                                               #
    # ---------------------------------------------------------------------- #

    def get_pairformer_output(
            self,
            input_feature_dict: dict[str, Any],
            N_cycle: int,
            inplace_safe: bool = False,
            chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        z_constraint = None

        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(
                input_feature_dict["constraint_feature"]
            )

        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        z_init = (
                self.linear_no_bias_zinit1(s_init)[..., None, :]
                + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init = z_init + z_constraint
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                    self.training
                    and (not self.train_confidence_only)
                    and cycle_no == (N_cycle - 1)
            ):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()

        return s_inputs, s, z

    # ---------------------------------------------------------------------- #
    # Diffusion / confidence helpers                                           #
    # ---------------------------------------------------------------------- #

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def sample_consistency(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        from protenix.model.generator import sample_consistency
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_consistency
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args, **kwargs):
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    # ---------------------------------------------------------------------- #
    # Inference loops                                                          #
    # ---------------------------------------------------------------------- #

    def main_inference_loop(
            self,
            input_feature_dict: dict[str, Any],
            label_dict: dict[str, Any],
            N_cycle: int,
            mode: str,
            inplace_safe: bool = True,
            chunk_size: Optional[int] = 4,
            N_model_seed: int = 1,
            symmetric_permutation: SymmetricPermutation = None,
            gff3_dir: Optional[str] = None,
            pdb_id: Optional[str] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Outer inference loop that runs :meth:`_hierarchical_inference_loop`
        for each model seed and combines the results.

        Args:
            input_feature_dict: Input features dictionary.
            label_dict: Label dictionary (used for symmetric permutation in eval mode).
            N_cycle: Number of recycling cycles.
            mode: One of ``'inference'`` or ``'eval'``.
            inplace_safe: Whether to use in-place tensor operations.
            chunk_size: Chunk size for memory-efficient pairformer attention.
            N_model_seed: Number of independent model seeds to run.
            symmetric_permutation: Symmetric permutation object for eval mode.
            gff3_dir: Optional directory containing GFF3 domain annotation files.
                When provided, domain boundaries are read from GFF3 files
                (named ``<pdb_id>_seq_<chain_id>.fasta.gff3``) and used as the
                primary domain-splitting source in :meth:`select_context_tokens`.
            pdb_id: PDB identifier used to locate the correct GFF3 file.

        Returns:
            all_pred_dict: Combined predictions from all seeds.
            all_log_dict: Combined log dictionaries.
            all_time_dict: Combined timing information.
        """
        pred_dicts = []
        log_dicts = []
        time_trackers = []

        for _ in range(N_model_seed):
            N_token = input_feature_dict["asym_id"].shape[0]
            # if N_token < 0:
            #     pred_dict, log_dict, time_tracker = self._single_chain_inference(input_feature_dict=input_feature_dict,
            #                                                                      label_dict=label_dict,
            #                                                                      N_cycle=N_cycle,
            #                                                                      mode=mode,
            #                                                                      inplace_safe=inplace_safe,
            #                                                                      chunk_size=chunk_size,
            #                                                                      symmetric_permutation=symmetric_permutation,
            #                                                                      use_consistency=False)
            # else:
            #     print('using hierarchical inference')
            pred_dict, log_dict, time_tracker = self._hierarchical_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                gff3_dir=gff3_dir,
                pdb_id=pdb_id,
            )
            pred_dicts.append(pred_dict)
            log_dicts.append(log_dict)
            time_trackers.append(time_tracker)

        def _cat(dict_list, key):
            return torch.cat([x[key] for x in dict_list], dim=0)

        def _list_join(dict_list, key):
            return sum([x[key] for x in dict_list], [])

        all_pred_dict = {
            "coordinate": _cat(pred_dicts, "coordinate"),
            "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
            "full_data": _list_join(pred_dicts, "full_data"),
            "plddt": _cat(pred_dicts, "plddt"),
            "pae": _cat(pred_dicts, "pae"),
            "pde": _cat(pred_dicts, "pde"),
            "resolved": _cat(pred_dicts, "resolved"),
        }

        all_log_dict = simple_merge_dict_list(log_dicts)
        all_time_dict = simple_merge_dict_list(time_trackers)
        return all_pred_dict, all_log_dict, all_time_dict

    def _hierarchical_inference_loop(
            self,
            input_feature_dict: dict[str, Any],
            label_dict: dict[str, Any],
            N_cycle: int,
            mode: str,
            inplace_safe: bool = True,
            chunk_size: Optional[int] = 4,
            symmetric_permutation: SymmetricPermutation = None,
            gff3_dir: Optional[str] = None,
            pdb_id: Optional[str] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Single-seed hierarchical inference loop.

        The algorithm proceeds in two stages:

        **Stage 1 — Pairwise pre-screening.**
        For every pair of polymer chains a lightweight prediction is run on the
        chain-pair subset (plus all ligands).  The resulting pairwise PDE and
        ranking score are used to (a) decide whether the two chains interact and
        (b) guide the context-token selection via :meth:`select_context_tokens`.

        **Stage 2 — Per-chain full prediction.**
        Each query chain is predicted together with its selected context window
        (up to ``max_tokens = 1000`` tokens).  Chains are processed in
        descending order of predicted interaction strength so that more
        confident placements serve as anchors for subsequent chains.  Predicted
        coordinates are rigidly aligned (Kabsch) onto the already-assembled
        complex before writing to the output tensor.

        Args:
            input_feature_dict: Full-complex input features.
            label_dict: Label dictionary (passed through to ``_single_chain_inference``).
            N_cycle: Number of recycling cycles.
            mode: One of ``'inference'`` or ``'eval'``.
            inplace_safe: Whether in-place tensor operations are safe.
            chunk_size: Chunk size for memory-efficient attention.
            symmetric_permutation: Symmetric permutation object.
            gff3_dir: Optional directory containing GFF3 domain annotation files.
                Passed through to :meth:`select_context_tokens`.
            pdb_id: PDB identifier used to locate the correct GFF3 file.
                Passed through to :meth:`select_context_tokens`.

        Returns:
            pred_dict: Assembled prediction dictionary with keys ``coordinate``,
                ``plddt``, ``pae``, ``pde``, ``resolved``,
                ``summary_confidence``, ``full_data``.
            log_dict: Empty dict (reserved for future logging).
            time_tracker: Dict with key ``'total'`` containing elapsed seconds.
        """
        t0 = time.time()
        N_sample = self.configs.sample_diffusion["N_sample"]
        device = next(self.parameters()).device
        # dtype = next(self.parameters()).dtype
        dtype = torch.float32

        # ------------------------------------------------------------------ #
        # Resolve per-token ligand flag and build token↔atom index maps       #
        # ------------------------------------------------------------------ #
        asym_id = input_feature_dict["asym_id"]
        input_feature_dict["sym_id_custom"] = asym_id
        is_ligand = input_feature_dict["is_ligand"].bool()
        frame_atom_index = input_feature_dict["frame_atom_index"]
        N_token = asym_id.shape[0]

        is_ligand_per_token = torch.tensor(
            [is_ligand[frame_atom_index[i][1]] for i in range(N_token)],
            device=device,
        ).bool()

        # All atoms are considered resolved (no ground-truth mask at inference)
        is_resolved_full = torch.ones(
            input_feature_dict["is_ligand"].shape[0], dtype=torch.bool, device=device
        )

        atom_to_token_index = input_feature_dict["atom_to_token_idx"].tolist()
        token_to_atom_index: List[List[int]] = [[] for _ in range(N_token)]
        for atom_idx, token_idx in enumerate(atom_to_token_index):
            token_to_atom_index[token_idx].append(atom_idx)

        polymer_chain_ids = torch.unique(asym_id[~is_ligand_per_token]).tolist()
        n_chains = len(polymer_chain_ids)

        # Per-chain accumulators for Stage 1 results
        selected_token_sets: List[List[torch.Tensor]] = [[] for _ in range(n_chains)]
        priority_info_sets: List[List[dict]] = [[] for _ in range(n_chains)]
        interaction_chain_counts: List[int] = [0] * n_chains
        interaction_token_counts: List[int] = [0] * n_chains
        chain_results: List[dict] = []
        # Offload large tensors to CPU when the complex is large to save GPU memory
        if N_token > 2000:
            for key in ["bond_mask", "msa", "has_deletion", "deletion_value",
                        "token_bonds", "ref_pos", "ref_mask", "ref_element",
                        "atom_to_tokatom_idx", "ref_charge", "ref_atom_name_chars",
                        "ref_space_uid", "is_protein", "is_dna", "is_rna", "is_ligand",
                        "mol_id", "mol_atom_index", "entity_mol_id",
                        "pae_rep_atom_mask", "plddt_m_rep_atom_mask",
                        "distogram_rep_atom_mask", "modified_res_mask"]:
                if key in input_feature_dict:
                    input_feature_dict[key] = input_feature_dict[key].cpu()
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------ #
        # Stage 1: Pairwise chain pre-screening                               #
        # ------------------------------------------------------------------ #
        for idx_i in range(n_chains):
            chain_id_i = polymer_chain_ids[idx_i]
            tokens_i = torch.where(asym_id == chain_id_i)[0]

            for idx_j in range(idx_i + 1, n_chains):
                chain_id_j = polymer_chain_ids[idx_j]
                tokens_j = torch.where(asym_id == chain_id_j)[0]

                logger.info(
                    f"Pairwise screening: chain {chain_id_i} vs chain {chain_id_j}"
                )

                # Build chain-pair subset (two chains + all ligands)
                pair_token_indices = torch.where(
                    (asym_id == chain_id_i)
                    | (asym_id == chain_id_j)
                    | is_ligand_per_token
                )[0]
                pair_feat, is_resolved_pair = self.create_subset_input_feature_dict(
                    input_feature_dict, pair_token_indices, is_resolved_full
                )


                # Run lightweight pairwise prediction (consistency mode, fewer steps)
                pair_pred, _, _ = self._single_chain_inference(
                    input_feature_dict=pair_feat,
                    label_dict=label_dict,
                    N_cycle=N_cycle,
                    mode=mode,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    symmetric_permutation=symmetric_permutation,
                    use_consistency=True,
                    chain_based_confidence=False
                )

                # Pick the sample with the highest ranking score
                ranking_scores = torch.tensor(
                    [pair_pred["summary_confidence"][i]["ranking_score"]
                     for i in range(N_sample)]
                )
                best_sample_idx = torch.argmax(ranking_scores).item()
                best_coordinates = pair_pred["coordinate"][best_sample_idx]
                pae = pair_pred["pae"][best_sample_idx].cuda()
                iptm_scores = [
                    pair_pred["summary_confidence"][i]["iptm"] for i in range(N_sample)
                ]
                mean_iptm = torch.mean(torch.tensor(iptm_scores)).item()

                # Context selection for chain i (query = chain_id_i)
                sel_i, priority_i = self.select_context_tokens(
                    query_chain_id=chain_id_i,
                    input_feature_dict=pair_feat,
                    pred_coordinates=best_coordinates,
                    is_resolved=is_resolved_pair,
                    top_percent=0.5,
                    pae=pae,
                    global_token_indices=pair_token_indices,
                    gff3_dir=gff3_dir,
                    pdb_id=pdb_id,
                )
                sel_i_global = pair_token_indices[sel_i]
                other_token_count_i = (~torch.isin(sel_i_global, tokens_i)).sum().item()
                if other_token_count_i > 0 and mean_iptm >= 0.5:
                    interaction_chain_counts[idx_i] += 1
                interaction_token_counts[idx_i] += other_token_count_i
                selected_token_sets[idx_i].append(sel_i_global)
                priority_info_sets[idx_i].append(priority_i)

                # Context selection for chain j (query = chain_id_j)
                sel_j, priority_j = self.select_context_tokens(
                    query_chain_id=chain_id_j,
                    input_feature_dict=pair_feat,
                    pred_coordinates=best_coordinates,
                    is_resolved=is_resolved_pair,
                    top_percent=0.5,
                    pae=pae,
                    global_token_indices=pair_token_indices,
                    gff3_dir=gff3_dir,
                    pdb_id=pdb_id,
                )
                sel_j_global = pair_token_indices[sel_j]
                other_token_count_j = (~torch.isin(sel_j_global, tokens_j)).sum().item()
                if other_token_count_j > 0 and mean_iptm >= 0.5:
                    interaction_chain_counts[idx_j] += 1
                interaction_token_counts[idx_j] += other_token_count_j
                selected_token_sets[idx_j].append(sel_j_global)
                priority_info_sets[idx_j].append(priority_j)

                del pair_pred, pair_feat, is_resolved_pair, best_coordinates, pae
                torch.cuda.empty_cache()

            # ------------------------------------------------------------------ #
            # Stage 2: Per-chain full prediction                                  #
            # ------------------------------------------------------------------ #
            MAX_CONTEXT_TOKENS = 2000
            chain_id = polymer_chain_ids[idx_i]
            original_chain_tokens = torch.where(asym_id == chain_id)[0]
            # Merge all pairwise context token sets for this chain
            if not selected_token_sets[idx_i]:
                # No inter-chain context found; fall back to chain + ligands
                context_tokens = torch.where(
                    (asym_id == chain_id) | is_ligand_per_token
                )[0]
            else:
                context_tokens = torch.cat(selected_token_sets[idx_i])

            context_tokens, _ = torch.sort(torch.unique(context_tokens))

            # Clip to budget if necessary
            if len(context_tokens) > MAX_CONTEXT_TOKENS:
                logger.info(
                    f"Chain {chain_id}: clipping context from "
                    f"{len(context_tokens)} to {MAX_CONTEXT_TOKENS} tokens "
                    f"(chain has {original_chain_tokens.shape[0]} tokens; "
                    f"complex has {N_token} tokens)"
                )
                context_tokens = self.clip_tokens_to_max(
                    context_tokens,
                    priority_info_sets[idx_i],
                    max_tokens=MAX_CONTEXT_TOKENS,
                )

            # Build the feature subset for this context window
            context_feat = self.create_subset_input_feature_dict(
                input_feature_dict, context_tokens
            )

            # Build local token→atom map for the subset
            atom_to_token_local = context_feat["atom_to_token_idx"].tolist()
            token_to_atom_local: List[List[int]] = [
                [] for _ in range(context_tokens.shape[0])
            ]
            for atom_idx, token_idx in enumerate(atom_to_token_local):
                token_to_atom_local[token_idx].append(atom_idx)

            torch.cuda.empty_cache()
            logger.info(
                f"Chain {chain_id}: running full prediction on "
                f"{context_tokens.shape[0]} tokens"
            )

            pred_dict, _, _ = self._single_chain_inference(
                input_feature_dict=context_feat,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                use_consistency=False,
                chain_based_confidence=False,
            )

            # Keep only what is needed for assembly (drop heavy tensors)
            del pred_dict["contact_probs"], pred_dict["pae"], pred_dict["pde"]
            for m in range(len(pred_dict["summary_confidence"])):
                for key in list(pred_dict["summary_confidence"][m].keys()):
                    if "ranking_score" not in key:
                        del pred_dict["summary_confidence"][m][key]
            for m in range(len(pred_dict["full_data"])):
                for key in list(pred_dict["full_data"][m].keys()):
                    if "atom_plddt" not in key:
                        del pred_dict["full_data"][m][key]

            ranking_score = max(
                pred_dict["summary_confidence"][i]["ranking_score"]
                for i in range(N_sample))
            chain_results.append(
                dict(
                    chain_id=chain_id,
                    tokens=context_tokens,
                    pred=pred_dict,
                    token_to_atom=token_to_atom_local,
                    ranking_score=ranking_score,
                    interaction_chain_count=interaction_chain_counts[idx_i],
                    interaction_token_count=interaction_token_counts[idx_i],
                )
            )

            del context_feat, pred_dict
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------ #
        # Assemble individual chain predictions into a full-complex tensor    #
        # ------------------------------------------------------------------ #
        # Chains with more predicted interactions are placed first so they
        # serve as the rigid reference frame for subsequent chains.
        chain_results.sort(
            key=lambda d: (
                    d["interaction_chain_count"] * 10_000
                    + d["interaction_token_count"]
                    + d["ranking_score"].item()
            ),
            reverse=True,
        )

        N_atoms = input_feature_dict["is_protein"].shape[0]
        assembled_coord = torch.full(
            (N_sample, N_atoms, 3), float("nan"), device=device, dtype=dtype
        )
        assembled_plddt = torch.zeros(
            (N_sample, N_atoms, 50), device=device,
            dtype=chain_results[0]["pred"]["plddt"].dtype,
        )
        assembled_atom_plddt = torch.zeros(
            (N_sample, N_atoms), device=device,
            dtype=chain_results[0]["pred"]["full_data"][0]["atom_plddt"].dtype,
        )
        assembled_resolved = torch.zeros(
            (N_sample, N_atoms, 2), device=device, dtype=dtype
        )
        placed_token_mask = torch.zeros(N_token, dtype=torch.bool, device=device)

        ligand_candidates: dict = defaultdict(list)

        for placement_idx, chain_info in enumerate(chain_results):
            cid = chain_info["chain_id"]
            tok_idx = chain_info["tokens"]
            tok_to_atom_local = chain_info["token_to_atom"]

            # Sort samples by ranking score so best sample is placed first
            ranking_scores = [
                chain_info["pred"]["summary_confidence"][i]["ranking_score"]
                for i in range(N_sample)
            ]
            _, rank_order = torch.sort(
                torch.tensor(ranking_scores), descending=True
            )

            coord_c = chain_info["pred"]["coordinate"][rank_order]  # [N_s, N_local_atoms, 3]
            plddt_c = chain_info["pred"]["plddt"][rank_order]
            resolved_c = chain_info["pred"]["resolved"]
            atom_plddt = torch.stack(
                [chain_info["pred"]["full_data"][i]["atom_plddt"] for i in range(N_sample)]
            )[rank_order]

            # ----------- Rigid alignment onto the assembled complex ----------
            if placement_idx == 0:
                # First chain: no alignment needed
                coord_c_aligned = coord_c
            else:
                # Anchor: tokens that are already placed AND belong to this chain
                already_placed = (
                        (~is_ligand_per_token[tok_idx])
                        & placed_token_mask[tok_idx]
                        & (asym_id[tok_idx] == cid)
                )
                use_all_placed = already_placed.shape[0] <= 40
                if use_all_placed:
                    already_placed = (~is_ligand_per_token[tok_idx]) & placed_token_mask[tok_idx]

                anchor_global_tokens = tok_idx[already_placed]
                anchor_global_atoms = []
                for t in anchor_global_tokens:
                    anchor_global_atoms.extend(token_to_atom_index[t])
                anchor_global_atoms = torch.tensor(anchor_global_atoms, device=device)
                if anchor_global_tokens.numel() >= 3:
                    local_anchor_positions = already_placed.nonzero(as_tuple=True)[0]
                    local_anchor_atoms = []
                    for t in local_anchor_positions:
                        local_anchor_atoms.extend(tok_to_atom_local[t])
                    local_anchor_atoms = torch.tensor(local_anchor_atoms, device=device)
                    coord_c_aligned = []
                    for s in range(N_sample):
                        global_conf = assembled_atom_plddt[s, anchor_global_atoms]
                        if use_all_placed:
                            conf_threshold = torch.quantile(global_conf.float(), 0.6)
                            high_conf_mask = global_conf > conf_threshold
                        else:
                            high_conf_mask = global_conf > 0

                        P = coord_c[s, local_anchor_atoms[high_conf_mask], :]
                        Q = assembled_coord[s, anchor_global_atoms[high_conf_mask], :]
                        R, t_vec = kabsch(P, Q)
                        coord_c_aligned.append(apply_rigid(coord_c[s], R, t_vec))
                    coord_c_aligned = torch.stack(coord_c_aligned)
                else:
                    logger.warning(
                        f"Chain {cid}: insufficient anchor atoms for alignment "
                        f"({anchor_global_tokens.numel()} tokens); using unaligned coordinates."
                    )
                    coord_c_aligned = coord_c

            chain_info["coord_aligned"] = coord_c_aligned

            # ----------- Write polymer coordinates --------------------------
            write_mask = (
                    (~placed_token_mask[tok_idx] | (asym_id[tok_idx] == cid))
                    & (~is_ligand_per_token[tok_idx])
            )
            if write_mask.any():
                g_tokens = tok_idx[write_mask]
                l_positions = write_mask.nonzero(as_tuple=True)[0]

                g_atoms = []
                l_atoms = []
                for t in g_tokens:
                    g_atoms.extend(token_to_atom_index[t])
                for t in l_positions:
                    l_atoms.extend(tok_to_atom_local[t])
                g_atoms = torch.tensor(g_atoms, device=device)
                l_atoms = torch.tensor(l_atoms, device=device)

                for s in range(N_sample):
                    assembled_coord[s, g_atoms, :] = coord_c_aligned[s, l_atoms, :]
                    assembled_plddt[s, g_atoms, :] = plddt_c[s, l_atoms, :]
                    assembled_resolved[s, g_atoms] = resolved_c[s, l_atoms]
                    assembled_atom_plddt[s, g_atoms] = atom_plddt[s, l_atoms]
                placed_token_mask[g_tokens] = True

            # ----------- Collect ligand candidates --------------------------
            lig_mask = is_ligand_per_token[tok_idx]
            if lig_mask.any():
                lig_asym_ids = asym_id[tok_idx][lig_mask]
                for lg in torch.unique(lig_asym_ids):
                    lg_mask = asym_id[tok_idx] == lg
                    lig_global_tokens = tok_idx[lg_mask]
                    lig_local_positions = lg_mask.nonzero(as_tuple=True)[0]

                    lig_global_atoms = []
                    lig_local_atoms = []
                    for t in lig_global_tokens:
                        lig_global_atoms.extend(token_to_atom_index[t])
                    for t in lig_local_positions:
                        lig_local_atoms.extend(tok_to_atom_local[t])
                    lig_global_atoms = torch.tensor(lig_global_atoms, device=device)
                    lig_local_atoms = torch.tensor(lig_local_atoms, device=device)

                    mean_conf = atom_plddt[:, lig_local_atoms].mean().item()
                    ligand_candidates[int(lg.item())].append((
                        mean_conf,
                        lig_global_atoms,
                        lig_global_tokens,
                        coord_c_aligned[:, lig_local_atoms, :],
                        plddt_c[:, lig_local_atoms, :],
                        atom_plddt[:, lig_local_atoms],
                        resolved_c[:, lig_local_atoms],
                    ))

        # ------------------------------------------------------------------ #
        # Choose the best ligand copy (highest mean pLDDT)                    #
        # ------------------------------------------------------------------ #
        for lg_id, copies in ligand_candidates.items():
            (best_conf, best_atoms, best_tok, best_coord,
             best_plddt, best_plddt_atom, best_resolved) = max(copies, key=lambda x: x[0])
            assembled_coord[:, best_atoms, :] = best_coord
            assembled_plddt[:, best_atoms] = best_plddt
            placed_token_mask[best_tok] = True
            assembled_resolved[:, best_atoms] = best_resolved
            assembled_atom_plddt[:, best_atoms] = best_plddt_atom

        # ------------------------------------------------------------------ #
        # Build output dict                                                   #
        # ------------------------------------------------------------------ #
        pred_dict = {
            "coordinate": assembled_coord,
            "plddt": assembled_plddt,
            "pae": torch.full(
                (N_atoms, N_atoms), float("nan"), device=device, dtype=dtype
            ),
            "pde": torch.full(
                (N_atoms, N_atoms), float("nan"), device=device, dtype=dtype
            ),
            "resolved": (assembled_atom_plddt > 0.5).float(),
            "contact_probs": torch.full(
                (N_token, N_token), float("nan"), device=device, dtype=dtype
            ),
            "summary_confidence": [
                {"ranking_score": torch.mean(assembled_atom_plddt[i])}
                for i in range(N_sample)
            ],
            "full_data": [
                {"atom_plddt": assembled_atom_plddt[i]} for i in range(N_sample)
            ],
        }

        log_dict = {}
        time_tracker = {"total": time.time() - t0}
        return pred_dict, log_dict, time_tracker

    def _single_chain_inference(
            self,
            input_feature_dict: dict[str, Any],
            label_dict: dict[str, Any],
            N_cycle: int,
            mode: str,
            inplace_safe: bool = True,
            chunk_size: Optional[int] = 4,
            symmetric_permutation: SymmetricPermutation = None,
            use_consistency: bool = False,
            chain_based_confidence: bool = True
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Single-pass inference on a given (sub-)set of tokens.

        Runs the pairformer trunk, samples diffusion trajectories, and computes
        confidence scores.  Used both for the pairwise pre-screening step
        (``use_consistency=True``, fewer diffusion steps via the consistency
        model) and for the final per-chain full prediction
        (``use_consistency=False``).

        Args:
            input_feature_dict: Input features for the token subset.
            label_dict: Label dictionary (may be ``None`` at inference).
            N_cycle: Number of recycling cycles.
            mode: ``'inference'`` or ``'eval'``.
            inplace_safe: Whether in-place tensor ops are safe.
            chunk_size: Chunk size for memory-efficient attention.
            symmetric_permutation: Symmetric permutation object.
            use_consistency: When ``True`` the lightweight consistency model
                (``self.consistency_contact``) is used for faster coordinate
                sampling during the pairwise pre-screening stage.

        Returns:
            pred_dict: Prediction dictionary with keys ``coordinate``,
                ``plddt``, ``pae``, ``pde``, ``resolved``,
                ``contact_probs``, ``summary_confidence``, ``full_data``.
            log_dict: Log dictionary.
            time_tracker: Timing dictionary.
        """
        step_st = time.time()
        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Free MSA / template features after the trunk to reduce memory
        if mode == "inference":
            for key in list(input_feature_dict.keys()):
                if "template_" in key or key in [
                    "msa", "has_deletion", "deletion_value",
                    "profile", "deletion_mean", "token_bonds",
                ]:
                    del input_feature_dict[key]

        time_tracker["pairformer"] = time.time() - step_st
        step_trunk = time.time()

        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]
        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )

        # Optionally pre-compute and cache shared diffusion variables
        cache: dict = {}
        use_cache = (
                self.enable_diffusion_shared_vars_cache
                and "d_lm" in input_feature_dict
        )
        if use_cache:
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                input_feature_dict["ref_pos"],
                input_feature_dict["ref_charge"],
                input_feature_dict["ref_mask"],
                input_feature_dict["ref_element"],
                input_feature_dict["ref_atom_name_chars"],
                input_feature_dict["atom_to_token_idx"],
                input_feature_dict["d_lm"],
                input_feature_dict["v_lm"],
                input_feature_dict["pad_info"],
                "",
                cache["pair_z"],
                False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]

        # Sample coordinates
        if use_consistency:
            pred_dict["coordinate"] = self.sample_consistency(consistency_net=self.consistency_contact,
                                                              inference_scheduler=self.inference_noise_scheduler,
                                                              ts=[0, 66, 132, 199], p_lm=None, c_l=None, pair_z=None,
                                                              input_feature_dict=input_feature_dict, s_inputs=s_inputs,
                                                              s_trunk=s, z_trunk=z, N_sample=N_sample,
                                                              noise_schedule=noise_schedule, inplace_safe=inplace_safe)
        else:
            pred_dict["coordinate"] = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=None if cache["pair_z"] is not None else z,
                pair_z=cache["pair_z"],
                p_lm=cache["p_lm/c_l"][0],
                c_l=cache["p_lm/c_l"][1],
                N_sample=N_sample,
                noise_schedule=noise_schedule,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=self.enable_efficient_fusion,
            )

        time_tracker["diffusion"] = time.time() - step_trunk
        step_diffusion = time.time()

        # Distogram → contact probabilities
        pred_dict["contact_probs"] = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )

        # Confidence head
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        time_tracker["confidence"] = time.time() - step_diffusion
        time_tracker["model_forward"] = time.time() - step_st

        # Symmetric permutation (eval mode only)
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=(
                        "pocket_mask" in label_dict
                        and "interested_ligand_mask" in label_dict
                ),
            )
            time_tracker["permutation"] = time.time() - step_diffusion

        # Summary confidence and full data
        interested_atom_mask = (
            label_dict.get("interested_ligand_mask", None)
            if label_dict is not None
            else None
        )
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=True,
            mol_id=(input_feature_dict["mol_id"] if mode != "inference" else None),
            elements_one_hot=(
                input_feature_dict["ref_element"] if mode != "inference" else None
            ),
            cal_chain_based=chain_based_confidence,
        )

        return pred_dict, log_dict, time_tracker

    # ---------------------------------------------------------------------- #
    # Training loop                                                            #
    # ---------------------------------------------------------------------- #

    def main_train_loop(
            self,
            input_feature_dict: dict[str, Any],
            label_full_dict: dict[str, Any],
            label_dict: dict,
            N_cycle: int,
            symmetric_permutation: SymmetricPermutation,
            inplace_safe: bool = False,
            chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        N_token = input_feature_dict["token_index"].shape[-1]

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        log_dict = {}
        pred_dict = {}

        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                input_feature_dict["ref_pos"],
                input_feature_dict["ref_charge"],
                input_feature_dict["ref_mask"],
                input_feature_dict["ref_element"],
                input_feature_dict["ref_atom_name_chars"],
                input_feature_dict["atom_to_token_idx"],
                input_feature_dict["d_lm"],
                input_feature_dict["v_lm"],
                input_feature_dict["pad_info"],
                "",
                cache["pair_z"],
                False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        # Mini-rollout: used for confidence and label permutation
        with torch.no_grad():
            # [..., 1, N_atom, 3]
            N_sample_mini_rollout = self.configs.sample_diffusion[
                "N_sample_mini_rollout"
            ]  # =1
            N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]

            coordinate_mini = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s.detach(),
                z_trunk=None if cache["pair_z"] is not None else z.detach(),
                pair_z=None if cache["pair_z"] is None else cache["pair_z"].detach(),
                p_lm=(
                    None
                    if cache["p_lm/c_l"][0] is None
                    else cache["p_lm/c_l"][0].detach()
                ),
                c_l=(
                    None
                    if cache["p_lm/c_l"][1] is None
                    else cache["p_lm/c_l"][1].detach()
                ),
                N_sample=N_sample_mini_rollout,
                noise_schedule=self.inference_noise_scheduler(
                    N_step=N_step_mini_rollout,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                ),
                enable_efficient_fusion=self.enable_efficient_fusion,
            )
            coordinate_mini.detach_()
            pred_dict["coordinate_mini"] = coordinate_mini

            # Permute ground truth to match mini-rollout prediction
            label_dict, perm_log_dict = (
                symmetric_permutation.permute_label_to_match_mini_rollout(
                    coordinate_mini,
                    input_feature_dict,
                    label_dict,
                    label_full_dict,
                )
            )
            log_dict.update(perm_log_dict)

        # Confidence: use mini-rollout prediction, and detach token embeddings
        drop_embedding = (
                random.random() < self.configs.model.confidence_embedding_drop_rate
        )
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coordinate_mini,
            use_embedding=not drop_embedding,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        pred_dict.update(
            {
                "plddt": plddt_pred,
                "pae": pae_pred,
                "pde": pde_pred,
                "resolved": resolved_pred,
            }
        )

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        # x_denoised: [..., N_sample, N_atom, 3]
        # x_noise_level: [..., N_sample]
        N_sample = self.diffusion_batch_size
        drop_conditioning = (
                random.random() < self.configs.model.condition_embedding_drop_rate
        )
        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            use_conditioning=not drop_conditioning,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(True)(self.distogram_head)(
                    z
                ),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        pred_dict, perm_log_dict, _, _ = (
            symmetric_permutation.permute_diffusion_sample_to_match_label(
                input_feature_dict, pred_dict, label_dict, stage="train"
            )
        )
        log_dict.update(perm_log_dict)

        return pred_dict, label_dict, log_dict

    # ---------------------------------------------------------------------- #
    # Top-level forward                                                        #
    # ---------------------------------------------------------------------- #

    def forward(
            self,
            input_feature_dict: dict[str, Any],
            label_full_dict: dict[str, Any],
            label_dict: dict[str, Any],
            mode: str = "inference",
            current_step: Optional[int] = None,
            symmetric_permutation: SymmetricPermutation = None,
            gff3_dir: Optional[str] = None,
            pdb_id: Optional[str] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the HierAFold model.

        Dispatches to :meth:`main_train_loop` (training) or
        :meth:`main_inference_loop` (inference / eval).

        Args:
            input_feature_dict: Input features dictionary.
            label_full_dict: Full (uncropped) label dictionary.
            label_dict: Cropped label dictionary.
            mode: One of ``'train'``, ``'inference'``, or ``'eval'``.
                Defaults to ``'inference'``.
            current_step: Current training step (used to seed N_cycle sampling).
            symmetric_permutation: Symmetric permutation object for eval / train.
            gff3_dir: Optional directory containing GFF3 domain annotation files.
                When provided, domain boundaries read from GFF3 files are used
                as the primary domain-splitting source during context-token
                selection.  Ignored in training mode.
            pdb_id: PDB identifier used to locate the correct GFF3 file.
                Ignored in training mode.

        Returns:
            pred_dict: Prediction dictionary.
            label_dict: (Possibly permuted) label dictionary.
            log_dict: Log / diagnostic dictionary.
        """
        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                gff3_dir=gff3_dir,
                pdb_id=pdb_id,
            )
            log_dict.update({"time": time_tracker})

        elif mode == "eval":
            if label_dict is not None:
                assert (
                        label_dict["coordinate"].size()
                        == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=symmetric_permutation,
                gff3_dir=gff3_dir,
                pdb_id=pdb_id,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict


# Backwards-compatibility alias
ProtenixSelect = HierAFold
