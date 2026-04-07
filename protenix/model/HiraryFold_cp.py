# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    sample_consistency,
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



class ProtenixSelect(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs) -> None:
        super(ProtenixSelect, self).__init__()
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

    def recall_and_acc(self, sel_token, sel_tokens2):
        sel_token = set(sel_token.tolist())
        sel_token2 = set(sel_tokens2.tolist())
        true_positives = len(sel_token.intersection(sel_token2))
        # Calculate precision
        precision = true_positives / (len(sel_token) + 1e-7)
        # Calculate recall
        recall = true_positives / (len(sel_token2) + 1e-7)
        return precision, recall

    def get_sym_id(self, input_feature_dict):
        asym_id = input_feature_dict["asym_id"]  # [N_token]
        sym = input_feature_dict["sym_id"]  # [N_token]
        sym_id_tensor = []
        sym_id = -1
        for i in torch.unique(asym_id):
            asym_token = torch.where(asym_id == i)[0]
            if asym_token.shape[0] > 1000:
                sym_id += 1
            else:
                sym_token = torch.unique(sym[asym_token])
                assert sym_token.shape[0] == 1, print('sym_token', sym_token)
                if sym_token == 0:
                    sym_id += 1
            sym_id_tensor.extend([sym_id for _ in range(asym_token.shape[0])])
        return torch.tensor(sym_id_tensor, dtype=asym_id.dtype, device=asym_id.device)


    def select_tokens_for_chain_subunit_v4(
            self,
            chain_id: int,
            input_feature_dict: dict[str, Any],
            gt_coordinates: torch.Tensor,
            is_resolved: torch.Tensor,
            top_percent: float = 0.5,
            sample_percent: float = 0.0,
            pde: torch.Tensor = None,
            sym=True,
            gff3_dir: str = None,
            pdb_id: str = None,
            pair_tokens = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Efficiently select tokens for the given protein chain.
        Modified to select the nearest subunit > 20 residues instead of all subunits < 20A.

        Returns:
            selected_tokens: Indices of selected tokens
            priority_info: Dict mapping subunit_id -> (priority_score, tokens, is_target_chain)
        """
        device = gt_coordinates.device
        asym_id = input_feature_dict["asym_id"]
        is_ligand = input_feature_dict["is_ligand"].bool()
        is_ligand_token = []
        is_resolved_token = []
        centre_atom_index = []
        frame_atom_index = input_feature_dict["frame_atom_index"]
        N_token = asym_id.shape[0]

        for i in range(N_token):
            assert frame_atom_index[i][1] != -1
            center_atom = frame_atom_index[i][1]
            centre_atom_index.append(center_atom)
            is_resolved_token.append(is_resolved[center_atom])
            is_ligand_token.append(is_ligand[center_atom])

        is_ligand_token = torch.tensor(is_ligand_token, device=device).bool()
        is_resolved_token = torch.tensor(is_resolved_token, device=device).bool()

        # 1. Get center coordinates for each token
        token_centers = gt_coordinates[centre_atom_index]

        # 2. Get tokens for the target chain
        chain_tokens = torch.where(asym_id == chain_id)[0]
        chain_tokens_isresolved = torch.where((asym_id == chain_id) & (is_resolved_token == True))[0]
        chain_center_coords = token_centers[chain_tokens_isresolved]

        # 3. Get all other chain ids (excluding ligands and target chain)
        all_chain_ids = torch.unique(asym_id[is_ligand_token == 0])
        other_chain_ids = all_chain_ids[all_chain_ids != chain_id]

        selected_tokens = [chain_tokens]
        priority_info = {}
        subunit_counter = 0

        # Add target chain with highest priority (score = 0)
        priority_info[f"target_{chain_id}"] = (0.0, pair_tokens[chain_tokens], True)
        subunit_counter += 1

        # 4. For each other chain, select top % tokens based on center-to-center distance
        for other_id in other_chain_ids.tolist():
            other_chain_tokens = \
                torch.where((asym_id == other_id) & (is_ligand_token == 0) & (is_resolved_token == True))[0]
            if len(other_chain_tokens) == 0:
                continue

            if len(other_chain_tokens) <= 40:
                priority_score = 1.0  # High priority for small chains
                priority_info[f"small_chain_{other_id}"] = (priority_score, pair_tokens[other_chain_tokens], False)
                selected_tokens.append(other_chain_tokens)
                subunit_counter += 1
                continue

            other_centers = token_centers[other_chain_tokens]
            # Compute pairwise distances [N_chain_tokens, N_other_chain_tokens]
            dists = torch.cdist(chain_center_coords, other_centers)
            min_dist_per_token, min_dist_per_token_indices = dists.min(dim=0)
            sorted_min_dist_per_token, _ = torch.sort(min_dist_per_token)

            # Select top % closest tokens for this chain
            n_select = max(min(50, len(other_chain_tokens)), int(top_percent * len(other_chain_tokens)))
            cutoff_index = torch.searchsorted(sorted_min_dist_per_token, 40).item()
            print('n_select', chain_id, other_id, n_select, cutoff_index)
            n_select = min(n_select, cutoff_index)
            sorted_indices = torch.argsort(min_dist_per_token)
            selected_chain_tokens = other_chain_tokens[sorted_indices[:n_select]]

            subunits = None
            replicate = 1
            print('replicate', replicate, other_id, self.pdb_id)

            if gff3_dir is not None and pdb_id is not None:
                gff3_filename = f"{pdb_id}_seq_{other_id}.fasta.gff3"
                gff3_path = os.path.join(gff3_dir, gff3_filename)
                sequence_length = len(torch.where(asym_id == other_id)[0])
                subunits = parse_gff3_domains(gff3_path, sequence_length, replicate=replicate)
                print(f'Subunits from GFF3 for chain {other_id}: {subunits}')
            else:
                print('no gff3 dir provided')

            if subunits is None or len(subunits) == replicate:
                if pde is not None:
                    pde_tmp = torch.argmax(pde, dim=-1)
                    pde_tmp_own_chain = pde_tmp[other_chain_tokens][:, other_chain_tokens]
                    print('pde tmp own chain,', pde_tmp_own_chain.max(), pde_tmp_own_chain.min())
                    if pde_tmp.max() > 10:
                        print('dividing subunits', chain_id, other_id)
                        subunits = get_domain_splits(pde_tmp_own_chain.cpu().numpy(), 10)
                        subunits = [(subunits[i - 1], subunits[i]) for i in range(1, len(subunits))]
                    else:
                        subunits = [(0, other_chain_tokens.shape[0])]

            print('len subunits', len(subunits))

            if pde is not None:
                pde_tmp = torch.argmax(pde, dim=-1)
                pde_tmp = pde_tmp[chain_tokens_isresolved][:, other_chain_tokens]

                print('before pde', int(torch.unique(selected_chain_tokens).shape[0]), other_chain_tokens.shape)
                pde_tmp_quantile = torch.quantile(pde_tmp.float(), 0.2, dim=0)
                pde_selected_token = other_chain_tokens[torch.where(pde_tmp_quantile <= 10)[0]]
                if pde_selected_token.shape[0] >= other_chain_tokens.shape[0] * 0.7:  # too much
                    pde_tmp_quantile = torch.quantile(pde_tmp.float(), 0.5, dim=0)
                    pde_selected_token = other_chain_tokens[torch.where(pde_tmp_quantile <= 3)[0]]
                selected_chain_tokens = torch.cat((selected_chain_tokens, pde_selected_token))
                print('after pde', torch.unique(selected_chain_tokens).shape, other_chain_tokens.shape, pde_tmp.shape,
                      pde_selected_token.shape)

            # --- ORIGINAL LOGIC: Select only the nearest subunit among large ones ---
            if len(subunits) > 1:
                HARD_DIST_CUTOFF = 5.0
                SOFT_DIST_CUTOFF = 20.0
                MAX_TIER2_K = 1  # Max number of "ambiguous" subunits to keep
                selected_chain_tokens_subunit = []
                large_subunits_candidates = []  # List to store (distance, token, subunit_info, pde_val)

                for subunit in subunits:
                    subunit_token = other_chain_tokens[subunit[0]:subunit[1]]

                    # Calculate metrics for subunits
                    subunit_centers = token_centers[subunit_token]
                    subunit_dists = torch.cdist(chain_center_coords, subunit_centers)
                    print('subunit dists shape', subunit_dists.shape)
                    min_subunit_dist = subunit_dists.min().item()

                    # Calculate 0.5 percentile PDE between subunit and target chain
                    pde_subunit = pde_tmp[:, subunit[0]:subunit[1]]
                    pde_subunit_percentile = torch.quantile(pde_subunit.float(), 0.5, dim=0)
                    pde_subunit_percentile = torch.quantile(pde_subunit_percentile, 0.5, dim=0)

                    # Priority score: distance + 2 * PDE (lower = better)
                    priority_score = min_subunit_dist + 2.0 * pde_subunit_percentile.item()

                    # Keep if PDE is strong (original OR condition logic)
                    if pde_subunit_percentile < 3:
                        selected_chain_tokens_subunit.append(subunit_token)
                        priority_info[f"chain_{other_id}_sub_{subunit_counter}_pde"] = (
                            priority_score, pair_tokens[subunit_token], False
                        )
                        subunit_counter += 1
                        continue

                    if min_subunit_dist < HARD_DIST_CUTOFF:
                        selected_chain_tokens_subunit.append(subunit_token)
                        priority_info[f"chain_{other_id}_sub_{subunit_counter}_close"] = (
                            priority_score, pair_tokens[subunit_token], False
                        )
                        subunit_counter += 1
                        continue

                    if min_subunit_dist < SOFT_DIST_CUTOFF:
                        # Store candidate for distance comparison
                        large_subunits_candidates.append(
                            (min_subunit_dist, subunit_token, subunit, pde_subunit_percentile, priority_score))
                    else:
                        print('--------------------------------------------------')
                        print('subunit not chosen', subunit, subunits, min_subunit_dist, pde_subunit_percentile)

                # ORIGINAL LOGIC: Select the single nearest subunit among those > 20
                if len(large_subunits_candidates) > 0:
                    # Sort by distance (ascending)
                    large_subunits_candidates.sort(key=lambda x: x[0])
                    # Add ONLY the nearest one
                    nearest_dist, nearest_token, nearest_sub, nearest_pde, nearest_priority = large_subunits_candidates[
                        0]
                    selected_chain_tokens_subunit.append(nearest_token)
                    priority_info[f"chain_{other_id}_sub_{subunit_counter}_nearest"] = (
                        nearest_priority, pair_tokens[nearest_token], False
                    )
                    subunit_counter += 1
                    print(f'Selected nearest large subunit: {nearest_sub} dist: {nearest_dist}')

                    # Log the ones that were not chosen
                    for dist, token, sub, pde_val, _ in large_subunits_candidates[1:]:
                        print('--------------------------------------------------')
                        print('subunit not chosen', sub, subunits, dist, pde_val)

                selected_chain_tokens = torch.cat(selected_chain_tokens_subunit)
            else:
                # Single subunit - calculate priority
                min_dist = dists.min().item()
                pde_score = 0.0
                if pde is not None:
                    pde_score = torch.quantile(pde_tmp.float(), 0.5).item()
                priority_score = min_dist + 2.0 * pde_score
                priority_info[f"chain_{other_id}_full"] = (priority_score, pair_tokens[selected_chain_tokens], False)

            selected_tokens.append(selected_chain_tokens)

            unresolved_token = \
                torch.where((asym_id == other_id) & (is_ligand_token == 0) & (is_resolved_token == False))[0]
            if len(unresolved_token) > 0:
                priority_info[f"chain_{other_id}_unresolved"] = (1000.0, pair_tokens[unresolved_token], False)
                selected_tokens.append(unresolved_token)

        # 5. Ligand logic: select all ligands
        ligand_tokens = torch.where(is_ligand_token == 1)[0]
        if len(ligand_tokens) > 0:
            priority_info["ligands"] = (0.5, pair_tokens[ligand_tokens], False)
            selected_tokens.append(ligand_tokens)

        # 6. Combine and deduplicate
        selected_tokens = torch.cat(selected_tokens).unique()
        sorted_selected, _ = torch.sort(selected_tokens)

        return sorted_selected, priority_info

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
            is_resolved=None,
    ) -> dict[str, Any]:
        # Mapping from original token indices to new indices (0 to N_selected - 1)
        """
            Create a subset of input_feature_dict for the selected tokens, adjusting indices appropriately.

            Args:
                input_feature_dict: Original input feature dictionary.
                selected_tokens: Indices of tokens to include in the subset.

            Returns:
                subset_dict: New input feature dictionary for the subset.
            """
        # input_feature_dict = _input_feature_dict.copy()
        subset_dict = {}
        device = selected_tokens.device
        selected_tokens = selected_tokens.cpu()
        N_selected = len(selected_tokens)

        # Mapping from original token indices to new indices (0 to N_selected - 1)
        token_mapping = {int(t.item()): i for i, t in enumerate(selected_tokens)}

        # Per-token features
        per_token_keys = ["residue_index", "asym_id", "entity_id", "sym_id", "restype", "has_frame", "profile",
                          "sym_id_custom",
                          "deletion_mean"]
        for key in per_token_keys:
            if key in input_feature_dict:
                subset_dict[key] = input_feature_dict[key][selected_tokens].clone()
        subset_dict['token_index'] = torch.arange(0, N_selected, device=device)

        # Pair features
        pair_keys = ["token_bonds"]
        for key in pair_keys:
            if key in input_feature_dict:
                subset_dict[key] = input_feature_dict[key][selected_tokens][:, selected_tokens].clone().cuda()

        # Atom features
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"].cpu()
        selected_atoms_mask = torch.isin(atom_to_token_idx, selected_tokens)
        selected_atoms = torch.where(selected_atoms_mask)[0].cpu()
        subset_dict["atom_to_token_idx"] = torch.tensor(
            [token_mapping[int(atom_to_token_idx[a].item())] for a in selected_atoms],
            dtype=atom_to_token_idx.dtype,
            device=device
        )
        if is_resolved is not None:
            is_resolved = is_resolved[selected_atoms]
        atom_mapping = {int(t.item()): i for i, t in enumerate(selected_atoms)}
        atom_mapping[-1] = -1
        frame_atom_index = input_feature_dict["frame_atom_index"]
        try:
            subset_dict["frame_atom_index"] = torch.tensor(
                [[atom_mapping[int(frame_atom_index[a][b].item())] for b in range(3)] for a in selected_tokens],
                dtype=frame_atom_index.dtype,
                device=device
            )
        except:
            pass
        atom_keys = ["ref_pos", "ref_mask", "ref_element", "atom_to_tokatom_idx", "ref_charge", "ref_atom_name_chars",
                     "ref_space_uid", "is_protein", "is_dna", "is_rna", "is_ligand",
                     "mol_id", "mol_atom_index", "entity_mol_id", "pae_rep_atom_mask", "plddt_m_rep_atom_mask",
                     "distogram_rep_atom_mask", "modified_res_mask"]
        for key in atom_keys:
            if key in input_feature_dict:
                subset_dict[key] = input_feature_dict[key][selected_atoms].clone().cuda()

        atom_char_keys = ["ref_atom_name", "ref_atom_name_chars"]
        for key in atom_char_keys:
            if key in input_feature_dict:
                subset_dict[key] = input_feature_dict[key][selected_atoms.cpu().numpy()].cuda()

        pair_keys = ["bond_mask"]
        for key in pair_keys:
            if key in input_feature_dict:
                subset_dict[key] = input_feature_dict[key][selected_atoms][:, selected_atoms].clone().cuda()

        kept_keys = ['resolution']
        for key in kept_keys:
            subset_dict[key] = input_feature_dict[key].clone()
        # MSA features (if present and needed)
        if "msa" in input_feature_dict:
            subset_dict["msa"] = input_feature_dict["msa"][:, selected_tokens.cpu()].clone().cuda()
            subset_dict["has_deletion"] = input_feature_dict["has_deletion"][:, selected_tokens.cpu()].clone().cuda()
            subset_dict["deletion_value"] = input_feature_dict["deletion_value"][:,
                                            selected_tokens.cpu()].clone().cuda()

        # Placeholder for template features (simplified)
        if "template_input" in input_feature_dict:
            subset_dict["template_input"] = None  # Adjust based on actual needs
        if 'coordinate' in input_feature_dict:
            subset_dict['coordinate'] = input_feature_dict['coordinate'][selected_atoms, :].clone()
        subset_dict = self.relative_position_encoding.generate_relp(
            subset_dict
        )
        # with torch.no_grad():
        #     token_res_idx = subset_dict["residue_index"]
        #     token_asym_id = subset_dict["asym_id"]
        #
        #     # Detect breaks: where residue index jumps != 1 or chain ID changes
        #     if len(token_res_idx) > 1:
        #         diff_res = token_res_idx[1:] - token_res_idx[:-1]
        #         diff_asym = token_asym_id[1:] - token_asym_id[:-1]
        #
        #         # A break exists if residues are not sequential OR chains are different
        #         is_break = (diff_res != 1) | (diff_asym != 0)
        #
        #         # Create unique segment IDs for each contiguous block
        #         segment_ids = torch.cat([torch.tensor([0], device=device), is_break.long()]).cumsum(dim=0)
        #
        #         # Map these segment IDs to the atoms
        #         # subset_dict["atom_to_token_idx"] maps atoms -> tokens
        #         atom_segment_ids = segment_ids[subset_dict["atom_to_token_idx"]]
        #
        #         # Update ref_space_uid. The update_input_feature_dict function uses this
        #         # to generate v_lm. Pairs with different UIDs get v_lm=0.
        #         subset_dict["ref_space_uid"] = atom_segment_ids.int()
        subset_dict = update_input_feature_dict(subset_dict)
        # for key in input_feature_dict:
        #     if 'template' not in key:
        #         assert key in subset_dict, print(key, input_feature_dict[key])
        if is_resolved is not None:
            return subset_dict, is_resolved
        return subset_dict

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

        s_init = self.linear_no_bias_sinit(s_inputs)  #  [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]
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
        gt_coordinates: torch.Tensor = None,
        is_resolved: torch.Tensor = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        pred_dicts = []
        log_dicts = []
        time_trackers = []
        gt_coordinates = gt_coordinates.cuda() if gt_coordinates is not None else None
        for _ in range(N_model_seed):
            pred_dict, log_dict, time_tracker = self._main_inference_loop_large_auc2(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                gt_coordinates=gt_coordinates,
                is_resolved=is_resolved,
            )
            pred_dicts.append(pred_dict)
            log_dicts.append(log_dict)
            time_trackers.append(time_tracker)

        # Combine outputs of multiple models
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

    def _main_inference_loop_large_auc2(
            self,
            input_feature_dict: dict[str, Any],
            label_dict: dict[str, Any],
            N_cycle: int,
            mode: str,
            inplace_safe: bool = True,
            gt_coordinates: torch.Tensor = None,
            is_resolved: torch.Tensor = None,
            chunk_size: Optional[int] = 4,
            symmetric_permutation: "SymmetricPermutation" = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        t0 = time.time()
        N_sample = self.configs.sample_diffusion["N_sample"]
        device, dtype = gt_coordinates.device, gt_coordinates.dtype
        # ---------------------------------------------------------------- #
        # 1.  Split polymer chains – every polymer chain is run separately #
        # ---------------------------------------------------------------- #
        # asym_id = input_feature_dict["asym_id"]  # [N_token]
        # sym = True
        # if torch.sum(input_feature_dict["sym_id"]!=0) == 0: # only one chain
        sym = False
        asym_id = input_feature_dict["asym_id"]
        input_feature_dict['sym_id_custom'] = asym_id
        is_ligand = input_feature_dict["is_ligand"].bool()  # [N_token]
        is_ligand_token = []
        N_token = asym_id.shape[0]
        frame_atom_index = input_feature_dict["frame_atom_index"]
        for i in range(N_token):
            is_ligand_token.append(is_ligand[frame_atom_index[i][1]])
        is_ligand_tok = torch.tensor(is_ligand_token, device=device).bool()
        is_resolved = torch.ones_like(is_ligand).bool()
        atom_to_token_index = input_feature_dict["atom_to_token_idx"].tolist()
        # Prepare a list for each token
        token_to_atom_index = [[] for _ in range(N_token)]
        for atom_idx, token_idx in enumerate(atom_to_token_index):
            token_to_atom_index[token_idx].append(atom_idx)
        polymer_ids = torch.unique(asym_id[~is_ligand_tok])
        chain_results: list[dict] = []  # stores per‑chain predictions
        chain_tokens: dict[int, torch.Tensor] = {}  # global token indices per chain
        polymer_ids = polymer_ids.tolist()
        sel_tokens_record = [[] for _ in range(len(polymer_ids))]
        priority_info_record = [[] for _ in range(len(polymer_ids))]
        interact_num_chain = [0 for _ in range(len(polymer_ids))]
        interact_num_token = [0 for _ in range(len(polymer_ids))]
        if N_token > 2000:
            input_feature_dict['bond_mask'] = input_feature_dict['bond_mask'].cpu()  # allocate to cpu to save memory
            input_feature_dict['msa'] = input_feature_dict['msa'].cpu()
            input_feature_dict['has_deletion'] = input_feature_dict['has_deletion'].cpu()
            input_feature_dict['deletion_value'] = input_feature_dict['deletion_value'].cpu()
            atom_keys = ["ref_pos", "ref_mask", "ref_element", "atom_to_tokatom_idx", "ref_charge",
                         "ref_atom_name_chars",
                         "ref_space_uid", "is_protein", "is_dna", "is_rna", "is_ligand",
                         "mol_id", "mol_atom_index", "entity_mol_id", "pae_rep_atom_mask", "plddt_m_rep_atom_mask",
                         "distogram_rep_atom_mask", "modified_res_mask"]
            for atom_key in atom_keys:
                input_feature_dict[atom_key] = input_feature_dict[atom_key].cpu()

            input_feature_dict['token_bonds'] = input_feature_dict['token_bonds'].cpu()
            torch.cuda.empty_cache()
        for idx in range(len(polymer_ids)):
            chain_id = polymer_ids[idx]
            original_chain_token = torch.where(asym_id == chain_id)[0]
            for idx2 in range(idx + 1, len(polymer_ids)):
                # if sym_id[torch.where(asym_id == chain_id)[0][0]] ==
                chain_id2 = polymer_ids[idx2]
                original_chain_token_chainid2 = torch.where(asym_id == chain_id2)[0]
                print('extracting pair representation and tokens', chain_id, chain_id2)
                pair_tokens = torch.where((asym_id == chain_id) | (asym_id == chain_id2) | (is_ligand_tok == True))[0]
                pair_feature_dict, is_resolved_pair = self.create_subset_input_feature_dict(input_feature_dict,
                                                                                            pair_tokens, is_resolved)
                N_step = self.configs.sample_diffusion["N_step"]
                pair_pred_dict, _, _ = self._main_inference_loop(pair_feature_dict, label_dict,
                                                                 N_cycle=N_cycle,
                                                                 mode=mode,
                                                                 inplace_safe=inplace_safe,
                                                                 chunk_size=chunk_size,
                                                                 symmetric_permutation=symmetric_permutation,
                                                                 cal_chain_based=False, consistency=True)
                iptm = [pair_pred_dict['summary_confidence'][i]['iptm'] for i in range(N_sample)]
                pair_coordinates = pair_pred_dict['coordinate']
                ranking_scores = torch.tensor(
                    [pair_pred_dict["summary_confidence"][i]["ranking_score"] for i in range(N_sample)])
                max_ranking_score_index = torch.argmax(ranking_scores)
                # max_ranking_score_index = torch.argmax(torch.tensor(iptm))
                pair_coordinates = pair_coordinates[max_ranking_score_index]
                assert pair_coordinates.shape[0] == pair_feature_dict['is_ligand'].shape[0]
                pde = pair_pred_dict['pae'][max_ranking_score_index].cuda()

                # sel_tokens = pair_tokens[self.select_tokens_for_chain_subunit_v4(
                #     chain_id=chain_id,
                #     input_feature_dict=pair_feature_dict,
                #     is_resolved=is_resolved_pair,
                #     gt_coordinates=pair_coordinates,
                #     top_percent=0.5,
                #     pde=pde,
                #     sym=sym,
                #     gff3_dir='/home/u3590540/cxlu/protein/Protenix4/Protenix/Interpro_subunit',
                #     pdb_id=self.pdb_id,
                # )]
                sel_tokens, priority_info = self.select_tokens_for_chain_subunit_v4(
                    chain_id=chain_id,
                    input_feature_dict=pair_feature_dict,
                    is_resolved=is_resolved_pair,
                    gt_coordinates=pair_coordinates,
                    top_percent=0.5,
                    pde=pde,
                    sym=sym,
                    gff3_dir='/home/u3590540/cxlu/protein/Protenix4/Protenix/Interpro_subunit',
                    pdb_id=self.pdb_id,
                    pair_tokens=pair_tokens,
                )
                sel_tokens = pair_tokens[sel_tokens]
                other_chain_token_num = torch.sum(~torch.isin(sel_tokens, original_chain_token))
                if other_chain_token_num and torch.mean(torch.tensor(iptm)) >= 0.5:  # do select some tokens
                    # print('chain id, chain id2 +1', chain_id, chain_id2)
                    interact_num_chain[idx] += 1
                interact_num_token[idx] += other_chain_token_num
                sel_tokens_record[idx].append(sel_tokens)  # 1‑D list/array of global token indices
                priority_info_record[idx].append(priority_info)
                # sel_tokens = pair_tokens[self.select_tokens_for_chain_subunit_v4(
                #     chain_id=chain_id2,
                #     input_feature_dict=pair_feature_dict,
                #     is_resolved=is_resolved_pair,
                #     top_percent=0.5,
                #     gt_coordinates=pair_coordinates,
                #     sym=sym,
                #     pde=pde,
                #     gff3_dir='/home/u3590540/cxlu/protein/Protenix4/Protenix/Interpro_subunit',
                #     pdb_id=self.pdb_id,
                # )]
                sel_tokens, priority_info = self.select_tokens_for_chain_subunit_v4(
                    chain_id=chain_id2,
                    input_feature_dict=pair_feature_dict,
                    is_resolved=is_resolved_pair,
                    top_percent=0.5,
                    gt_coordinates=pair_coordinates,
                    sym=sym,
                    pde=pde,
                    gff3_dir='/home/u3590540/cxlu/protein/Protenix4/Protenix/Interpro_subunit',
                    pdb_id=self.pdb_id,
                    pair_tokens=pair_tokens,
                )
                sel_tokens = pair_tokens[sel_tokens]

                original_chain_token_chainid2 = torch.where(asym_id == chain_id2)[0]
                other_chain_token_num = torch.sum(~torch.isin(sel_tokens, original_chain_token_chainid2))
                if other_chain_token_num and torch.mean(torch.tensor(iptm)) >= 0.5:  # do select some tokens
                    # print('chain id2, chain id +1', chain_id, chain_id2)
                    interact_num_chain[idx2] += 1
                interact_num_token[idx2] += other_chain_token_num
                # print('chain id2, + tokens', chain_id, chain_id2, other_chain_token_num)
                sel_tokens_record[idx2].append(sel_tokens)  # 1‑D list/array of global token indices
                priority_info_record[idx2].append(priority_info)
                del pair_pred_dict, pair_feature_dict, is_resolved_pair, pair_coordinates, pde
                torch.cuda.empty_cache()

            # print('pair finished')
            sel_tokens = sel_tokens_record[idx]
            if len(sel_tokens) == 0:
                sel_tokens.append(torch.where((asym_id == chain_id) | (is_ligand_tok == True))[0])
            sel_tokens = torch.cat(sel_tokens)
            sel_tokens, _ = torch.sort(torch.unique(sel_tokens))
            # max_tokens = (N_token - original_chain_token.shape[0]) * 3 // 4 + original_chain_token.shape[0]
            max_tokens = 2000
            if len(sel_tokens) > max_tokens:
                print(f"Chain {chain_id}: Clipping from {len(sel_tokens)} to {max_tokens} tokens, {original_chain_token.shape[0]} original tokens, {N_token} total tokens")
                sel_tokens = self.clip_tokens_to_max(sel_tokens, priority_info_record[idx], max_tokens=max_tokens)
                print(f"After clipping: {len(sel_tokens)} tokens")

            chain_tokens[chain_id] = sel_tokens
            sel_tokens2 = self.select_tokens_for_chain(
                chain_id=chain_id,
                input_feature_dict=input_feature_dict,
                is_resolved=is_resolved,
                gt_coordinates=gt_coordinates,
                sym=sym,
            )
            # sel_tokens = sel_tokens2
            # sel_tokens = torch.arange(0, N_token, device=device)
            total_other_chain_token = []
            for tmp_idx in range(len(polymer_ids)):
                if tmp_idx == idx:
                    continue
                chain_id_tmp = polymer_ids[tmp_idx]
                tmp_chain_tokens = torch.where(asym_id == chain_id_tmp)[0]
                if tmp_chain_tokens.shape[0] > 0:
                    total_other_chain_token.append(tmp_chain_tokens)
            if len(total_other_chain_token) == 0:
                total_other_chain_token = sel_tokens2
            else:
                total_other_chain_token = torch.cat(total_other_chain_token)
            precision, recall = self.recall_and_acc(sel_tokens[~torch.isin(sel_tokens, original_chain_token)],
                                                    sel_tokens2[~torch.isin(sel_tokens2, original_chain_token)])
            print('precision', precision, 'recall', recall,
                  sel_tokens[~torch.isin(sel_tokens, original_chain_token)].shape,
                  sel_tokens2[~torch.isin(sel_tokens2, original_chain_token)].shape, total_other_chain_token.shape[0])
            # sel_tokens = sel_tokens2
            chain_tokens[chain_id] = sel_tokens
            # build a mini feature dictf
            # from copy import deepcopy
            # input_feature_dict_origin = deepcopy(input_feature_dict)
            sub_feat = self.create_subset_input_feature_dict(
                input_feature_dict, sel_tokens
            )
            token_to_atom_index_subset = [[] for _ in range(sel_tokens.shape[0])]
            atom_to_token_index_subset = sub_feat["atom_to_token_idx"].tolist()
            for atom_idx, token_idx in enumerate(atom_to_token_index_subset):
                token_to_atom_index_subset[token_idx].append(atom_idx)
            torch.cuda.empty_cache()
            print('chain id:', chain_id, 'token num:', sel_tokens.shape[0])
            pred_dict, _, _ = self._main_inference_loop(
                input_feature_dict=sub_feat,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                cal_chain_based=False,
            )
            del pred_dict['contact_probs']
            del pred_dict['pae']
            del pred_dict['pde']
            summary_confidence_keys = deepcopy(list(pred_dict['summary_confidence'][0].keys()))
            for key in summary_confidence_keys:
                # print('summary confidence', key)
                if 'ranking_score' not in key:
                    for m in range(len(pred_dict['summary_confidence'])):
                        del pred_dict['summary_confidence'][m][key]
            full_data_keys = deepcopy(list(pred_dict['full_data'][0].keys()))
            for key in full_data_keys:
                if 'atom_plddt' not in key:
                    for m in range(len(pred_dict['full_data'])):
                        del pred_dict['full_data'][m][key]
                # print('full data', key)

            # return chain_results[0]['pred'], log_dsict, time_tracke
            ranking_score = 0.
            for i in range(N_sample):
                ranking_score = max(ranking_score, pred_dict["summary_confidence"][i]["ranking_score"])
            # print('chain id , final interacting chain num', chain_id, interact_num_chain[idx], interact_num_token[idx])
            chain_results.append(
                dict(chain_id=chain_id,
                     tokens=sel_tokens,
                     pred=pred_dict,
                     token_to_atom=token_to_atom_index_subset,
                     ranking_score=ranking_score,
                     interact_num_chain=interact_num_chain[idx],
                     interact_num_token=interact_num_token[idx])
            )
            # memory: free tensors that will not be used again
            del sub_feat, pred_dict
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------- #
        # 2.  Assemble the chains in descending order of confidence         #
        # ---------------------------------------------------------------- #
        # chain_results.sort(
        #     key=lambda d: d["ranking_score"].item(), reverse=True
        # )
        chain_results.sort(
            key=lambda d: d["interact_num_chain"] * 10000 + d['interact_num_token'] + d['ranking_score'].item(),
            reverse=True
        )
        chain_id_results_mapping = {}
        for i in range(len(chain_results)):
            chain_id_results_mapping[chain_results[i]["chain_id"]] = i
        # chain_results = [chain_results[1], chain_results[0], chain_results[2]]
        N_token = input_feature_dict["residue_index"].shape[-1]
        N_atoms = input_feature_dict['is_protein'].shape[0]
        assembled_coord = torch.full(
            (N_sample, N_atoms, 3), float("nan"),
            device=device, dtype=dtype
        )
        placed_mask = torch.zeros(N_token, dtype=torch.bool, device=device)
        plddt_logits_feat = 50
        assembled_plddt = torch.zeros((N_sample, N_atoms, plddt_logits_feat), device=device,
                                      dtype=chain_results[0]["pred"]["plddt"].dtype)
        assembled_atom_plddt = torch.zeros((N_sample, N_atoms), device=device,
                                           dtype=chain_results[0]["pred"]['full_data'][0]["atom_plddt"].dtype)
        assembled_resolved = torch.zeros((N_sample, N_atoms, 2), device=device, dtype=dtype)

        placed_chain_ids: List[int] = []
        # ligand candidates: lig_id → list[(mean_conf, local_tokens, coords[Ns,k,3])]
        ligand_candidates = defaultdict(list)
        # assembling the results
        for idx in range(len(chain_results)):
            info = chain_results[idx]
            ranking_scores = [info['pred']["summary_confidence"][i]["ranking_score"] for i in range(N_sample)]
            _, ranking_sorted_indice = torch.sort(torch.tensor(ranking_scores), descending=True)
            # ranking_sorted_indice = list(range(N_sample))
            cid = info["chain_id"]
            print('placing ', cid)
            tok_idx = info["tokens"]
            # if idx != 0 or idx != 1:
            #     continue
            tok_to_atom_idx_local = info['token_to_atom']
            coord_c = info["pred"]["coordinate"][ranking_sorted_indice]  # [Ns, m, 3]
            plddt_c = info["pred"]["plddt"][ranking_sorted_indice]  # [m]
            resolved_c = info["pred"]["resolved"]  # [m]
            atom_plddts = []
            for i in range(N_sample):
                atom_plddts.append(info["pred"]['full_data'][i]["atom_plddt"])
            atom_plddt = torch.stack(atom_plddts)[ranking_sorted_indice]
            # ---------------- transform ---------------------------------- #
            if idx == 0:  # first chain → identity
                coord_c_t = coord_c
            else:
                # anchors = non‑ligand tokens that are already placed
                placed_poly = (~is_ligand_tok[tok_idx]) & (placed_mask[tok_idx] & (asym_id[tok_idx] == cid))
                short_poly = False
                if placed_poly.shape[0] <= 40:
                    short_poly = True
                    placed_poly = (~is_ligand_tok[tok_idx]) & placed_mask[tok_idx]
                common_global_idx = tok_idx[placed_poly]
                common_global_atom_idx = []
                for i in common_global_idx:
                    common_global_atom_idx.extend(token_to_atom_index[i])
                common_global_atom_idx = torch.tensor(common_global_atom_idx, device=device)
                if common_global_idx.numel() >= 3:
                    # indices in the local array
                    local_anchor = placed_poly.nonzero(as_tuple=True)[0]
                    local_anchor_atom = []
                    for i in local_anchor:
                        local_anchor_atom.extend(tok_to_atom_idx_local[i])
                    local_anchor_atom = torch.tensor(local_anchor_atom, device=device)
                    coord_c_t = []
                    for i in range(N_sample):
                        # filtered by confidence
                        global_confidence = assembled_atom_plddt[i, common_global_atom_idx]
                        if short_poly:
                            confidence_threshold = torch.quantile(global_confidence.float(), 0.6)
                            high_conf_mask = global_confidence > confidence_threshold
                        else:
                            high_conf_mask = global_confidence > 0
                        common_global_atom_idx_high_conf = common_global_atom_idx[high_conf_mask]
                        local_anchor_atom_high_conf = local_anchor_atom[high_conf_mask]
                        P = coord_c[i, local_anchor_atom_high_conf, :]  # current
                        Q = assembled_coord[i, common_global_atom_idx_high_conf, :]  # assembled
                        R, t = kabsch(P, Q)
                        # P = coord_c[i, local_anchor_atom, :]  # current
                        # Q = assembled_coord[i, common_global_atom_idx, :]  # assembled
                        # confidence = assembled_atom_plddt[i, common_global_atom_idx]
                        # R, t = weighted_kabsch(P, Q, confidence / 100)
                        coord_c_t.append(apply_rigid(coord_c[i], R, t))
                    coord_c_t = torch.stack(coord_c_t)
                else:
                    print('not enough tokens for estimation')
                    coord_c_t = coord_c
            info["coord_transformed"] = coord_c_t  # cache
            # ---------------- write polymer coordinates ------------------ #
            # do NOT overwrite previously filled positions
            # write_mask = (~placed_mask[tok_idx]) & (~is_ligand_tok[tok_idx])  # polymer only, only write the current chain
            write_mask = (~placed_mask[tok_idx] | (asym_id[tok_idx] == cid)) & (
                ~is_ligand_tok[tok_idx])  # polymer only, only write the current chain
            # write_mask = (~is_ligand_tok[tok_idx])  # polymer only, only write the current chain
            # write_mask = torch.ones_like(tok_idx)  # polymer only, only write the current chain
            assert write_mask.shape == tok_idx.shape
            if write_mask.any():
                g_idx = tok_idx[write_mask.bool()]
                l_idx = write_mask.nonzero(as_tuple=True)[0]
                assert g_idx.shape == l_idx.shape
                g_idx_atom = []
                l_idx_atom = []
                for i in g_idx:
                    g_idx_atom.extend(token_to_atom_index[i])
                for i in l_idx:
                    l_idx_atom.extend(tok_to_atom_idx_local[i])
                g_idx_atom = torch.tensor(g_idx_atom, device=device)
                l_idx_atom = torch.tensor(l_idx_atom, device=device)
                for i in range(N_sample):
                    # high_conf_mask = (assembled_atom_plddt[i, g_idx_atom_all] < atom_plddt[i, l_idx_atom_all])
                    # g_idx_atom = g_idx_atom_all[high_conf_mask]
                    # l_idx_atom = l_idx_atom_all[high_conf_mask]
                    assembled_coord[i, g_idx_atom, :] = coord_c_t[i, l_idx_atom, :]
                    assembled_plddt[i, g_idx_atom, :] = plddt_c[i, l_idx_atom, :]
                    assembled_resolved[i, g_idx_atom] = resolved_c[i, l_idx_atom]
                    assembled_atom_plddt[i, g_idx_atom] = atom_plddt[i, l_idx_atom]
                placed_mask[g_idx] = True
            placed_chain_ids.append(cid)

            # ---------------- collect ligand candidates ------------------ #
            lig_local_mask = is_ligand_tok[tok_idx].bool()
            if lig_local_mask.any():
                print('placing ligand')
                lig_global_ids = asym_id[tok_idx][lig_local_mask]
                # each ligand asym_id appears only once in the local subset
                for lg in torch.unique(lig_global_ids):
                    mask_lg = (asym_id[tok_idx] == lg)
                    ligand_global_tokens = tok_idx[mask_lg]
                    ligand_global_atoms = []
                    for i in ligand_global_tokens:
                        ligand_global_atoms.extend(token_to_atom_index[i])
                    ligand_global_atoms = torch.tensor(ligand_global_atoms, device=device)
                    ligand_local_tokens = mask_lg.nonzero(as_tuple=True)[0]
                    ligand_local_atoms = []
                    for i in ligand_local_tokens:
                        ligand_local_atoms.extend(tok_to_atom_idx_local[i])
                    ligand_local_atoms = torch.tensor(ligand_local_atoms, device=device)
                    mean_conf = atom_plddt[:, ligand_local_atoms].mean().item()
                    print('mean conf', mean_conf)
                    coords_lg = coord_c_t[:, ligand_local_atoms, :]
                    plddt_lg = plddt_c[:, ligand_local_atoms, :]
                    plddt_lg_atom = atom_plddt[:, ligand_local_atoms]
                    ligand_candidates[int(lg.item())].append((mean_conf, ligand_global_atoms, ligand_global_tokens,
                                                              coords_lg, plddt_lg, plddt_lg_atom,
                                                              resolved_c[:, ligand_local_atoms]))

        # ---------------------------------------------------------------- #
        #  (4) choose the best copy for every ligand                       #
        # ---------------------------------------------------------------- #
        for lg_id, copies in ligand_candidates.items():
            # pick max by mean_conf
            best_conf, best_atoms, best_tok, best_coord, best_plddt, best_plddt_atom, best_resolved = max(copies,
                                                                                                          key=lambda x:
                                                                                                          x[0])
            assembled_coord[:, best_atoms, :] = best_coord
            assembled_plddt[:, best_atoms] = best_plddt
            placed_mask[best_tok] = True
            assembled_resolved[:, best_atoms] = best_resolved
            assembled_atom_plddt[:, best_atoms] = best_plddt_atom

        pred_dict = {
            "coordinate": assembled_coord,  # [N_s, N_token, 3]
            "plddt": assembled_plddt,  # [N_token]
            "pae": torch.full((N_atoms, N_atoms), float("nan"),
                              device=device, dtype=dtype),
            "pde": torch.full((N_atoms, N_atoms), float("nan"),
                              device=device, dtype=dtype),

            "resolved": (assembled_atom_plddt > 0.5).float(),  # crude
            "contact_probs": torch.full((N_token, N_token), float("nan"),
                                        device=device, dtype=dtype),
        }
        pred_dict["summary_confidence"] = [{'ranking_score': torch.mean(assembled_atom_plddt[i])} for i in
                                           range(N_sample)]
        # pred_dict["summary_confidence"] = [{'ranking_score': torch.tensor(float(1 - i/10 + 0.1))} for i in
        #                                    range(N_sample)]
        pred_dict["full_data"] = [{"atom_plddt": assembled_atom_plddt[i]} for i in range(N_sample)]
        # assert torch.sum(pred_dict['coordinate'] !=chain_results[0]['pred']['coordinate'])==0
        # # assert torch.sum()
        # for i in range(N_sample):
        #     assert torch.sum(pred_dict['full_data'][i]['atom_plddt'] !=chain_results[0]["pred"]['full_data'][i]["atom_plddt"])==0
        log_dict = {}
        time_tracker = {"total": time.time() - t0}

        return pred_dict, log_dict, time_tracker

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        cal_chain_based=True,
        consistency=False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["token_index"].shape[-1]

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        if mode == "inference":
            keys_to_delete = []
            for key in input_feature_dict.keys():
                if "template_" in key or key in [
                    "msa",
                    "has_deletion",
                    "deletion_value",
                    "profile",
                    "deletion_mean",
                    "token_bonds",
                ]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del input_feature_dict[key]
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        cache = dict()
        use_cache = self.enable_diffusion_shared_vars_cache and "d_lm" in input_feature_dict
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
        if consistency:
            pred_dict["coordinate"] = self.sample_consistency(consistency_net=self.consistency_contact,
                                                              inference_scheduler=self.inference_noise_scheduler,
                                                              ts=[0, 66, 132, 199],  p_lm=None, c_l=None, pair_z=None,
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

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]

        # Confidence logits
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

        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        time_tracker.update({"model_forward": time.time() - step_st})

        # Permutation: when label is given, permute coordinates and other heads
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            last_step_seconds = step_confidence
            time_tracker.update({"permutation": time.time() - last_step_seconds})

        # Summary Confidence & Full Data
        # Computed after coordinates and logits are permuted
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get("interested_ligand_mask", None)
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
            cal_chain_based=cal_chain_based,
        )

        return pred_dict, log_dict, time_tracker

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

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
        gt_coordinates: torch.Tensor = None,
        is_resolved: torch.Tensor = None,
        pdb_id=None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        self.pdb_id = pdb_id
        # self.pdb_id = None
        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None
        # input_feature_dict = self.relative_position_encoding.generate_relp(
        #     input_feature_dict
        # )
        # input_feature_dict = update_input_feature_dict(input_feature_dict)
        # print('starting:', input_feature_dict['d_lm'].shape, 'tokens', input_feature_dict['token_index'].shape)
        # print('starting:', input_feature_dict['v_lm'].shape, 'atoms', input_feature_dict['ref_pos'].shape, input_feature_dict['ref_space_uid'].shape)
        # print('starting: asym ids', input_feature_dict['asym_id'].shape)
        # print('starting: ref pos:', input_feature_dict['relp'].shape)
        # print('pdb_id', pdb_id)

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
                gt_coordinates=gt_coordinates,
                is_resolved=is_resolved,
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
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
