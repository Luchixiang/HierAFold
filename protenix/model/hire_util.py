import numpy as np
import torch
from typing import List, Tuple


def parse_gff3_domains(gff3_path: str, sequence_length: int, replicate:int=1) -> List[Tuple[int, int]]:
    """
    Parse GFF3 file to extract domain regions and fill gaps.

    Args:
        gff3_path: Path to the GFF3 file
        sequence_length: Total length of the sequence

    Returns:
        List of tuples (start, end) representing subunits (0-indexed, end-exclusive)
    """
    import os
    sequence_length = sequence_length // replicate
    if not os.path.exists(gff3_path):
        # If file doesn't exist, return the whole sequence as one subunit
        if replicate <= 1:
            return [(0, sequence_length)]
        else:
            result = []
            for i in range(replicate):
                offset = i * sequence_length
                result.append((offset, offset + sequence_length))
            return result

    # Parse GFF3 file to get annotated regions
    annotated_regions = []
    with open(gff3_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            # Extract start and end (convert from 1-indexed to 0-indexed)
            start = int(parts[3]) - 1  # GFF3 is 1-indexed
            end = int(parts[4])  # end is inclusive in GFF3, but we want exclusive
            annotated_regions.append((start, end))

    if not annotated_regions:
        # No annotations found, return whole sequence
        if replicate <= 1:
            return [(0, sequence_length)]
        else:
            result = []
            for i in range(replicate):
                offset = i * sequence_length
                result.append((offset, offset + sequence_length))
            return result

    # Sort regions by start position
    annotated_regions.sort(key=lambda x: x[0])

    # Build complete subunit list including gaps
    subunits = []
    current_pos = 0

    for start, end in annotated_regions:
        # Add gap before this region if exists
        if start > current_pos:
            subunits.append((current_pos, start))

        # Add the annotated region
        subunits.append((start, end))
        current_pos = end

    # Add final gap if exists
    if current_pos < sequence_length:
        subunits.append((current_pos, sequence_length))
    if replicate <= 1:
        return subunits
    all_subunits = []
    for replica_idx in range(replicate):
        offset = replica_idx * sequence_length
        for start, end in subunits:
            all_subunits.append((start + offset, end + offset))
    return all_subunits

def get_domain_splits(pae_matrix: np.ndarray, split_threshold: float, min_domain_size: int = 20) -> List[int]:
    """
    Recursively splits a PAE matrix into domains based on inter-domain PAE.

    Args:
        pae_matrix: A square 2D NumPy array representing the Predicted Aligned Error matrix.
        split_threshold: The mean PAE value required between two potential domains to consider them separate.
        min_domain_size: The minimum number of residues a domain can have. A segment will not be split
                         if it would result in domains smaller than this size.

    Returns:
        A sorted list of integers representing the boundaries of the identified domains.
        For example, a return value of [0, 280, 490, 650] indicates three domains:
        0-280, 280-490, and 490-650.
    """
    n_residues = pae_matrix.shape[0]

    # Use a set to store the final boundary points to avoid duplicates
    boundaries = {0, n_residues}

    # Use a queue (as a list) to manage segments that need to be checked for potential splits
    segments_to_check = [(0, n_residues)]

    while segments_to_check:
        start, end = segments_to_check.pop(0)

        # A segment can only be split if it's at least twice the minimum domain size
        if end - start < 2 * min_domain_size:
            continue

        best_split_point = -1
        max_inter_domain_pae = -1

        # Iterate through all possible split points within the current segment
        # ensuring that the resulting sub-domains are not smaller than min_domain_size
        for split_point in range(start + min_domain_size, end - min_domain_size):

            # Calculate the mean PAE in the two off-diagonal blocks that represent
            # the interaction between the two potential new domains.
            # Block 1: pae_matrix[start:split_point, split_point:end]
            # Block 2: pae_matrix[split_point:end, start:split_point]
            # We take the average of both.
            pae_block1 = pae_matrix[start:split_point, split_point:end]
            pae_block2 = pae_matrix[split_point:end, start:split_point]

            # Ensure blocks are not empty before calculating mean
            if pae_block1.size == 0 or pae_block2.size == 0:
                continue

            inter_domain_pae = (np.mean(pae_block1) + np.mean(pae_block2)) / 2

            if inter_domain_pae > max_inter_domain_pae:
                max_inter_domain_pae = inter_domain_pae
                best_split_point = split_point

        # If the best split found has an inter-domain PAE above the threshold, accept the split
        if max_inter_domain_pae > split_threshold:
            boundaries.add(best_split_point)

            # Add the two new, smaller segments to the queue to check them for further splits
            segments_to_check.append((start, best_split_point))
            segments_to_check.append((best_split_point, end))
    initial_boundaries = sorted(list(boundaries))
    # # Step 2: Calculate intra-domain PAE for each subunit
    boundaries_to_remove = set()
    max_intra_pae = 10
    for i in range(len(initial_boundaries) - 1):
        start = initial_boundaries[i]
        end = initial_boundaries[i + 1]

        # Extract intra-domain PAE block (diagonal block)
        intra_pae_block = pae_matrix[start:end, start:end]
        mean_intra_pae = np.mean(intra_pae_block)

        # Mark boundaries for removal if intra-PAE is too high
        if mean_intra_pae > max_intra_pae:
            if i < len(initial_boundaries) - 2:  # Don't remove end boundary (n_residues)
                intra_pae_block_next = pae_matrix[initial_boundaries[i + 1]:initial_boundaries[i + 2],
                                       initial_boundaries[i + 1]:initial_boundaries[i + 2]]
                if np.mean(intra_pae_block_next) > max_intra_pae:
                    boundaries_to_remove.add(end)
    # Step 3: Remove boundaries and merge domains
    filtered_boundaries = [b for b in initial_boundaries if b not in boundaries_to_remove]
    return filtered_boundaries

def kabsch(P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Classic Kabsch (orthogonal Procrustes) that returns a 3×3 rotation R and
    1×3 translation t such that  Q ≈ (P @ Rᵀ) + t
    P, Q : [N, 3]       (N ≥ 3 recommended)
    Returns
        R : [3, 3]
        t : [1, 3]
    """
    with torch.amp.autocast(enabled=False, device_type="cuda", dtype=torch.float32):
        P_mean, Q_mean = P.mean(0, keepdim=True), Q.mean(0, keepdim=True)
        P_cent, Q_cent = P - P_mean, Q - Q_mean
        H = P_cent.T @ Q_cent
        H = H.float()
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        R = Vh.T @ U.T
        if torch.linalg.det(R) < 0:
            # reflection – flip last column of V
            Vh[-1, :] *= -1
            R = Vh.T @ U.T
    t = Q_mean - P_mean @ R.T
    return R, t


def apply_rigid(x: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x [..., 3];  R [3,3];  t [1,3] –> same shape as x"""
    return x @ R.T + t