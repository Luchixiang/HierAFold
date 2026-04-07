"""
Pinpoint the exact algorithmic differences between the two versions.
"""
import re

with open('protenix/model/HiraryFold_cp.py') as f:
    cp_lines = f.readlines()
with open('protenix/model/HierAFold.py') as f:
    cl_lines = f.readlines()

def extract_method(lines, name):
    in_method = False
    method_lines = []
    start = None
    indent = None
    for i, line in enumerate(lines):
        if re.match(rf'\s+def {name}\s*\(', line):
            in_method = True
            indent = len(line) - len(line.lstrip())
            start = i
            method_lines = [line]
            continue
        if in_method:
            if line.strip() == '':
                method_lines.append(line)
                continue
            cur_indent = len(line) - len(line.lstrip())
            if cur_indent <= indent and line.strip().startswith('def '):
                break
            method_lines.append(line)
    return start, method_lines

# ============================================================
# DIFF 1: pde_tmp domain-scoring columns vs pae_cross columns
# ============================================================
print("="*70)
print("DIFF 1: How per-domain PAE score is computed")
print("="*70)

# CP: pde_tmp is [chain_q, other_chain] cross-chain slice
# Then pde_subunit = pde_tmp[:, subunit[0]:subunit[1]]
# The subunit spans come from get_domain_splits(pde_tmp_own_chain)
# where pde_tmp_own_chain = pde_tmp_BEFORE_reassignment[other_chain][:,other_chain]
# i.e. the intra-chain PAE from the FULL pair matrix

# CLEAN: pae_cross = pae_discrete[query][:,other]
# pae_domain = pae_cross[:, span[0]:span[1]]
# spans come from get_domain_splits(pae_intra)
# where pae_intra = pae_discrete[other][:,other]

# KEY: pde_tmp_own_chain uses pde_tmp = argmax(pde) where pde is the FULL pair PAE
#      pae_intra uses pae_discrete = argmax(pae) where pae is also the FULL pair PAE
# These should be identical. BUT:
# pde_tmp_own_chain is [N_other, N_other] sliced by token indices
# pae_intra is [N_other, N_other] sliced by token indices
# SAME.

# However there is a subtle issue: after domain detection, pde_tmp is REASSIGNED:
#   pde_tmp = torch.argmax(pde, dim=-1)       <- recomputed
#   pde_tmp = pde_tmp[chain_q][:, other_chain] <- cross-chain
# Then pde_subunit = pde_tmp[:, subunit[0]:subunit[1]]
# subunit[0]:subunit[1] are POSITIONS IN other_chain_tokens
# pde_tmp columns ARE other_chain_tokens -> correct

# In CLEAN:
# pae_cross = pae_discrete[query][:,other]
# pae_domain = pae_cross[:, span[0]:span[1]]
# span positions are POSITIONS IN other_resolved_tokens
# pae_cross columns ARE other_resolved_tokens -> correct
# SAME behavior.

print("Domain PAE scoring: IDENTICAL (both correctly slice cross-chain PAE by subunit positions)")

# ============================================================
# DIFF 2: The iptm availability key
# ============================================================
print("\n" + "="*70)
print("DIFF 2: iptm key availability from _main_inference_loop vs _single_chain_inference")
print("="*70)

# CP: calls _main_inference_loop(..., cal_chain_based=False, consistency=True)
# which calls compute_full_data_and_summary(..., cal_chain_based=False)
# Then accesses: pair_pred_dict['summary_confidence'][i]['iptm']

# CLEAN: calls _single_chain_inference(..., use_consistency=True, chain_based_confidence=False)
# which calls compute_full_data_and_summary(..., cal_chain_based=False)  <-- chain_based_confidence=False
# Then accesses: pair_pred["summary_confidence"][i]["iptm"]

# Let's check what keys compute_full_data_and_summary returns when cal_chain_based=False
import subprocess
result = subprocess.run(['grep', '-n', 'iptm', 'protenix/model/sample_confidence.py'],
                       capture_output=True, text=True)
print("iptm in sample_confidence.py:")
for line in result.stdout.split('\n')[:30]:
    if line: print(' ', line)

# ============================================================
# DIFF 3: Interaction counting (iptm threshold)
# ============================================================
print("\n" + "="*70)
print("DIFF 3: Interaction counting - same iptm >= 0.5 threshold")
print("="*70)
print("CP:    if other_chain_token_num and torch.mean(torch.tensor(iptm)) >= 0.5")
print("CLEAN: if other_token_count_i > 0 and mean_iptm >= 0.5")
print("IDENTICAL")

# ============================================================
# DIFF 4: Priority info token indices (pair_tokens vs global_token_indices)
# ============================================================
print("\n" + "="*70)
print("DIFF 4: priority_info stores pair_tokens[x] vs global_token_indices[x]")
print("="*70)
print("CP:    priority_info[key] = (score, pair_tokens[subunit_token], False)")
print("CLEAN: priority_info[key] = (score, global_token_indices[domain_tokens], False)")
print("pair_tokens in CP == pair_token_indices (global) == global_token_indices in CLEAN")
print("BUT: in CP subunit_token is LOCAL (index into pair_feature_dict tokens)")
print("     pair_tokens[subunit_token] = global indices -> CORRECT")
print("In CLEAN domain_tokens is LOCAL (index into pair_feat tokens)")
print("     global_token_indices[domain_tokens] = global indices -> CORRECT")
print("IDENTICAL for clip_tokens_to_max")

# ============================================================
# DIFF 5: What sel_tokens are appended in the outer loop
# ============================================================
print("\n" + "="*70)
print("DIFF 5: sel_tokens recorded BEFORE vs AFTER clip")
print("="*70)

cp_outer_start, cp_outer = extract_method(cp_lines, '_main_inference_loop_large_auc2')
cl_outer_start, cl_outer = extract_method(cl_lines, '_hierarchical_inference_loop')

# Find the sel_tokens_record/priority_info_record append lines
print("\nCP recording (before/after clip?):")
for i, line in enumerate(cp_outer):
    if 'sel_tokens_record' in line or 'priority_info_record' in line:
        print(f"  L{cp_outer_start+1+i}: {line}", end='')

print("\nCLEAN recording:")
for i, line in enumerate(cl_outer):
    if 'selected_token_sets' in line or 'priority_info_sets' in line:
        print(f"  L{cl_outer_start+1+i}: {line}", end='')

# ============================================================
# DIFF 6: Clipping in CP is AFTER accumulating all pairs for a chain,
#         CLEAN clips after accumulating all pairs for a chain too
#         BUT the priority_info passed to clip differs!
# ============================================================
print("\n" + "="*70)
print("DIFF 6: What priority_info is passed to clip_tokens_to_max")
print("="*70)
print("\nCP clip call:")
for i, line in enumerate(cp_outer):
    if 'clip_tokens_to_max' in line or 'priority_info_record' in line:
        # show context
        start_ctx = max(0, i-2)
        end_ctx = min(len(cp_outer), i+3)
        for j in range(start_ctx, end_ctx):
            print(f"  L{cp_outer_start+1+j}: {cp_outer[j]}", end='')
        print()

print("\nCLEAN clip call:")
for i, line in enumerate(cl_outer):
    if 'clip_tokens_to_max' in line or 'priority_info_sets' in line:
        start_ctx = max(0, i-2)
        end_ctx = min(len(cl_outer), i+3)
        for j in range(start_ctx, end_ctx):
            print(f"  L{cl_outer_start+1+j}: {cl_outer[j]}", end='')
        print()

# ============================================================
# DIFF 7: inner inference loop cal_chain_based for STAGE 2
# ============================================================
print("\n" + "="*70)
print("DIFF 7: Stage 2 inference - cal_chain_based parameter")
print("="*70)
print("\nCP Stage-2 call:")
for i, line in enumerate(cp_outer):
    if '_main_inference_loop(' in line or 'cal_chain_based' in line:
        start_ctx = max(0, i-1)
        end_ctx = min(len(cp_outer), i+8)
        for j in range(start_ctx, end_ctx):
            print(f"  L{cp_outer_start+1+j}: {cp_outer[j]}", end='')
        print()
        break

print("\nCLEAN Stage-2 call:")
for i, line in enumerate(cl_outer):
    if '_single_chain_inference(' in line and 'use_consistency=False' in ''.join(cl_outer[i:i+10]):
        start_ctx = max(0, i-1)
        end_ctx = min(len(cl_outer), i+10)
        for j in range(start_ctx, end_ctx):
            print(f"  L{cl_outer_start+1+j}: {cl_outer[j]}", end='')
        print()
        break

