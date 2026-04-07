"""
THE critical diff: priority_info_record[idx] in CP is a LIST of dicts
(one per pairwise partner), while priority_info_sets[idx] in CLEAN is also
a list of dicts. But clip_tokens_to_max iterates over them differently.

Also: in CP, priority_info passed for chain idx2 is the SAME priority_info
object returned by the chain_id2-perspective call. That is correct.

But the KEY bug: in CP the priority_info stored for idx2 is the priority_info
from the CHAIN_ID2-perspective call (select for chain_id2 as query),
but for idx it's the priority_info from the CHAIN_ID-perspective call.

In CLEAN: priority_i (chain_id_i perspective) stored for idx_i,
          priority_j (chain_id_j perspective) stored for idx_j.
SAME.

BUT WAIT: look at what clip_tokens_to_max receives:

CP:   self.clip_tokens_to_max(sel_tokens, priority_info_record[idx], ...)
      priority_info_record[idx] is a LIST of dicts

CLEAN: self.clip_tokens_to_max(context_tokens, priority_info_sets[idx], ...)
       priority_info_sets[idx] is also a LIST of dicts

Now look at clip_tokens_to_max:
"""

with open('protenix/model/HiraryFold_cp.py') as f:
    cp_lines = f.readlines()
with open('protenix/model/HierAFold.py') as f:
    cl_lines = f.readlines()

import re

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

cp_clip_start, cp_clip = extract_method(cp_lines, 'clip_tokens_to_max')
cl_clip_start, cl_clip = extract_method(cl_lines, 'clip_tokens_to_max')

print("CP clip_tokens_to_max body:")
for i, line in enumerate(cp_clip):
    print(f"  {cp_clip_start+1+i}: {line}", end='')

print("\n\nCLEAN clip_tokens_to_max body:")
for i, line in enumerate(cl_clip):
    print(f"  {cl_clip_start+1+i}: {line}", end='')

# Now the BIG one: in CP, sel_tokens_record[idx2] always gets the SAME
# priority_info as idx2 used when it was the NON-query chain in pair (idx, idx2).
# That priority_info has tokens for chain_id2 as QUERY — correct.
# BUT the priority_info key "target_..." has pair_tokens[chain_tokens]
# which are GLOBAL indices of chain_id2's tokens.
#
# The clip function uses these to decide which tokens to keep.
#
# The subtle CP bug: for chain idx2, priority_info comes from calling
# select_tokens_for_chain_subunit_v4(chain_id=chain_id2, ...)
# This returns priority_info with keys like "target_{chain_id2}",
# "chain_{other_id}_sub_..." etc.
# The tokens stored are pair_tokens[subunit_token] = GLOBAL indices.
#
# BUT THEN: sel_tokens = pair_tokens[sel_tokens]
# -> sel_tokens are global indices
# clip receives (sel_tokens_global, priority_info_with_global_tokens) -> CORRECT
#
# In CLEAN: sel_i are LOCAL indices, sel_i_global = pair_token_indices[sel_i]
# priority_i has global_token_indices[domain_tokens] = GLOBAL indices
# clip receives (context_tokens_global, priority_info_with_global_tokens) -> CORRECT
#
# KEY REMAINING QUESTION: does CP correctly handle the case where
# priority_info_record[idx] has MULTIPLE entries (one per pair)?
print("\n\nLooking at how priority_info_record is used in clip:")
cp_outer_start, cp_outer = extract_method(cp_lines, '_main_inference_loop_large_auc2')
for i, line in enumerate(cp_outer):
    if 'priority_info_record' in line or 'clip_tokens_to_max' in line:
        start_ctx = max(0, i-1)
        end_ctx = min(len(cp_outer), i+3)
        for j in range(start_ctx, end_ctx):
            print(f"  L{cp_outer_start+1+j}: {cp_outer[j]}", end='')
        print()

