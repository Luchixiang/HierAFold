"""
Final precise diff: find every line that differs between the two selection
functions and outer loops, ignoring whitespace/comments/variable renames.
"""
import re

cp = open('protenix/model/HiraryFold_cp.py').readlines()
cl = open('protenix/model/HierAFold.py').readlines()

def get_method(lines, name):
    out, start, base_indent = [], None, None
    for i, line in enumerate(lines):
        if re.match(rf'    def {name}\s*\(', line):
            start = i; base_indent = 4; out = [line]; continue
        if start is not None:
            if line.strip() == '': out.append(line); continue
            ind = len(line) - len(line.lstrip())
            if ind <= base_indent and line.strip().startswith('def '): break
            out.append(line)
    return start, out

def normalise(line):
    """Strip comments, whitespace, rename variables to canonical names."""
    line = re.sub(r'#.*', '', line)
    line = line.strip()
    # rename cp->clean variable names
    line = line.replace('selected_chain_tokens', 'CTX_TOKENS')
    line = line.replace('context_selected_tokens', 'CTX_TOKENS')
    line = line.replace('distance_selected_tokens', 'DIST_TOKENS')
    line = line.replace('other_chain_tokens', 'OTHER_TOKENS')
    line = line.replace('other_resolved_tokens', 'OTHER_TOKENS')
    line = line.replace('chain_tokens_isresolved', 'QUERY_TOKENS')
    line = line.replace('query_resolved_tokens', 'QUERY_TOKENS')
    line = line.replace('pair_tokens[', 'GLOBAL[')
    line = line.replace('global_token_indices[', 'GLOBAL[')
    line = line.replace('pde_tmp', 'PAE_DISCRETE')
    line = line.replace('pae_discrete', 'PAE_DISCRETE')
    line = line.replace('pde_subunit', 'PAE_DOMAIN')
    line = line.replace('pae_domain', 'PAE_DOMAIN')
    line = line.replace('pde_score', 'PAE_SCORE')
    line = line.replace('pae_score', 'PAE_SCORE')
    line = line.replace('pde_selected_token', 'PAE_SEL')
    line = line.replace('pae_selected', 'PAE_SEL')
    line = line.replace('pde_tmp_quantile', 'PAE_QUANTILE')
    line = line.replace('pae_quantile', 'PAE_QUANTILE')
    line = line.replace('pde_tmp_own_chain', 'PAE_INTRA')
    line = line.replace('pae_intra', 'PAE_INTRA')
    line = line.replace('pde_subunit_percentile', 'PAE_DOM_SCORE')
    line = line.replace('pae_domain_score', 'PAE_DOM_SCORE')
    line = line.replace('selected_chain_tokens_subunit', 'DOM_KEEP')
    line = line.replace('domain_tokens_to_keep', 'DOM_KEEP')
    line = line.replace('large_subunits_candidates', 'SOFT_CANDS')
    line = line.replace('soft_candidates', 'SOFT_CANDS')
    line = line.replace('subunit_token', 'DOM_TOKENS')
    line = line.replace('domain_tokens', 'DOM_TOKENS')
    line = line.replace('subunit_centers', 'DOM_CENTERS')
    line = line.replace('domain_centers', 'DOM_CENTERS')
    line = line.replace('subunit_dists', 'DOM_DISTS')
    line = line.replace('domain_dists', 'DOM_DISTS')
    line = line.replace('min_subunit_dist', 'MIN_DOM_DIST')
    line = line.replace('min_domain_dist', 'MIN_DOM_DIST')
    line = line.replace('subunit[0]:subunit[1]', 'SPAN')
    line = line.replace('span[0]: span[1]', 'SPAN')
    line = line.replace('span[0]:span[1]', 'SPAN')
    line = line.replace('chain_center_coords', 'QUERY_COORDS')
    line = line.replace('query_center_coords', 'QUERY_COORDS')
    line = line.replace('chain_tokens_isresolved', 'QUERY_TOKENS')
    line = line.replace('for subunit in subunits', 'for SPAN_ITER in subunits')
    line = line.replace('for span in subunits', 'for SPAN_ITER in subunits')
    line = line.replace('pae_cross', 'PAE_CROSS')
    line = line.replace('pde_tmp', 'PAE_DISCRETE')  # catch remainder
    line = re.sub(r'\s+', ' ', line)
    return line

# Compare the two selection functions
cp_sel_start, cp_sel = get_method(cp, 'select_tokens_for_chain_subunit_v4')
cl_sel_start, cl_sel = get_method(cl, 'select_context_tokens')

cp_norm = [(i, normalise(l), l.rstrip()) for i, l in enumerate(cp_sel) if normalise(l)]
cl_norm = [(i, normalise(l), l.rstrip()) for i, l in enumerate(cl_sel) if normalise(l)]

cp_set = {n for _, n, _ in cp_norm}
cl_set = {n for _, n, _ in cl_norm}

print("="*70)
print("Lines in CP select func with NO equivalent in CLEAN (after normalisation):")
print("="*70)
for i, n, raw in cp_norm:
    if n not in cl_set:
        print(f"  CP L{cp_sel_start+1+i}: {raw}")

print()
print("="*70)
print("Lines in CLEAN select func with NO equivalent in CP (after normalisation):")
print("="*70)
for i, n, raw in cl_norm:
    if n not in cp_set:
        print(f"  CL L{cl_sel_start+1+i}: {raw}")

# Now compare the outer inference loops
print()
print("="*70)
print("OUTER LOOP DIFFS")
print("="*70)
cp_out_start, cp_out = get_method(cp, '_main_inference_loop_large_auc2')
cl_out_start, cl_out = get_method(cl, '_hierarchical_inference_loop')

cp_out_norm = [(i, normalise(l), l.rstrip()) for i, l in enumerate(cp_out) if normalise(l)]
cl_out_norm = [(i, normalise(l), l.rstrip()) for i, l in enumerate(cl_out) if normalise(l)]
cp_out_set = {n for _, n, _ in cp_out_norm}
cl_out_set = {n for _, n, _ in cl_out_norm}

print("\nLines in CP outer loop with NO equivalent in CLEAN:")
for i, n, raw in cp_out_norm:
    if n not in cl_out_set:
        print(f"  CP L{cp_out_start+1+i}: {raw}")

print("\nLines in CLEAN outer loop with NO equivalent in CP:")
for i, n, raw in cl_out_norm:
    if n not in cp_out_set:
        print(f"  CL L{cl_out_start+1+i}: {raw}")

