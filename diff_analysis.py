"""
Deep diff of the two inference loops: _main_inference_loop_large_auc2 (cp)
vs _hierarchical_inference_loop (clean). Focus on every behavioural difference.
"""
import re

with open('protenix/model/HiraryFold_cp.py') as f:
    cp_lines = f.readlines()
with open('protenix/model/HierAFold.py') as f:
    cl_lines = f.readlines()

def extract_method(lines, name):
    """Return (start_lineno, list_of_lines) for a method."""
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

# Extract the two main inference methods
cp_start, cp_method = extract_method(cp_lines, '_main_inference_loop_large_auc2')
cl_start, cl_method = extract_method(cl_lines, '_hierarchical_inference_loop')

print(f"CP _main_inference_loop_large_auc2: line {cp_start+1}, {len(cp_method)} lines")
print(f"CLEAN _hierarchical_inference_loop: line {cl_start+1}, {len(cl_method)} lines")

# Also extract select functions
cp_sel_start, cp_sel = extract_method(cp_lines, 'select_tokens_for_chain_subunit_v4')
cl_sel_start, cl_sel = extract_method(cl_lines, 'select_context_tokens')

print(f"\nCP select_tokens_for_chain_subunit_v4: line {cp_sel_start+1}, {len(cp_sel)} lines")
print(f"CLEAN select_context_tokens: line {cl_sel_start+1}, {len(cl_sel)} lines")

# Extract _main_inference_loop (cp only) and _single_chain_inference (clean only)
cp_inner_start, cp_inner = extract_method(cp_lines, '_main_inference_loop')
cl_inner_start, cl_inner = extract_method(cl_lines, '_single_chain_inference')

print(f"\nCP _main_inference_loop: line {cp_inner_start+1}, {len(cp_inner)} lines")
print(f"CLEAN _single_chain_inference: line {cl_inner_start+1}, {len(cl_inner)} lines")

# Now do focused line-by-line diff of the selection functions
print("\n" + "="*80)
print("KEY DIFFS in selection logic")
print("="*80)

# Print all lines that mention key words in cp select method
keywords = ['pde_tmp', 'pde_subunit', 'pde_score', 'subunit', 'domain', 'quantile', 'cutoff', 'n_select', 'max_tokens', 'iptm']
print("\n--- CP select_tokens_for_chain_subunit_v4 key lines ---")
for i, line in enumerate(cp_sel):
    for kw in keywords:
        if kw in line:
            print(f"  L{cp_sel_start+1+i}: {line}", end='')
            break

print("\n--- CLEAN select_context_tokens key lines ---")
for i, line in enumerate(cl_sel):
    for kw in keywords:
        if kw in line:
            print(f"  L{cl_sel_start+1+i}: {line}", end='')
            break

# Print the outer inference loop comparisons
print("\n" + "="*80)
print("KEY DIFFS in outer inference loop")
print("="*80)
print("\n--- CP _main_inference_loop_large_auc2 key lines ---")
for i, line in enumerate(cp_method):
    for kw in ['max_tokens', 'iptm', 'interact', 'clip', 'MAX', 'consistency', 'cal_chain']:
        if kw in line:
            print(f"  L{cp_start+1+i}: {line}", end='')
            break

print("\n--- CLEAN _hierarchical_inference_loop key lines ---")
for i, line in enumerate(cl_method):
    for kw in ['max_tokens', 'iptm', 'interact', 'clip', 'MAX', 'consistency', 'cal_chain']:
        if kw in line:
            print(f"  L{cl_start+1+i}: {line}", end='')
            break

