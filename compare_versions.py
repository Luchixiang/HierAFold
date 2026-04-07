import re

with open('protenix/model/HiraryFold_cp.py') as f:
    cp = f.read()
with open('protenix/model/HierAFold.py') as f:
    cl = f.read()

cp_funcs = re.findall(r'    def (\w+)\s*\(', cp)
cl_funcs = re.findall(r'    def (\w+)\s*\(', cl)
print('=== CP methods ===')
for f in cp_funcs: print(' ', f)
print('=== CLEAN methods ===')
for f in cl_funcs: print(' ', f)
print()
print('In CP but not CLEAN:', set(cp_funcs) - set(cl_funcs))
print('In CLEAN but not CP:', set(cl_funcs) - set(cp_funcs))

