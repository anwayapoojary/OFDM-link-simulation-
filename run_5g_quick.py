import runpy
from pathlib import Path
p = Path(r'd:\project 01\5glinksimulation.py')
src = p.read_text()
if 'N_symbols = 3000' in src:
    src = src.replace('N_symbols = 3000', 'N_symbols = 50')
else:
    print('Warning: could not find exact N_symbols pattern; running original script')

tmp = Path(r'd:\project 01\5glinksimulation_quick.py')
tmp.write_text(src)
runpy.run_path(str(tmp), run_name='__main__')
