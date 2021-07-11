import sys

sys.path.append('../dataset_data/constants/')
from var_types import VAR_TYPES

out_str_order = ''

for idx, dsName in enumerate(VAR_TYPES.keys()):
    out_str_order += f'{idx},{dsName};'

exit(out_str_order)
