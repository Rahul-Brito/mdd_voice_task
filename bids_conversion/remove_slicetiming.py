#!/usr/bin/env python

from glob import glob
import json
from pathlib import Path
import os
import sys

root = Path(sys.argv[1])

funcs = root / 'sub-*' / '*' / 'func' / '*.json'

def remove_json_element(fl, ele):
    os.chmod(fl, 0o640)
    with open(fl) as fp:
        data = json.load(fp)

    data.pop(ele, None)

    with open(fl, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
    os.chmod(fl, 0o440)

# TODO: make sure top-level folders are also cleared
for fl in glob(str(funcs)):
    remove_json_element(fl, 'SliceTiming')

print("Complete")
