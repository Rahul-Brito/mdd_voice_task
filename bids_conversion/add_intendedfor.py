#!/usr/bin/env python

# Adopted from Gregory Ciccarelli's orginal script of the same name

from bids import BIDSLayout
from glob import glob
import os
import os.path as op
import json
import logging
import sys


ROOT = sys.argv[1]

if op.exists('intendedfor.log'):
    raise RuntimeError('Already generated - delete log to rerun')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='intendedfor.log',
                    filemode='w')

layout = BIDSLayout(ROOT)

# key: fieldmap run ID
# values: corresponding functional scans

fmap_key = {
    '1': layout.get_tasks(),
    '2': layout.get_tasks()
    }

def process_fieldmaps(fmaps, subj, session):
    for fmap in fmaps:
        run = fmap.split('_run-')[-1][:1]
        if run in fmap_key.keys():
            funcs = grab_funcs(fmap_key[run], subj, session)
        else:
            logging.warning('{} has incorrect number of runs, skipping. Fix and rerun'.format(subj))
            return
        if not funcs:
            return
        intended = {'IntendedFor': funcs}
        add_metadata(fmap, intended)

def grab_funcs(vals, subj, session):
    funcs = []
    for val in vals:
        funcs += glob(op.join(ROOT, 'sub-{}'.format(subj), 'ses-{}'.format(session), 'func', '*{}*nii.gz'.format(val)))
    if not funcs:
        logging.warning('No functionals found for {}'.format(subj))
        return

    return relpath(funcs, subj)

def relpath(funcs, subj):
    splitter = op.join(ROOT, 'sub-{}/'.format(subj))
    return [x.split(splitter)[-1] for x in funcs]

def load_json(filename):
    """ easy load of json dict """
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def add_metadata(fl, data, ind=4):
    """Adds dict items to exisiting json
    Parameters
    ----------
    fl : File (path to json file)
    data : dict (items to add)
    ind: indent amount for prettier print
    Returns
    ----------
    Metadata json
    """
    os.chmod(fl, 0o640)
    meta = load_json(fl)
    meta.update(data)
    with open(fl, 'wt') as fp:
        json.dump(meta, fp, indent=ind, sort_keys=True)
    os.chmod(fl, 0o440)

def main(subjs):
    for subj in subjs:
        print(subj)
        for session in layout.get_sessions(): 
            fmaps = layout.get(subject=subj, datatype = 'fmap', session=session, extension='.json', return_type='file')
            print(fmaps)
            process_fieldmaps(fmaps, subj, session)
            logging.info("Fieldmap intended for successfully added for {}".format(subj))

# run across all bids subjects
# TODO: add configparser
print('starting')
subjs = layout.get_subjects()
main(subjs)
