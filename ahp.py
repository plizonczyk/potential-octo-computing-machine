from configparser import ConfigParser
import ipdb

import numpy as np
import pandas as pd


class AHP(object):
    def __init__(self, filename='default.conf'):
        self.cfg = ConfigParser()
        self.cfg.read(filename)
        self.criteria = pd.read_csv(self.cfg.get('filenames', 'criteria'), header=None, dtype=np.float).as_matrix()
        self.names = pd.read_csv(self.cfg.get('filenames', 'alternatives'), header=None, dtype=np.str).as_matrix()[0]
        self.judgements = self._read_judgements()
        self.precision = self.cfg.getfloat('parameters', 'precision')

    def _read_judgements(self):
        judgements = {}
        for name in self.cfg['judgements']:
            judgements[name] = pd.read_csv(self.cfg.get('judgements', name), header=None, dtype=np.float).as_matrix()
        return judgements
