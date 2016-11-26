from configparser import ConfigParser

import numpy as np
import pandas as pd


class AHP(object):
    def __init__(self, filename='default.conf'):
        self.cfg = ConfigParser()
        self.cfg.read(filename)
        self.names = pd.read_csv(self.cfg.get('filenames', 'alternatives'), header=None, dtype=np.str).as_matrix()[0]
        criteria = pd.read_csv(self.cfg.get('filenames', 'criteria'), dtype=np.float)
        self.criteria = criteria.as_matrix()
        self.criteria_names = list(criteria.columns)
        self.judgements = self._read_judgements()

        self.criteria_eig, self.judgements_eig, self.scores = None, None, None

    def _read_judgements(self):
        judgements = {}
        for name in self.cfg['judgements']:
            judgements[name] = pd.read_csv(self.cfg.get('judgements', name), header=None, dtype=np.float).as_matrix()
        return judgements

    def _calculate_eigenvectors(self):
        criteria_eig = self._get_eig(self.criteria)
        judgements_eig = {name: self._get_eig(matrix) for name, matrix in self.judgements.items()}
        return criteria_eig, judgements_eig

    def _get_eig(self, matrix):
        values, vectors = np.linalg.eig(matrix)
        abs_values = np.absolute(values)
        max_value_indice = np.argmax(abs_values)
        return np.real((vectors[:, max_value_indice] / sum(vectors[:, max_value_indice])))

    def _calculate_scores(self):
        scores = np.zeros(3)
        for i, name in enumerate(self.criteria_names):
            scores += self.judgements_eig[name] * self.criteria_eig[i]
        return {name: score for name, score in zip(self.names, scores)}

    def run(self):
        self.criteria_eig, self.judgements_eig = self._calculate_eigenvectors()
        self.scores = self._calculate_scores()
        print(self.scores)
