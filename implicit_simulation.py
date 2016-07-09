import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from gensim.models.word2vec import Word2Vec

from minerva import ClassificationMinerva


class ImplicitLearningSimulation(object):

    def __init__(self, model, dataframe):
        self.model = Word2Vec.load(model)
        self.df = pd.read_csv(dataframe)

    def process_datasets(self):
        raise NotImplementedError

    def run_simulation(self, nb_sims):
        raise NotImplementedError


class Williams2005(ImplicitLearningSimulation):

    def process_datasets(self):
        self.training = self.df[self.df['condition'] == 'training']
        self.testing = self.df[self.df['condition'] == 'testing']

        training_matrix = {det: self.model.syn0[
            [self.model.vocab[w].index
             for w in self.training[
                     self.training['determiner'] == det]['noun'].tolist()], :]
                           for det in 'gi ro ul ne'.split()}
        testing_matrix = {noun: self.model[noun]
                          for noun in self.testing['noun'].tolist()}
        self.training_matrix = training_matrix
        self.testing_matrix = testing_matrix

    def run_simulation(self, nb_sims):
        results = []

        for i in xrange(nb_sims):
            cm = ClassificationMinerva('cosine', 0.2)
            for det, matrix in self.training_matrix.iteritems():
                cm.add_category(matrix, det)
            test = cm.test(self.testing_matrix)

            correct = []
            incorrect = []

            for word, d in test.iteritems():
                correct.append(
                    d[self.testing[
                        self.testing['noun'] ==
                        word]['same_anim_diff_near'].tolist()[0]])
                incorrect.append(
                    d[self.testing[
                        self.testing['noun'] ==
                        word]['diff_anim_same_near'].tolist()[0]])
            correct = np.array(correct, ndmin=2)
            incorrect = np.array(incorrect, ndmin=2)
            correct[correct < 0] = 0
            incorrect[incorrect < 0] = 0
            raw = np.array([correct, incorrect]).T
            raw = raw[:, 0, :]
            raw /= raw.sum(axis=1)[:, np.newaxis]
            results.append(raw[:, 0])
        return results


class Paciorek(ImplicitLearningSimulation):

    def process_datasets(self):
        self.training = self.df[self.df['condition'] == 'training']
        self.testing = self.df[self.df['condition'] == 'testing']

        training_matrix = {det: self.model.syn0[
            [self.model.vocab[w].index
             for w in self.training[
                     self.training['determiner'] == det]['noun'].tolist()], :]
                           for det in 'gouble powter conell mouten'.split()}
        testing_matrix = {noun: self.model[noun]
                          for noun in self.testing['noun'].tolist()}
        self.training_matrix = training_matrix
        self.testing_matrix = testing_matrix

    def run_simulation(self, nb_sims):
        results = []

        for i in xrange(nb_sims):
            cm = ClassificationMinerva('circ_conv', 'cosine', 0.2)
            for det, matrix in self.training_matrix.iteritems():
                cm.add_category(matrix, det)
            test = cm.test(self.testing_matrix)
            correct = []
            incorrect = []

            for word, d in test.iteritems():
                correct.append(
                    d[self.testing[
                        self.testing['noun'] ==
                        word]['same_conc_diff_incr'].tolist()[0]])
                incorrect.append(
                    d[self.testing[
                        self.testing['noun'] ==
                        word]['diff_conc_same_incr'].tolist()[0]])
            # correct = np.array(correct, ndmin=2)
            # incorrect = np.array(incorrect, ndmin=2)
            # correct[correct < 0] = 0
            # incorrect[incorrect < 0] = 0
            correct = np.array(correct).clip(min=0)
            incorrect = np.array(incorrect).clip(min=0)
            raw = np.array([correct, incorrect]).T
            # raw = raw[:, 0, :]
            raw /= raw.sum(axis=1)[:, np.newaxis]
            results.append(raw[:, 0])
        return results


def runSimulation(dataset, dataClass,
                  corpus='/home/da/Projects/corpora/bnc_300',
                  nb_sims=100):
    sim = dataClass(corpus, dataset)
    sim.process_datasets()
    results = sim.run_simulation(nb_sims)
    results = np.vstack(results)
    return results

# williams2005 = runSimulation('/home/da/Projects/minerva/williams2005.csv',
#                              Williams2005)
paciorek2012a = runSimulation('/home/da/Projects/minerva/paciorek2012.csv',
                              Paciorek)
paciorek2012b = runSimulation('/home/da/Projects/minerva/paciorek2012b.csv',
                              Paciorek)

results = np.nanmean(np.vstack(paciorek2012a), axis=1).mean()
print 'paciorek 2012a : {}'.format(results)
results = np.nanmean(np.vstack(paciorek2012b), axis=1).mean()
print 'paciorek 2012b : {}'.format(results)
# hist = np.vstack(results).mean(axis=1)
# hist = hist[~np.isnan(hist)]
# plt.hist(hist)
# plt.show()
# print hist
