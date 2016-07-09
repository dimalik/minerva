import numpy as np
from scipy.spatial.distance import cosine, euclidean


class Minerva(object):

    def __init__(self, similarity_function, L_rate):
        self.L_rate = L_rate
        if similarity_function == 'cosine':
            self.sim_fun = self.cosine_similarity
        elif similarity_function == 'euclidean':
            self.sim_fun = self.euclidean

    @staticmethod
    def cosine_similarity(v1, v2):
        return 1 - cosine(v1, v2)

    @staticmethod
    def euclidean(v1, v2):
        return 1 / (1 + euclidean(v1, v2))

    @staticmethod
    def activation(sim_fun, probe, trace):
        return sim_fun(trace, probe) ** 3 * trace

    @staticmethod
    def conv_circ(signal, ker):
        return np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(ker)))

    @staticmethod
    def apply_forgetting(matrix, forgetting_rate):
        r, c = matrix.shape
        dropout = np.random.binomial(1, forgetting_rate,
                                     size=[r, c])
        return matrix * dropout


class ClassificationMinerva(Minerva):

    def __init__(self, method='addition', *args, **kwargs):
        super(ClassificationMinerva, self).__init__(*args, **kwargs)
        self.categories = {}
        self.intensities = {}
        if method not in ['addition', 'circ_conv']:
            raise TypeError('Only vector addition and circular' +
                            'convolution are supported')
        self.method = method

    def add_category(self, matrix, category_name=None):
        if not category_name:
            pass
        self.categories[category_name] = self.apply_forgetting(matrix,
                                                               self.L_rate)
        if self.method == 'circ_conv':
            intensity = np.random.uniform(-.05, .05, size=matrix.shape[1])
        else:
            intensity = np.zeros(matrix.shape[1])
        self.intensities[category_name] = intensity

    def test(self, testing_matrix, rho=1):
        ans = {}
        for word, probe in testing_matrix.iteritems():
            for category, matrix in self.categories.iteritems():
                for trace in matrix:
                    activation = self.activation(self.sim_fun, probe,
                                                 trace)
                    if self.method == 'circ_conv':
                        res = self.conv_circ(self.intensities[category],
                                             activation)
                    else:
                        res = activation
                    self.intensities[category] = res
            # abstraction
            for category, intensity in self.intensities.iteritems():
                self.intensities[category] *= (np.abs(intensity) <
                                               (rho * max(intensity)))
            ans[word] = {category: self.sim_fun(probe, intensity)
                         for category, intensity in
                         self.intensities.iteritems()}
        return ans
