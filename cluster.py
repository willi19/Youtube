import numpy as np
from matplotlib import pyplot as plt

class vector:
    def __init__():
        pass

class cluster:
    def __init__(self, pts):
        self._unit_vector = {}
        self.dim = pts.shape[0]
        self.pts = pts

    def _load_unit_vectors(self, n, division):
        if n in self._unit_vector.keys():
            return
        if n == 1:
            self._unit_vector[n] = np.ones((1))
            return
        
        deg = np.pi / division
        self._load_unit_vectors(n-1,division)
        len_n = self._unit_vector[n-1].shape[0]
        unit_vec = np.zeros((1+(division-1)*len_n,n))
        for i in range(0, division-1):
            unit_vec[len_n*i:len_n*(i+1),0] = np.cos(deg*(i+1))
            unit_vec[len_n*i:len_n*(i+1),1:] = self._unit_vector[n-1] * np.sin(deg*(i+1))
        unit_vec[-1,0] = 1
        self._unit_vector[n] = unit_vec

    def approximate(self, division=6):
        self._load_unit_vectors(self.dim, division)
        inner_prod = np.dot(self._unit_vector[self.dim],self.pts)
        
        self.range = [np.max(inner_prod,1),np.min(inner_prod,1)]

    def plot(self):
        plt.plot(self.pts[0], self.pts[1],'ro',ms=1)
        uv = self._unit_vector[self.dim]
        for i in range(len(self._unit_vector[self.dim])):
            slope = -uv[i][0]/(uv[i][1]+1e-9)
            plt.axline(self.range[0][i]*uv[i],slope = slope, color='black')
            plt.axline(self.range[1][i]*uv[i],slope = slope, color='black')
        plt.show()