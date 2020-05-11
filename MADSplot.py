import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


class DeterministicDFO():

    def __init__(self, alpha, beta, gamma, x0, ncols=2, nrows=2, ms=5):
        """ Global parameters """
        self.pb_dim = np.shape(x0)[0]
        self.x0 = x0
        self.alpha = alpha
        self.beta = beta
        self.ncols = ncols
        self.nrows = nrows
        self.gamma = gamma
        self.ms = ms

        self.plotf()

    def function(self, x):
        return np.sqrt(1 + x[0]**2) + np.sqrt(1 + x[1]**2)

    def generatePSS(self, dim):
        b1 = np.ones((dim, dim))
        b1[1, 1] = -b1[1, 1]
        b2 = -np.ones((dim, dim))
        b2[1, 1] = -b2[1, 1]
        basis = np.concatenate((np.eye(dim), -np.eye(dim), b1, b2), axis=0)
        return basis[..., np.newaxis]

    def plotf(self):
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.function([X, Y]).T
        self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols, sharex=True, sharey=True, figsize=(6, 6))
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.ax[i][j].contour(X, Y, Z, levels=50, cmap=plt.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), zorder=0)
                if(j == 0):
                    self.ax[i][j].set_ylabel("Y-coordinate")
                if(i == self.nrows - 1):
                    self.ax[i][j].set_xlabel("X-coordinate")

    def optimize(self, fitness):
        self.current_x = self.x0
        self.alphas = [self.alpha]
        count = 0
        for k in range(self.nrows):
            for l in range(self.ncols):
                count += 1
                if not k + l:
                    self.ax[k][l].plot(self.current_x[0], self.current_x[1], 'bo', label="Last point", markersize=self.ms, zorder=5)
                else:
                    self.ax[k][l].plot(self.current_x[0], self.current_x[1], 'bo', markersize=self.ms, zorder=5)
                self.PSS = self.generatePSS(self.pb_dim)
                current_f = self.function(self.current_x)[0]
                ite = False
                self.ax[k][l].plot([
                    self.alpha * self.PSS[4][0][0] + self.current_x[0][0],
                    self.alpha * self.PSS[5][0][0] + self.current_x[0][0],
                    self.alpha * self.PSS[6][0][0] + self.current_x[0][0],
                    self.alpha * self.PSS[7][0][0] + self.current_x[0][0],
                    self.alpha * self.PSS[4][0][0] + self.current_x[0][0],
                ], [
                    self.alpha * self.PSS[4][1][0] + self.current_x[1][0],
                    self.alpha * self.PSS[5][1][0] + self.current_x[1][0],
                    self.alpha * self.PSS[6][1][0] + self.current_x[1][0],
                    self.alpha * self.PSS[7][1][0] + self.current_x[1][0],
                    self.alpha * self.PSS[4][1][0] + self.current_x[1][0],
                ], 'k-o', alpha=0.5, markersize=2)
                if not k + l:
                    self.ax[k][l].plot([
                        self.alpha * self.PSS[0][0][0] + self.current_x[0][0],
                        self.alpha * self.PSS[2][0][0] + self.current_x[0][0],
                    ], [
                        self.alpha * self.PSS[0][1][0] + self.current_x[1][0],
                        self.alpha * self.PSS[2][1][0] + self.current_x[1][0],
                    ], 'k-o', alpha=0.5, label="Mesh", markersize=2)
                else:
                    self.ax[k][l].plot([
                        self.alpha * self.PSS[0][0][0] + self.current_x[0][0],
                        self.alpha * self.PSS[2][0][0] + self.current_x[0][0],
                    ], [
                        self.alpha * self.PSS[0][1][0] + self.current_x[1][0],
                        self.alpha * self.PSS[2][1][0] + self.current_x[1][0],
                    ], 'k-o', alpha=0.5, markersize=2)
                self.ax[k][l].plot([
                    self.alpha * self.PSS[1][0][0] + self.current_x[0][0],
                    self.alpha * self.PSS[3][0][0] + self.current_x[0][0],
                ], [
                    self.alpha * self.PSS[1][1][0] + self.current_x[1][0],
                    self.alpha * self.PSS[3][1][0] + self.current_x[1][0],
                ], 'k-o', alpha=0.5, markersize=2)
                # for i in self.PSS:
                #     self.ax[k][l].scatter(self.alpha * self.PSS[:, 0, 0] + self.current_x[0][0], self.alpha * self.PSS[:, 1, 0] + self.current_x[1][0], c='k', s=2)
                for i in self.PSS:
                    temp_x = self.current_x + self.alpha * i
                    if(self.function(temp_x)[0] < current_f):
                        self.current_x = temp_x
                        ite = True
                        break
                if(ite):
                    self.alpha = self.gamma * self.alpha
                else:
                    self.alpha = self.beta * self.alpha
                if not k + l:
                    self.ax[k][l].plot(self.current_x[0], self.current_x[1], 'ro', label="New point", markersize=self.ms, zorder=5)
                else:
                    self.ax[k][l].plot(self.current_x[0], self.current_x[1], 'ro', markersize=self.ms, zorder=5)
                self.ax[k][l].set_title("Iteration no." + str(count))
                self.ax[k][l].set(adjustable='box', aspect='equal')
                self.alphas += [self.alpha]
        self.fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.95), prop={'size': 9})


if __name__ == '__main__':
    Opti = DeterministicDFO(1, 0.5, 1.5, np.array([[5], [5]]), 5, 2)
    Opti.optimize(1e-5)
    plt.show()
