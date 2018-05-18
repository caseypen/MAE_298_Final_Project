import numpy as np

class lqr(object):
    # linear quadratic regulator with MPC control
    # T is planning horizon
    # F is dynamic system
    # C is cost matrix
    # c is final state cost vector
    def __init__(self, T, F, C, c):
        self.F = F
        self.C = C
        self.c = c
        self.T = T
    # input design method outputs optimal control signal
    # given best estimation of current state
    def input_design(self, x):
        
        V = self.C[:4, :4]
        v = np.zeros((4))
        Ks = []
        for t in range(self.T, -1, -1):
            # Qt
            Qt = self.C + np.matmul(self.F.T, np.matmul(V, self.F))
            qt = self.c + np.matmul(self.F.T, v)

            Quu = Qt[-1:,-1:]
            Qux = Qt[-1:,:-1]
            Qxu = Qt[:-1, -1:]

            qu = qt[-1:]

            Qut_inv = np.linalg.inv(Quu)

            Kt = -np.matmul(Qut_inv, Qux)
            kt = -np.matmul(Qut_inv, qu)

            Ks.append((Kt, kt))

            V = Qt[:4, :4] + np.matmul(Qxu, Kt) + np.matmul(Kt.T, Qux) + np.matmul(Kt.T, np.matmul(Quu, Kt))
            v = qt[:4] + np.matmul(Qxu, kt) + np.matmul(Kt.T, qu) + np.matmul(Kt.T, np.matmul(Quu, kt))

            Kt, kt = Ks[-1]
            # ut = np.matmul(Kt, x.reshape((1, -1)).T) + kt
            ut = np.matmul(Kt, x.reshape((1, -1)).T) + kt
        
        return ut