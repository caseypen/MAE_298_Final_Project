import numpy as np
from filterpy.kalman import KalmanFilter

class KF_estimate(object):
    """ KF_estimate """
    def __init__(self, A, B, H, x_0, R, P_0, Q):
        self.A = A
        self.B = B
        self.H = H
        self.KF = KalmanFilter(dim_x=4, dim_z=2, dim_u=1, compute_log_likelihood=True)
        self.KF.x = x_0 # initial state
        self.KF.F = np.copy(A) # transition matrix
        self.KF.B = np.copy(B) # control matrix
        self.KF.R = np.copy(R) # measurement noise
        self.KF.H = np.copy(H)
        self.KF.P = np.copy(P_0)
        self.KF.Q = np.copy(Q)

# estimate states from input of measurement
    def state_estimate(self, force, y):
        # update estimation from kalman filter
        self.KF.predict(u=force)
        self.KF.update(y)

        # return updated estimated state
        return self.KF.x