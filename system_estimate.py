import numpy as np
from filterpy.kalman import KalmanFilter
from system_dynamics import non_linearized_model, measurement, UKF_model, UKF_measurement
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise

class KF_estimate(object):
    """ KF_estimate """
    def __init__(self, env, A, B, H, x_0, R, P_0, Q):
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
        self.env = env

# estimate states from input of measurement
    def state_estimate(self, force, y):
        # update estimation from kalman filter
        # Kalman filter
        self.KF.predict(u=force)        
        self.KF.update(y)

        # return updated estimated state
        return self.KF.x
class EKF_estimate(object):
    """ KF_estimate """
    def __init__(self, env, A, B, H, x_0, R, P_0, Q):
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
        self.env = env

# estimate states from input of measurement
    def state_estimate(self, force, y):
        # update estimation from kalman filter
        # Kalman filter
        # self.KF.predict(u=force)
        # Extended Kalman filter
        self.KF.x = non_linearized_model(self.env, self.KF.x, force)
        self.KF.P = self.KF._alpha_sq * np.dot(np.dot(self.KF.F, self.KF.P), self.KF.F.T) + self.KF.Q

        # save prior
        self.KF.x_prior = self.KF.x.copy()
        self.KF.P_prior = self.KF.P.copy()
        
        self.KF.update(y)

        # return updated estimated state
        return self.KF.x

class UKF_estimate(object):
    """ UKF_estimate """
    def __init__(self, env, dim_x, dim_y, X_0, P, Q, R):
        # self.sigmas = JulierSigmaPoints(dim_x, alpha=.1, beta=2., kappa=1.)
        self.sigmas = JulierSigmaPoints(dim_x)
        self.env = env
        self.ukf = UnscentedKalmanFilter( dim_x=dim_x, dim_z=dim_y, dt=env.tau, hx=UKF_measurement, 
                                          fx=UKF_model, points=self.sigmas)
        self.ukf.x = X_0[:,0]
        self.ukf.P = np.copy(P)
        self.ukf.R = np.copy(R)
        self.ukf.Q = np.copy(Q)

# estimate states from input of measurement
    def state_estimate(self, force, y):
        # update estimation from kalman filter
        # Kalman filter
        # self.KF.predict(u=force)
        # Extended Kalman filter
        self.ukf.predict(env=self.env, u=force)
        self.ukf.update(y)
        
        # print(self.ukf.x)
        # return updated estimated state
        return self.ukf.x
