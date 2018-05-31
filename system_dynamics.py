import numpy as np
import gym
import math
from scipy import signal


def discrete_model(env):
    # linearized continuous model
    J = 1/12*(env.masspole*2)**2
    beta = env.masspole*env.masscart*env.length**2 + J*(env.masspole+env.masscart)
    ac = -(env.masspole**2*env.length**2*env.gravity)/beta
    bc = (J+env.masspole*env.length**2)/beta
    cc = env.masspole*env.length*env.gravity*(env.masscart+env.masspole)/beta
    dc = -env.masspole*env.length/beta

    A_c = np.array([[0,1,0,0],
              [0,0,ac,0],
              [0,0,0,1],
              [0,0,cc,0]])
    B_c = np.array([[0],
                    [bc],
                    [0],
                    [dc]])
    # C_c = np.array([[1,env.tau,0,0],
    #                 [0,1,0,0],
    #                [0,0,0,1]])
    # D_c = np.array([[0],
    #                 [0],
    #                [0]])
    C_c = np.array([[1,0,0,0]])
    D_c = np.array([[0]])
    # discrete linearized model
    sys = signal.StateSpace(A_c, B_c, C_c, D_c)
    discrete_sys = sys.to_discrete(env.tau)
    A = discrete_sys.A
    B = discrete_sys.B
    C = discrete_sys.C
    D = discrete_sys.D
    
    # print(A)
    # print(B)
    # print(C)
    # while True:
    #     pass
    return A, B, C, D

# linearized function around equilibrum state
# output is system dynamics
def linearized_model_control(env):

    gamma = (4.0 / 3.0 - env.masspole / env.total_mass)

    a = -env.gravity * env.masspole / (env.total_mass * gamma)
    b = (1.0 / env.total_mass * (1 + env.masspole / (env.total_mass * gamma)))
    c = env.gravity / (env.length * gamma)
    d = -1.0 / (env.total_mass * env.length * gamma)

    tau = env.tau
    F = np.array([
        [1, tau,       0,   0,       0],
        [0,   1, tau * a,   0, tau * b],
        [0,   0,       1, tau,       0],
        [0,   0, tau * c,   1, tau * d],
      ])


    return F

# linearized function around equilibrum state
# output is linearized state space model
def linearized_model_estimate(env):

    gamma = (4.0 / 3.0 - env.masspole / env.total_mass)

    a = -env.gravity * env.masspole / (env.total_mass * gamma)
    b = (1.0 / env.total_mass * (1 + env.masspole / (env.total_mass * gamma)))
    c = env.gravity / (env.length * gamma)
    d = -1.0 / (env.total_mass * env.length * gamma)

    tau = env.tau
    A = np.array([
        [1, tau,       0,   0],
        [0,   1, tau * a,   0],
        [0,   0,       1, tau],
        [0,   0, tau * c,   1],
      ])

    B = np.array([[ 0,
                    tau * b,
                    0,
                    tau * d
                    ]]).T
    # H = np.array([[1, env.tau, 0, 0],
    #               [0, 1, 0, 0],
    #               [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0]])
    # print(A)
    # print(B)
    # print(H)
    # while True:
    #     pass
    return A, B, H

def UKF_model(state, dt, **kwargs):

    env = kwargs["env"]
    force = kwargs["u"]

    x = state[0]
    x_dot = state[1]
    theta = state[2]
    theta_dot = state[3]
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + env.polemass_length * theta_dot * theta_dot * sintheta) / env.total_mass
    thetaacc = (env.gravity * sintheta - costheta* temp) / (env.length * (4.0/3.0 - env.masspole * costheta * costheta / env.total_mass))
    xacc  = temp - env.polemass_length * thetaacc * costheta / env.total_mass
    x  = x + env.tau * x_dot
    x_dot = x_dot + env.tau * xacc
    theta = theta + env.tau * theta_dot
    theta_dot = theta_dot + env.tau * thetaacc
    state_next = np.array([x,x_dot,theta,theta_dot])
    
    return state_next

def UKF_measurement(x):
    # x = np.array([x]).T
    # y = np.array([x[1,0]*0.02+x[0,0], x[1,0], x[3,0]])
    y = np.array([x[0]])
    # print("measurement",y.shape)
    return y

def non_linearized_model(env, state, u):

    force = u
    x = state[0]
    x_dot = state[1]
    theta = state[2]
    theta_dot = state[3]
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + env.polemass_length * theta_dot * theta_dot * sintheta) / env.total_mass
    thetaacc = (env.gravity * sintheta - costheta* temp) / (env.length * (4.0/3.0 - env.masspole * costheta * costheta / env.total_mass))
    xacc  = temp - env.polemass_length * thetaacc * costheta / env.total_mass
    x  = x + env.tau * x_dot
    x_dot = x_dot + env.tau * xacc
    theta = theta + env.tau * theta_dot
    theta_dot = theta_dot + env.tau * thetaacc
    state_next = np.array([x,x_dot,theta,theta_dot])
    
    return state_next

def measurement(x):
    
    # y = np.array([x[0,0]+x[1,0]*0.02, x[1,0], x[3,0]])
    y = np.array([x[0,0]])
    return y

