import numpy as np
import gym
import math
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
    H = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 1]])
    return A, B, H

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
    y = np.array([x[1,0], x[3,0]])

    return y