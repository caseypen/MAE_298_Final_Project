import numpy as np
import gym

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