#! /usr/local/bin/python3
import gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from cartpole_util import CartPoleEnv
import scipy.io as sio
from linear_quadratic_regulator import lqr
from system_dynamics import linearized_model_control

# env = gym.make('CartPole-v0').env
env = CartPoleEnv() 
# set random seed
env.seed(1)

# x, xdot, theta, thetadot
x = env.reset()
x_noise = np.copy(x)
x_noise = x+env.np_random.uniform(low=-0.2, high=0.2, size=(4,))

# gamma = (4.0 / 3.0 - env.masspole / env.total_mass)

# a = -env.gravity * env.masspole / (env.total_mass * gamma)
# b = (1.0 / env.total_mass * (1 + env.masspole / (env.total_mass * gamma)))
# c = env.gravity / (env.length * gamma)
# d = -1.0 / (env.total_mass * env.length * gamma)

# tau = env.tau
# F = np.array([
#     [1, tau,       0,   0,       0],
#     [0,   1, tau * a,   0, tau * b],
#     [0,   0,       1, tau,       0],
#     [0,   0, tau * c,   1, tau * d],
#   ])

# C = np.array([
#     [1,  0, 0,  0,   0],
#     [0,  0, 0,  0,   0],
#     [0,  0, 1,  0,   0],
#     [0,  0, 0,  0,   0],
#     [0,  0, 0,  0,   1],
#   ])

# # decide final position
# c = np.array([0, 0, 0, 0, 0]).T

# linearized model for lqr controller
F = linearized_model_control(env)

""" lqr controller """
# control design parameters
C = np.array([
    [1,  0, 0,  0,   0],
    [0,  0, 0,  0,   0],
    [0,  0, 1,  0,   0],
    [0,  0, 0,  0,   0],
    [0,  0, 0,  0,   1],
  ])
c = np.array([0, 0, 0, 0, 0]).T
T = 500
# construct controller
controller = lqr(T, F, C, c)

frame = 0
done = False
X = []
X_N = []
U = []
TH = []
TH_N = []
time = []
# save for mat file
mat_data = {}
states_no_noise = []
states_w_noise = []
inputs = []
states_no_noise.append(x)
states_w_noise.append(x_noise)

while 1:
    
    # Ks = []
    # T = 500
    # # V = np.zeros((4, 4))
    # # v = np.zeros((4))
    # V = C[:4, :4]
    # v = np.zeros((4))
    # for t in range(T, -1, -1):
    #     # Qt
    #     Qt = C + np.matmul(F.T, np.matmul(V, F))
    #     qt = c + np.matmul(F.T, v)


    #     Quu = Qt[-1:,-1:]
    #     Qux = Qt[-1:,:-1]
    #     Qxu = Qt[:-1, -1:]

    #     qu = qt[-1:]

    #     Qut_inv = np.linalg.inv(Quu)

    #     Kt = -np.matmul(Qut_inv, Qux)
    #     kt = -np.matmul(Qut_inv, qu)

    #     Ks.append((Kt, kt))

    #     V = Qt[:4, :4] + np.matmul(Qxu, Kt) + np.matmul(Kt.T, Qux) + np.matmul(Kt.T, np.matmul(Quu, Kt))
    #     v = qt[:4] + np.matmul(Qxu, kt) + np.matmul(Kt.T, qu) + np.matmul(Kt.T, np.matmul(Quu, kt))

    #     Kt, kt = Ks[-1]
    #     # ut = np.matmul(Kt, x.reshape((1, -1)).T) + kt
    #     ut = np.matmul(Kt, x_noise.reshape((1, -1)).T) + kt
    ut = controller.input_design(x)
            
    print (x, ut)
    print (ut)

    x = env.execute(ut)
    x_noise = x + env.np_random.uniform(low=-0.1, high=0.1, size=(4,))
    print(done)
    # for plotting
    X.append(x[0])
    X_N.append(x_noise[0])
    TH.append(x[2])
    TH_N.append(x_noise[2])
    U.append(ut)

    # saving for mat file
    states_no_noise.append(x) 
    states_w_noise.append(x_noise)

    frame += 1
    time.append(frame*env.tau)
    env.render()
    # sleep(2)
    if frame > 500:
        break
# print(frame)
mat_data["time"] = time
mat_data["states of ground truth"] = states_no_noise
mat_data["states with noise"] = states_w_noise
mat_data["inputs of lqr"] = U
sio.savemat('./data.mat', mat_data)

plt.subplot(2,1,1)
plt.plot(time, X,  '--',label="cart position")
plt.plot(time, X_N, 'y-',label='noisy cart position')
plt.title('cart position')
plt.grid()

plt.subplot(2,1,2)
plt.plot(time, TH, '--',label='pendulum angle')
plt.plot(time, TH_N,'y-',label='noisy pendulum angle')
plt.title('angle of pole')
plt.grid()

plt.show()