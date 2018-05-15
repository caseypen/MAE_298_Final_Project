import gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from cartpole_util import CartPoleEnv
from system_dynamics import linearized_model_control, linearized_model_estimate
from system_estimate import KF_estimate
from linear_quadratic_regulator import lqr
import scipy.io as sio

# env = gym.make('CartPole-v0').env
env = CartPoleEnv() 
# set random seed
env.seed(1)

# # x, xdot, theta, thetadot
x = env.reset()
x_0 = np.copy(x).reshape(4,1) # this is observed without error

# linearized model for lqr controller
F = linearized_model_control(env)
A, B, H = linearized_model_estimate(env)

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

""" State estimator """
# contruct state estimator
Q = np.zeros((4, 4)) # assume that there are no process error
R = np.array([[2e-4, 0],
              [0,    2e-4]])
# assume accurate initial variance of states
P_0 = np.eye(4) 
estimator = KF_estimate(A, B, H, x_0, R, P_0, Q)

frame = 0
done = False

X = []
Est_X = []
U = []
TH = []
Est_TH = []
time = []
# save for mat file
mat_data = {}
states_actual = []
estimated_states = []
inputs = []
measurements = []
states_actual.append(x_0)

while 1:
    # calculate input value
    if frame is 0:
        ut = controller.input_design(x)
    else:
        ut = controller.input_design(estimate_x)
        # ut = controller.input_design(x)
    # ut = controller.input_design(x)
    # execute the force in simulated environment
    x = env.execute(ut)
    # get measurement of states
    y = env.sensor_measurement(x)

    estimate_x = estimator.state_estimate(ut[0,0], y)
    print("estimated x", estimate_x)
    print("actual x", x)
    # for plotting
    X.append(x[0])
    Est_X.append(estimate_x[0,0])
    TH.append(x[2])
    Est_TH.append(estimate_x[2,0])
    # U.append(ut)

    # saving for mat file
    # states_actual.append(x) 
    # estimated_states.append(estimate_x)
    # measurements.append(y)

    frame += 1
    time.append(frame*env.tau)
    env.render()
    # sleep(2)
    if frame > 300:
        break

# print(frame)
# mat_data["time"] = time
# mat_data["states of ground truth"] = states_actual
# mat_data["state measurements"] = measurements
# mat_data["estimated states"] = estimated_states
# mat_data["inputs of lqr"] = U
# sio.savemat('./data.mat', mat_data)

plt.subplot(2,1,1)
plt.plot(time, X,  '--',label="cart position")
plt.plot(time, Est_X, 'y-',label='Estimated position')
plt.title('cart position')
plt.grid()

plt.subplot(2,1,2)
plt.plot(time, TH, '--',label='pendulum angle')
plt.plot(time, Est_TH,'y-',label='Estimated angle')
plt.title('angle of pole')
plt.grid()

plt.show()