#!/usr/bin/python
import gym
import numpy as np
from time import sleep
from cartpole_util import CartPoleEnv
from system_dynamics import linearized_model_control, linearized_model_estimate, discrete_model
from system_estimate import KF_estimate, EKF_estimate, UKF_estimate
from linear_quadratic_regulator import lqr
import argparse
import os
import scipy.io as sio
from math import pi

def main():
    # get parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator', '-est', type=str, default='KF')
    parser.add_argument('--store', action='store_true')
    parser.add_argument('--v_x_est', '-xest',type=float, default=2e-2)
    parser.add_argument('--v_xdot_est', '-vest',type=float, default=2e-2)
    parser.add_argument('--v_theta_est', '-thest',type=float, default=2e-2)
    parser.add_argument('--v_thetadot_est','-west', type=float, default=2e-2)
    parser.add_argument('--system_noise','-noise', type=float, default=2e-2)
    parser.add_argument('--starting_angle', '-angle', type=float, default=30)
    parser.add_argument('--measurements', '-mn', type=int, default=2)
    parser.add_argument('--frames', '-n', type=int, default=500)
    args = parser.parse_args()

    noise = args.system_noise
    env = CartPoleEnv(noise, pi*args.starting_angle/180, args.measurements) 
    # set random seed
    env.seed(1)

    # # x, xdot, theta, thetadot
    x = env.reset()
    x_0 = np.copy(x).reshape(4,1) # this is observed without error

    # linearized model for lqr controller
    F = linearized_model_control(env)
    # linearized model for state estimation
    A, B, H = linearized_model_estimate(env, args.measurements)
    # A, B, H, _ = discrete_model(env, args.measurements)
    
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

    # print(args.estimator == "EKF")
    """ State estimator """
    # contruct state estimator
    # Q = np.zeros((4, 4)) # assume that there are no process error
    Q = np.array([[args.v_x_est, 0,               0,                0],
                  [0,            args.v_xdot_est, 0,                0],
                  [0,            0,               args.v_theta_est, 0],
                  [0,            0,               0,                args.v_thetadot_est]])

    # assume that R is known from datasheet of sensors which are accurate
    # R = np.array([[2e-6, 0, 0],
    #               [0,  2e-6, 0],
    #               [0,    0, 2e-6]])
    if args.measurements is 1:
        R = np.array([[4e-4]])
    elif args.measurements is 2:
        R = np.array([[4e-4, 0],
                      [0,  4.4e-5]])
    # R = np.array([[2e-6, 0],
    #               [0,  2e-6]])
    # assume accurate initial variance of states
    P_0 = np.copy(Q) 
    # P_0 = np.zeros((4,4))
    dim_x = len(x)
    dim_y = R.shape[0]
    if args.estimator == "KF":
        estimator = KF_estimate(env, A, B, H, x_0, R, P_0, Q)
    elif args.estimator == "EKF":
        estimator = EKF_estimate(env, A, B, H, x_0, R, P_0, Q)
    elif args.estimator == "UKF":
        estimator = UKF_estimate(env, dim_x, dim_y, x_0, P_0, Q, R, args.measurements)

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
    states_actual.append(x)
    ut_prev = 0
    delta_f_max = 10
    epsilon = 0.1

    while 1:

        # calculate input value
        if frame is 0:
            ut_desired = controller.input_design(x)
        else:
            ut_desired = controller.input_design(estimate_x)
            # ut = controller.input_design(x)
        ut_desired = ut_desired[0,0]
        # print (ut_desired)
        while True:
            # ut = controller.input_design(x)
            # execute the force in simulated environment
            delta_f = abs(ut_desired - ut_prev)
            if delta_f > delta_f_max:
                ut = ut_prev + delta_f_max*np.sign(ut_desired - ut_prev)
                ut_prev = ut
            else:
                ut = ut_desired            
            # print (ut)
            
            ut = np.atleast_2d(ut)
            x = env.execute(ut)
            # introduce noise of input
            ut += env.np_random.normal(0, args.system_noise)
            # ut += env.np_random.normal(0, args.system_noise+(delta_f*0.12)**2)
            # get measurement of states
            y = env.sensor_measurement(x)

            estimate_x = estimator.state_estimate(ut[0,0], y)
            print("estimated x", estimate_x.T)
            print("actual x", x)
            
            if args.estimator == "UKF":
                estimate_x = np.atleast_2d(estimate_x).T

            # for plotting
            X.append(x[0])
            Est_X.append(estimate_x[0,0])
            TH.append(x[2])
            Est_TH.append(estimate_x[2,0])
            U.append(ut)

            # saving for mat file
            states_actual.append(x) 
            estimated_states.append(estimate_x)
            measurements.append(y)

            frame += 1
            time.append(frame*env.tau)
            env.render()

            if abs(ut_desired - ut[0,0]) < epsilon:
                break
        # sleep(2)
        if frame > args.frames:
            break

    if args.store:
        if not(os.path.exists('data')):
            os.makedirs('data')
        mat_data["time"] = time
        mat_data["states_act"] = states_actual
        mat_data["state_meas"] = measurements
        mat_data["states_est"] = estimated_states
        mat_data["inputs"] = U
        logdir = './data/' + args.estimator + '_' \
                 + str(args.frames) + '_' \
                 + str(args.starting_angle) \
                 + str(args.system_noise) + '_mn' \
                 + str(args.measurements)+'.mat'
        sio.savemat(logdir, mat_data)
        print("file saved")

if __name__ == "__main__":
    main()
