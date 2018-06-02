import scipy.io as sio
import argparse
import matplotlib.pyplot as plt
import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='EKF_500.mat')
    parser.add_argument('--frames', type=int, default=500)
    args = parser.parse_args()

    logdir = args.file
    frames = args.frames
    print(logdir)

    saved_data = sio.loadmat(logdir)
    time = saved_data["time"][0,:frames]
    Est_X = saved_data["states_est"][:frames,0,0]
    Est_TH = saved_data["states_est"][:frames,2,0]
    Est_xdot = saved_data["states_est"][:frames,1,0]
    Est_thdot = saved_data["states_est"][:frames,3,0]

    TH = saved_data["states_act"][1:frames+1,2]
    X = saved_data["states_act"][1:frames+1,0]
    Xdot = saved_data["states_act"][1:frames+1,1]
    THdot = saved_data["states_act"][1:frames+1,3]    

    print("root mean square error of estimated x", rmse(Est_X, X))
    print("root mean square error of estimated theta", rmse(Est_TH, TH))

    plt.subplot(2,2,1)
    plt.plot(time, X,  '--',label="cart position")
    plt.plot(time, Est_X, 'y-',label='Estimated position')
    plt.title('cart position')
    plt.legend()
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(time, TH, '--',label='pendulum angle')
    plt.plot(time, Est_TH,'y-',label='Estimated angle')
    plt.title('angle of pole')
    plt.grid()
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(time, THdot, '--',label='pendulum anglular velocity')
    plt.plot(time, Est_thdot,'y-',label='Estimated anglular velocity')
    plt.title('angular velocity of pole')
    plt.grid()
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(time, Xdot, '--',label='Cart velocity')
    plt.plot(time, Est_xdot,'y-',label='Estimated cart velocity')
    plt.title('Cart velocity')
    plt.grid()
    plt.legend()    

    plt.show(5)

if __name__ == "__main__":
    main()