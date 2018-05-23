import scipy.io as sio
import argparse
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='EKF_500.mat')
    parser.add_argument('--frames', type=int, default=500)
    args = parser.parse_args()

    logdir = "./data/" + args.file
    frames = args.frames

    saved_data = sio.loadmat(logdir)
    time = saved_data["time"][0,:frames]
    Est_X = saved_data["states_est"][:frames,0,0]
    Est_TH = saved_data["states_est"][:frames,2,0]
    TH = saved_data["states_act"][1:frames+1,2]
    X = saved_data["states_act"][1:frames+1,0]

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

    plt.show(5)

if __name__ == "__main__":
    main()