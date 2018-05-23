# MAE_298_Final_Project
## Running environment  
- gym/cartpole
## Language
- Python 3.6
- Matlab for state estimation
## How to start:
### Install gym package for visualization
- pip install gym
### Run real time control with lqr
- python main.py -est EKF -n 500 -angle 35 -noise 1e-1 --store  
  * "-est": Specify estimator
  * "--store": Store data file in ./data folder, if added 
  * "-n": specify running frames
  * "-noise": specify system noise
  * "-angle": specify starting angles of system  
  
### Data is saved in ./data/somename.mat
- Data check in "Data check.ipynb"
- "time": time steps array
- "state_act": actual states
- "state_meas": measurement result of sensors
- "state_est": estimated state from estimator
- "inputs": inputs calculated from lqr
### Plot of estimation result
- python plot_estimation.py --frames 500 --file "EKF_500.mat"
  * frames: how many frames to draw in a file
  * file: specify name of file your want to draw
## System equations and LQR control
- Introduced in proposal.ipynb
