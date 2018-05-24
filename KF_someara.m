%MAE 298 Project
%By Sarah O'Meara
%May 22, 2018

clc
clear all
close all

%%%%%%%%%%%%%%% Input Data %%%%%%%%%%%%%%%%%%%%%%%%%%%
load data_KF

%%%%%%%%%%%%%%% System Parameters %%%%%%%%%%%%%%%%%%%%
g = 9.8; %[m/s^2] gravity
l = 0.5; %[m] distance to cg
M = 1; %[kg] mass of cart
m = 0.1; %[kg] mass of pole
dt = 0.02;  % sampling period [sec]

beta = (4/3) - (m*l)/(M+m);
a32 = (-dt*m*l*g)/((M+m)*beta);
a43 = (g*dt)/(l*beta);
b21 = (beta*(M+m)+m*l)/(beta*l*(M+m)^2);
b41 = -1/(beta*l*(M+m));

%%%%%%%%%%%%%%% State Space Matrices %%%%%%%%%%%%%%%%%
A = [1 dt 0 0; 0 1 a32 0; 0 0 1 dt; 0 0 a43 1];
B = [0; b21; 0; b41];
C = [0 1 0 0; 0 0 0 1];
D = [0];
Q = [10^(-2) 0 0 0; 0 10^(-2) 0 0; 0 0 10^(-2) 0; 0 0 0 10^(-2)];
R = [2*10^(-4) 0; 0 2*10^(-4)];

%%%%%%%%%%%%%%% Initial States %%%%%%%%%%%%%%%%%%%%%%%
x_post = transpose(states_act(1,:));
P_post = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0];

%%%%%%%%%%%%%%% Recursive calculation over time %%%%%%%
for i = 2:length(time)
    
    %KF Model Prediction
    x_prior(:,i) = A*x_post(:,i-1) + B*inputs(i-1);
    P_prior(1:4,4*i-3:4*i) = A*P_post(1:4,4*(i-1)-3:4*(i-1))*transpose(A) + Q;
    
    %KF Model Update
    L = P_prior(1:4,4*i-3:4*i)*transpose(C)*((C*P_prior(1:4,4*i-3:4*i)*transpose(C) + R)^(-1));
    y(1:2,i) = transpose(state_meas(i,:));
    x_post(:,i) = x_prior(:,i) + L*(y(1:2,i) - C*x_prior(:,i) - D*inputs(i));
    P_post(1:4,4*i-3:4*i) = P_prior(1:4,4*i-3:4*i) - L*C*P_prior(1:4,4*i-3:4*i);
end

pos_act = states_act(2:end,1);
vel_act = states_act(2:end,2);
theta_act = states_act(2:end,3);
omega_act = states_act(2:end,1);

pos_est = x_post(1,:);
vel_est = x_post(2,:);
theta_est = x_post(3,:);
omega_est = x_post(4,:);

figure
subplot(2,2,1)
plot(time, pos_act, time, pos_est)
title('Position of Cart')
legend('Actual','KF Estimate')

subplot(2,2,2)
plot(time, vel_act, time, vel_est)
title('Velocity of Cart')
legend('Actual','KF Estimate')

subplot(2,2,3)
plot(time, theta_act, time, theta_est)
title('Angle of Pole')
legend('Actual','KF Estimate')

subplot(2,2,4)
plot(time, omega_act, time, omega_est)
title('Angular Velocity of Pole')
legend('Actual','KF Estimate')

