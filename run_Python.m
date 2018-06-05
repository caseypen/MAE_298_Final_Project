%###############################################################
%Creator: D. Jordan McCrone
clc;clear;close all;format compact
set(0,'defaultAxesFontSize', 16);% Set axis font size to 16
set(0,'defaulttextInterpreter','latex') %latex axis labels
%###############################################################

% Run this file in MATLAB to generate date for the configurations listed

angles = [5 15 25 30 40];
estim = {'KF','EKF','UKF'};
Q = [0.1 1.0 10.0];
mn = [1 2];
for i = 1:length(angles);
    for j = 1:length(estim);
        for k = 1:length(Q);
            for l = 1:length(mn);
                %python main.py -est estim{j} -angle angles(i) -mn mn(l) -Q Q(k)
                out = sprintf('python main.py -est %s -angle %1.1f -mn %i -Q %1.1f',estim{j},angles(i),mn(l),Q(k));
                system(out)
            end
        end
    end
end
