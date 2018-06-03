#!/bin/bash
angles=( 5 15 25 30 40 )
estim=( "KF" "EKF" "UKF" )
Q=( 0.1 1 10 )
mn=( 1 2 )
for i in "${angles[@]}"; do
	for j in "${estim[@]}"; do
		for k in "${Q[@]}"; do
			for l in "${mn[@]}"; do
				echo "python main.py -est $j -angle $i -Q $k -mn $l --store"
			done
		done
	done
done
