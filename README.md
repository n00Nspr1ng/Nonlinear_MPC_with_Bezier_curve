# Nonlinear_MPC_with_Bezier_curve
Nonlinear MPC with reference Bezier Curve in Autonomous Driving

This was made as a class project of Yonsei University class MEU6505: Optimal Control and Reinforcement Learning.

## Dependencies
- python==3.8
- numpy == 1.21.6
- scipy == 1.9.0
- matplotlib == 3.5.2
- casadi == 3.5.5
- gym == 0.24.1
- redis  

You should download the [acados](https://github.com/acados/acados) package.

## To Run
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"

python online.py
```
