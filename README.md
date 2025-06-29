# nonstational-ARM
This is the code associated with non-stational asymptotical regularization method (ARM) algorithm used in [A New Strategy for Convergence Rates of  Non-stationary Asymptotical Regularization Method for Linear Inverse Problems].
## Usage
The code can be downloaded and placed in any local directory. It can be executed like any regular program in VS Code.
The folder contains code for two numerical examples (ex1/ and ex2/). For each example:

H1_Forward.py computes:    
    
    N: Order and number of zeros of bessel functions,
    M_H1_final: Discrete matrix of the forward operator T under two sets of bases,
    coe_H1_four: Coefficient vector of the true solution in the discrete basis,
    jmn_values: Coefficient matrix of basis function compositions,
    input_index: Permutation index matrix for basis functions.

ARMH1.py calculates:
    ARM model computation
    
    T_seris:time range,
    alpha_seris:parameter alpha range,
    rmse_val: optimal rmse under T_seris and alpha_seris,
    alpha_val: optimal alpha correspongding rmse_val,
    T_val: time range,
    Tcoe: coefficients for the estimated solution under basis function achieving minimal RMSE with optimal parameters at        final time of T_seris.

    Output Visualization:
    rmse_vs_T.eps: rmse_val variation over time T_val,
    alpha_vs_T.eps: Optimal alpha_val evolution over time T_val.

drawrecons.py plot:
    Visualizes the optimal RMSE solution at the final timestep, and exports Cartesian coordinate matrices of the plotted solution-data_plot.txt.

drawtrue.py plot:
    Visualizes the true solution at the final timestep, and exports Cartesian coordinate matrices of the plotted solution-true_sol.txt.
    
optimal_H1.m: Visualizes the optimal RMSE and optimal alpha variations over time T_val ,
recons_true_H1.m:Visualizes the optimal RMSE solution and the true solution at the final timestep under Cartesian coordinate matrices.

    
