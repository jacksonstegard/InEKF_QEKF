# Comparison of Quaternion and Invariant Extended Kalman Filter (QEKF and InEKF)

## Introduction
The purpose of this project is to explore advanced state estimation techniques that are applicable to the Aerospace industry. Specifically, I wanted to compare a Quaternion Extended Kalman Filter (QEKF) to a Invariant Extended Kalman Filter (InEKF). Both filters provide unique methods to tackle the problem of nonlinear state estimation in systems involving attitude. This report serves as a resource for background information on the formulation of each filter and provides a monte carlo simulation comparing each filter.

## Running the code
To run the code, use the runFilter.py script. Within this file, you can control the execution of each filter by setting the runFilters variable to True or False for each filter individually.

To specify the number of Monte Carlo runs, adjust the numMCs variable. Setting numMCs to zero will perform a single, non-randomized simulation run. Additional configuration options, such as noise parameters and plotting settings are available in the "Simulation Setup" section of the runFilter.py script for further customization.


## Documentation
The **InEKF_and_QEKF_Filter_Comparison_Report.pdf**
provides essential background information for this project. The report explains the IMU and GPS models, covers the theoretical foundations of the QEKF and InEKF, and outlines the respective algorithms. It then presents Monte Carlo simulation results, including detailed figures, and concludes with a summary and list of future work.
