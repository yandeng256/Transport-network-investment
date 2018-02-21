# Transport-network-investment
This folder includes transport network criticality analysis, economic loss analysis, investment optimization, and stress-testing

Fiji case:

ecoLoss_Fiji.py: python script for economic loss analysis given an investment set;

stressTest_Fiji: python script for stress testing

optimization_FJ: the optimization script to priorize the infrastructure to invest under budget. We use Gurobi as the optimization solver. A paper is attached as a reference for the algorithm. 'Pre-disaster investment decisions for strengthening a highway network' 

network_lib: road network library developed by Delft U

FJ_inputs: input files with all road network, structure with WD information, and OD information for Fiji north/south island

option1, option2, north_clean, south_clean: investment set from gorvernment. structure represented by structure OBJECTID. You can change to any new set if you have.

scenarios_ranges: range of uncertain parameters in stress-testing

Mozambique case:

mozambique case is similar with Fiji case, with the name MZ.
