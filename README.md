### Stochastic Simulation: project 7
# Stochastic modeling of pollutant transport in aquifers

This is the code for the project of the Stochastic Simulation course at EPFL, academic year 2021/2022. The only author is Matteo Calafà.
The folder is composed by the project report and the following Python scripts:
- `parameters.py`: a script that contains all the main parameters values
- `finite_element.py`: code for the Finite Element resolution (section 2 of report)
- `standard_monte_carlo.py`: code for the standard Monte Carlo resolution (in the introduction of report)
- `importance_sampling.py`: code for the Importance Sampling technique implementation (section 3 of report)
- `splitting_method.py`: code for the Splitting Method implementation (section 4 of report)

Every script, except `parameters.py`, is independent from the others and can be run independently through the main at the bottom of the code. In general, various functions are implemented in the same script and then different functions can be called by the main depending on what one needs to perform, e.g. a simple simulation or instead a convergence test.
As a last note, `finite_element.py` needs the `fenics` library installation (https://fenicsproject.org/) to be run.
