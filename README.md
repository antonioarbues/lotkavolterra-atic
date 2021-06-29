# Effects of predator migration on two predator-prey systems from a Lotka-Volterra perspective

##### Table of Contents
1. [Introduction](#intro)
2. [Getting Started](#getstarted)
3. [License](#license)
4. [Acknowledgments](#acknowledgments)

<a name="intro"></a>
## Introduction
This is a numerical simulator of a coupled system of two Lotka-Volterra subsystems. The simulator and the [project report](https://github.com/antonioarbues/lotkavolterra-atic/blob/main/report.pdf) include:
- the global and local stability analisys of the system,
- the state estimation with an Extended Kalman Filter,
-  and the control with a Positive Controller and an Optimal Controller.

For a detailed explanation of the system, the methods, and the results, refer to the [project report](https://github.com/antonioarbues/lotkavolterra-atic/blob/main/report.pdf).

![Phase-Space plots of the ecosystem](https://github.com/antonioarbues/lotkavolterra-atic/blob/feature/estimation/phase_0.05.png)

<a name="getstarted"></a>
## Getting started
### Prerequisites
The code is completely based on `Python3`. So, it is sufficient to have a `Python3` interpreter, and have installed the libraries `NumPy` and `MatPlotLib` to be able to run the code.

### Installing
1. Clone the repository from the `main` branch.
2. Update the configuration file `config.yaml` or leave the standard configuration.
### Running the simulation
Run `main.py`.

<a name="contributors"></a>
### Contributors
#### Authors
This repository contains the final project for the class Advanced Topics In Control (2021) at ETHZ by the students:
+ [Patricia Apostol](mailto:papostol@ethz.ch) - MSc Robotics, Systems and Control ETHz (Zürich, CH), BSc Aerospace Engineering TU Delft (Delft, NL)
+ [Antonio Arbues](mailto:aarbues@ethz.ch) - MSc Robotics, Systems and Control ETHz (Zürich, CH), BSc Mechanical Engineering Politecnico di Milano (Milan, IT)
+ [Sandra Wells](mailto:swells@ethz.ch) - MSc Robotics, Systems and Control ETHz (Zürich, CH), BSc Mathematics Universitat Politècnica de Catalunya (Barcelona, ES), BSc Physics Engineering UPC-ETSETB (Barcelona, ES)

<a name="license"></a>
## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/antonioarbues/lotkavolterra-atic/blob/main/LICENSE) file for details.

