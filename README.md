# Overview
*PyMixSwarms* is an extensible framework for Particle Swarm Optimization (PSO) in Python.
Its philosophy is to take advantage of multiple inheritance (Mixins) in Python, so as to allow great development and experimentation flexibility; PSO variations may be implemented independently as building blocks and then combined hierarchically, without having to change any parts of the implementation.

# Vision
The vision of this project is for a PSO variation "database" to be developed by the community in the form of small building blocks. In turn, any combination of those building blocks may be used, so as to find problem-custom algorithms. Additionally synergies between different PSO variations can be discovered quickly.

# Features

## Overview
- Built-in benchmark functions to test the PSO variations. Developers may easily add more.
- Initialization of particles is controllable thanks to (optional) user-provided seed.\
    One reason for doing this is so as to have fair benchmarking between different PSO variations.\
    NOTE: The underlying iterations are still executed at random, as they should in PSO.
- Experiments (i.e. PSO executions) can be run sequentially or in parallel.
- Automatic pipeline for storing experimental results in .csv files.
    Information to be stored is easily customizable.
- Interactive [Dash](https://plotly.com/dash/) plotting environment for scalar values (e.g. euclidean distance from goal point, objective value) history comparisons.
- (For Developers and Researchers): Extensible framework for implementing your own techniques, as well as smoothly mixing them with existing ones.

- **Free software**: GPLv3 License
- **Python versions**: 3.6 and above

## Particle Swarm Optimization (PSO) variations
The variations currently offered are:
- The class upon which all variations extend upon: [`PSOBackbone`](./src/classes/PSOs/pso_backbone.py), which provides essential elements, like inertia weight $\omega$, learning factors $c_1$ and $c_2$, as well as dictates the PSO step cycle.
- [Standard PSO](./src/classes/PSOs/Mixins/standard_pso/standard_pso.py) (SPSO) by [Shi and Eberhart](https://doi.org/10.1109/CEC.1999.785511)
- [Adaptive Particle Swarm Optimization](./src/classes/PSOs/Mixins/adaptive_pso/adaptive_pso.py) (APSO) by [Zhan et al.](https://doi.org/10.1109/TSMCB.2009.2015956)
- [Enhanced Information Sharing](./src/classes/PSOs/Mixins/enhanced_information_sharing_pso/enhanced_information_sharing_pso.py) (EIS PSO) by [Xueying Lv et al.](https://doi.org/10.1155/2018/5025672).<br>
*NOTE*: Only the third velocity component has been implemented; the Last-Eliminated Principle has not been included (yet).
- [Clan Particle Swarm Optimization](./src/classes/PSOs/clan_pso.py) (Clan PSO) by [Danilo Ferreira de Carvalho and Carmelo José Albanez Bastos‐Filho](https://doi.org/10.1108/17563780910959875)

## Benchmark functions
The objective functions provided are:

<a name="objective_functions_table"></a>
| Name      | ID        | Formula $(\mathbf{x})$   |
| :----:    | :--:      | :-------: |
| Sphere    | `sphere`  | $\sum_{i} x_i^2$ |
| Quadric   | `quadric` | $\sum_i^{D} (\sum_{j}^{i} x_j)^2$ |
| Schwefel N22.2 | `schwefel222` | $\sum_i^D \|x_i\| + \prod_i^D \|x_i\|$ |
| Rosenbrock | `rosenbrock` | $\sum_i^{D-1} 100(x_{i+1} - x_i)^2 + (1 - x_i)^2$ |
| Rastrigin | `rastrigin` | $10 \cdot D  + \sum_i^{D} [x_i^2 - 10 \cos(2 \pi x_i)]$ |
| Ackley | `ackley` | $-a \exp \left(-b \sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2} \right)  -\exp \left( \frac{1}{D} \sum_{i=1}^{D} \cos(2 \pi x_i) \right) + a + e$ |
| Salomon | `salomon` | $1 - \cos \left( 2 \pi \sqrt{\sum_{i=1}^{D} x_i^2} \right) +  0.1 \sqrt{\sum_{i=1}^{D}x_i^2}$ |
| Alpine N.01 | `alpinen1` | $\sum_{i=1}^{D} \| x_i \sin(x_i) + 0.1 x_i \|$ |

Developers may easily add new objective functions that suit their needs. 
# User Manual

## Installation
1) Clone this repository to the directory of your choice.
2) Run the command `pip install -r requirements.txt` to install the dependencies of the project.

## Configure
Create a *json* configuration file, wherever you wish.
For your convenience, a `configs` directory has already been created with two example configurations: `configs/classic_base.json` and `configs/clan_base.json`.

First, provide:
- `"nr_experiments"`: the total number of independent PSO algorithm instances that will be run.
- `"max_iterations"`: the maximum number of steps that each PSO instance will take.
- `"objective_function"`: the objective function **ID** that is going to be used as the evaluation criterion. See the table of (currently) provided objective functions <a href="objective_functions_table">above</a> 
- the topology: `"classic"` or `"clan"` 
- The arguments required by your choice of PSO variations. Read through the documentation of each variation (inside `classes/PSOs`), and consult the example configuration files.

To dictate the search space's number of dimensions, change the value of the `domain_dimensions` variable in script [`src/scripts/benchmark_functions/benchmark_functions.py`](./src/scripts/benchmark_functions/benchmark_functions.py)

## Run
To start the PSO experiments, while in the root folder of the project, run the command

`python -m main <path/to/configuration/file>`

If it doesn't exist already, a new folder, `experiments`, will appear, which provides snapshots of the experiments conducted. Specifically, 4 files will be generated in each experiment folder:
- `config.json`, which is a copy of the configuration file you used.
- `positions.csv`, which contains the positions of all particles at each step of the PSO.
- `gbest_positions.csv`, which contains the position of the global best position at each step of the PSO, and
- `gbest_values.csv`, which contains the objective value of the global best position at each step of the PSO

In addition to the basic execution command described above, there are two arguments that can be provided by the user in the command prompt.

- the `-c` option activates the **parallel execution** of the program. One program instance is created for each thread of the computer's CPU, so as to distribute the load to multiple cores and complete the execution of the program faster.<br>
*WARNING*: Although, execution of all experiments will terminate faster, this option will take over the *entirety* of the CPU.

- the `-s` option allows the user to provide a (integer) **seed** for generating the **initial positions** of the swarms. This feature is provided for:
    + reproducability of results
    + comparing different variations that had identical initial positions

For example, the user may run

`python -m main <path/to/configuration/file> -c -s 1412`

to activate both parallelism and enforce an initial configuration with the seed `1412`.

## Visualize

Although this is not strictly part of the project, a small web-app that is run locally was developed so that users may quickly check the performance of their devised variations.

1. Run `python -m extras.experiments_visualization`
2. If your browser does not automatically open a new tab, provide the IP address where the application is running.<br>
e.g. `* Running on http://127.0.0.1:8050`
3. In the `keyword input` type "gbest_values" (without quotation marks)
4. Click the *Select Directory* button; navigate to and select the path where your experiment folders are collected.<br>
e.g. `classic/particles30/`

The application should plot a trace of the *mean* of the files provided. The user may filter which data points are displayed with the use of
- the *Objective value range* and
- the *Iterations* double slider

located at the bottom of the application.


By following steps 4. and 5., **multiple traces may be plotted** for easier comparison of the provided data.

*NOTE: As visualization of the results is outside of the scope of the project, the web-app will* not *be developed further. Users are encouraged to utilize the experimental data files in the `experiments` directory to create their own visualizations.*

