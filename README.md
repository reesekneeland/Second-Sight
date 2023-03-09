# Second-Sight

## Installation Commands (Execute in order)

```
git clone https://github.com/reesekneeland/Second-Sight.git
cd Second-Sight
git submodule init
git submodule update
conda env create SS -f environment.yml`
conda activate SS
cd stable-diffusion
pip install .

```
## Running instructions
There are a number of files designed to be starting points for running the code, namely:
```
driver.py
driver_library.py
single_trial_search.py
```
These files set the GPU for experiments and call all of the other functional classes. The main method of each file is designed to be modified to run the experiments you want.
