# Second-Sight

# THIS REPOSITORY IS A WORK IN PROGRESS, THIS FLAG WILL BE REMOVED WHEN THE FINAL CODE IS RELEASED (eta AUGUST 2023)

## Installation Commands (Execute in order)

```
git clone https://github.com/reesekneeland/Second-Sight.git
cd Second-Sight
conda env create -f environment.yml`
conda activate SS_UC

```
## Running instructions
To reproduce our results, you can run the following scripts, which will download all of the preprocessed data for the image library, the brain data, and do the necessary processing
```
driver.py
driver_library.py
single_trial_search.py
```
These files set the GPU for experiments and call all of the other functional classes. The main method of each file is designed to be modified to run the experiments you want.
