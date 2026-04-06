
# Convergence-Rate-Optimal Regression and Optimization over Mixed-Type Domains Using Gaussian Process Models.
This respository contains code to run the experiments in the paper Convergence-Rate-Optimal Regression and Optimization over Mixed-Type Domains Using Gaussian Process Models. 

For reproducing the experiments, refer to the each subdirectory in the `tests/` folder.

## Installation

```bash
pip install -r requirements.txt
```

### Training the WEGP Model

To train the WEGP model, run the following command:

```bash
python ./tests/functions/run_script.py --save_dir results --which_func borehole --train_factor 1 --n_jobs 1 --n_repeats 1
```
**Parameter Descriptions:**
- `--save_dir`: Directory to save the experiment results.
- `--train_factor`: A factor to adjust the scale of training.
- `--which_func`: Selects the test function (e.g., `borehole`).
- `--n_jobs`: Number of parallel jobs.
- `--n_repeats`: Number of times to repeat the experiment.
- `--budget`: Budget parameter controlling the number of function evaluations.



### Running the WENBO Algorithm

Execute the following command to run the WEBO algorithm for optimization experiments. You can adjust the experimental settings using the command-line parameters:

```bash 
bash run_cmaes_func2C.sh
```
