export PYTHONPATH=.

SAVE_DIR="results"
WHICH_FUNC="func2C"
TRAIN_FACTOR=1
N_JOBS=1
N_REPEATS=4 
MAXFUN=500
NOISE=false
BUDGET=100
NUM_PERMUTATIONS=0
INIT_SIZE=6
N_CAND=10

WARMUP_STEPS=800
NUM_SAMPLES=1000
NUM_MODEL_SAMPLES=128


python tests/functions/optimization_GP_cmaes.py \
    --save_dir $SAVE_DIR \
    --which_func $WHICH_FUNC \
    --train_factor $TRAIN_FACTOR \
    --n_jobs $N_JOBS \
    --n_repeats $N_REPEATS \
    --maxfun $MAXFUN \
    --noise $NOISE \
    --budget $BUDGET \
    --num_permutations $NUM_PERMUTATIONS \
    --num_samples $NUM_SAMPLES \
    --warmup_steps $WARMUP_STEPS \
    --num_model_samples $NUM_MODEL_SAMPLES \
    --init_size $INIT_SIZE \
    --N_cand $N_CAND \
