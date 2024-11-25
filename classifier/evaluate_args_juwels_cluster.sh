#!/bin/bash
LINE=$(((($SLURM_ARRAY_TASK_ID - 1)* 4) + $SLURM_PROCID + 1))
ARGS=`sed "${LINE}q;d" arguments_cluster.txt`

if [ -n "${ARGS}" ]; then
    python -u classifier.py --mode test_lava $ARGS
fi
