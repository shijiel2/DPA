#!/bin/bash

K="800"
echo 'partitioning...'
python3 partition_data_norm_hash.py --dataset mnist --partitions $K
echo 'training...'
python3 train_mnist_nin_baseline.py --num_partitions $K --num_partition_range $K --start_partition 0
echo 'evaluating...'
python3 evaluate_mnist_nin_baseline.py --models mnist_nin_baseline_partitions_$K
echo 'certifying...'
python3 certify.py --evaluations mnist_nin_baseline_partitions_$K.pth