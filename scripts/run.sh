#/bin/bash

for seed in "0 1 2 3 4 5 6 7 8 9"
do
    echo Running on seed=$seed
    bash scripts/command.sh --seed=$seed
done
