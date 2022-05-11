
dimInd=$(( $1 % 4 ))
seed=$(( $1 % 50 ))

dims=(branincurrin zdt2 carside penicillin)
batches=(50 50 29 29)
dim=${dims[$dimInd]}
batch=${batches[$dimInd]}

echo $seed $dim

python ./runner.py --seed=$seed --n_batch=$batch \
       --mc_samples=64 --problem=$dim --batch_size=3 --dim=5 \
       --output=/scratch/wjm363/conformalbo/${dim}_q3_${seed}.pt

