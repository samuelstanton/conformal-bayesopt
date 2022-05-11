
dimInd=$(( $1 % 4 ))
seed=$(( $1 % 50 ))

dims=(5 10 20 50)
dim=${dims[$dimInd]}

echo $seed $dim

python ./runner.py --seed=$seed --n_batch=50 \
       --problem=levy --dim=$dim --batch_size=3 \
       --output=/scratch/wjm363/conformalbo/levy${dim}_q3_${seed}.pt

