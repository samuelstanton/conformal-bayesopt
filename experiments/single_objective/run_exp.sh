doExp() {
	for j in {0..6};
	do
		seed=$((j*8 + $1))
		echo $1 $seed
		export CUDA_VISIBLE_DEVICES=$1; python runner.py --seed=$seed --batch_size=1 --output=results/branin2_${seed}.pt
	done
}

for i in 0 1 2 3 4 5 7;
do
	doExp $i &
done
