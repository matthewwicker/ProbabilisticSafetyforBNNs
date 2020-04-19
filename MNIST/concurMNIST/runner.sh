for (( IMNUM=$2; IMNUM<=$3; IMNUM++ ))
do
	python IBP-MNIST-Verify.py --imnum $IMNUM --eps 0.025 --samples 500 --width $1 --margin 2.5
	python IBP-MNIST-Verify.py --imnum $IMNUM --eps 0.025 --samples 500 --width $1 --margin 0.0
done
