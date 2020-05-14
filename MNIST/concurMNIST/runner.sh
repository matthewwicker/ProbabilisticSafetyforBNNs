for IMNUM in 0 #(( IMNUM=$2; IMNUM<=$3; IMNUM++ ))
do
	python IBP-MNIST-Verify.py --imnum $IMNUM --eps 0.025 --samples 1250 --width $1 --margin 2.0
	python IBP-MNIST-Verify.py --imnum $IMNUM --eps 0.025 --samples 100 --width $1 --margin 0.0
done
