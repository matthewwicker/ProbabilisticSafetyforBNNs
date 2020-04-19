#script 
for im in {21..40}
do
	python IBP-MNIST-Verify.py --imnum $im --width 256 --eps 0.025 --margin 2.0 --samples 1000 &
done
