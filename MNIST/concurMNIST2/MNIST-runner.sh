#script 
for im in {0..100}
do
	python IBP-MNIST-Verify.py --imnum $im --width 256 --eps 0.025 --margin 2.0 --samples 1000 &
        wait
	python IBP-MNIST-Verify.py --imnum $im --width 256 --eps 0.025 --margin 0.0 --samples 1000 &
        wait
done
