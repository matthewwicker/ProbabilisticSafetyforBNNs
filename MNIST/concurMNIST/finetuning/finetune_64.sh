for im in 1 2 4 8 10 11 18 25 26 33 35
do
	python IBP-VCAS-Verify.py --imnum $im --eps 0.025 --samples 2250 --width 64 --margin 1.5
done
