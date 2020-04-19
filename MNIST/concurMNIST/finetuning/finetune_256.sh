for im in 1 2 5 6 7 8 9 11 12 15 16 17 18 26 29 31 33 34 35
do
	python IBP-VCAS-Verify.py --imnum $im --eps 0.025 --samples 2500 --width 256 --margin 1.8
done
