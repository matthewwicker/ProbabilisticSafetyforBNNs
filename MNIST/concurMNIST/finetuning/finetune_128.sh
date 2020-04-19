for im in 1 3 5 7 8 9 11 14 16 18 23 25 31 33 35
do
	python IBP-VCAS-Verify.py --imnum $im --eps 0.025 --samples 2250 --width 128 --margin 1.5
done
