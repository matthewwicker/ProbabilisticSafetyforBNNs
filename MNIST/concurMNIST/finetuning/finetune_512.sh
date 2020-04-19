for im in 15 18 21 25 26 35
do
	python IBP-VCAS-Verify.py --imnum $im --eps 0.025 --samples 3000 --width 512 --margin 1.75
done
