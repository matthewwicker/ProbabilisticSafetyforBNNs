#for (( IMNUM=$2; IMNUM<=$3; IMNUM++ ))
for IMNUM in 1 4 6 7 8 9 11 12 15 16 18 19 20 22 23 24 27 30 31 33 35
do
	python IBP-VCAS-Verify.py --imnum $IMNUM --eps 0.0025 --samples 2250 --width 128 --margin 1.5 
done
