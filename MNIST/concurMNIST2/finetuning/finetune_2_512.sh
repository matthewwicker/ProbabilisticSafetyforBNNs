#for (( IMNUM=$2; IMNUM<=$3; IMNUM++ ))
for IMNUM in 4 6 7 9 12 16 20 21 33 34
do
	python IBP-VCAS-Verify.py --imnum $IMNUM --eps 0.0025 --samples 100 --width 512 --margin 1.0 
done
