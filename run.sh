if [ ! -d "log/train-cre-25-size-224-seed-7-max-large-7-train-7-5-15-test-5-5-15" ]; then
	mkdir log/train-cre-25-size-224-seed-7-max-large-7-train-7-5-15-test-5-5-15
fi
	
CUDA_VISIBLE_DEVICES=4 python main.py -Tw 7 -Ts 5 -Tq 15 -Vw 5 -Vs 5 -Vq 15 --train True \
	--exp "train-cre-25-size-224-seed-7-max-large-7-train-7-5-15-test-5-5-15" > log/train-cre-25-size-224-seed-7-max-large-7-train-7-5-15-test-5-5-15/miniimagenet_5way_5shot.txt &
	
if [ ! -d "log/train-cre-25-size-224-seed-7-max-large-7-train-7-1-15-test-5-1-15" ]; then
	mkdir log/train-cre-25-size-224-seed-7-max-large-7-train-7-1-15-test-5-1-15
fi

CUDA_VISIBLE_DEVICES=5 python main.py -Tw 7 -Ts 1 -Tq 15 -Vw 5 -Vs 1 -Vq 15 --train True \
	--exp "train-cre-25-size-224-seed-7-max-large-7-train-7-1-15-test-5-1-15" > log/train-cre-25-size-224-seed-7-max-large-7-train-7-1-15-test-5-1-15/miniimagenet_5way_1shot.txt &
