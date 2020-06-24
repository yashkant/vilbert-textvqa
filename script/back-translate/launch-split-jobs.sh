for i in {0..7}
do
CUDA_VISIBLE_DEVICES=$i nohup python build_splits.py --seq_id $i > logs/build-splits-$i.txt &
done


