export SAVE_PATH=./results/
if [ ! -d $SAVE_PATH  ];then
  mkdir $SAVE_PATH
else
  echo dir exist
fi
python main_modfr.py --data_path ./data/ \
--save_path $SAVE_PATH \
--E1 300 \
--epochs 300 \
--batch_size 128 \
--mask_size 128 \
--f 0.5 \
--p 5 \
--random_mask_size 500 \
--FS 50 \
--use_pr