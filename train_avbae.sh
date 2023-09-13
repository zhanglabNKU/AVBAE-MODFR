export SAVE_PATH=./results/
if [ ! -d $SAVE_PATH  ];then
  mkdir $SAVE_PATH
else
  echo dir exist
fi
python main_avbae.py --data_path ./data/ \
--save_path $SAVE_PATH \
--epoch 40 \
--lr_backbone 0.001 \
--lr_dis 0.001 \
--batch_size 256

