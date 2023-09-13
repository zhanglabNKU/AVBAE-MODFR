export SAVE_PATH=./data/h5/
if [ ! -d $SAVE_PATH  ];then
  mkdir $SAVE_PATH
else
  echo dir exist
fi
python preprocess.py --data_path ./data/ \
--save_path $SAVE_PATH \
--dataset both
