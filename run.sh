src_dir=src
data_dir=training_data
pred_dir=predictions
model_dir=models
python2 $src/train.py $data_dir/$1/train.in $data_dir/$1/dev.in $data_dir/$1/test.in \
    $model_dir/$1.mdl $pred_dir/$1.csv
echo "Program terminated."
