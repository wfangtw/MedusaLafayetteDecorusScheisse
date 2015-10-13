src_dir=src
data_dir=training_data
pred_dir=predictions
model_dir=models
log_dir=log
python2 $src_dir/train.py $data_dir/$1/train.in $data_dir/$1/dev.in $data_dir/$1/test.in \
    $model_dir/$2.mdl $pred_dir/$2.csv | tee $logdir/$2.log
echo "Program terminated."
