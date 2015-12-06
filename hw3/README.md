[Environment]
Linux Speech-GTX-780 3.8.0-32-generic #47~precise1-Ubuntu SMP Wed Oct 2 16:19:35 UTC 2013 x86_64 x86_64 x86_64 GNU/Linux

[Language]
Python2

[How to execute]
1. Produce Kaggle output
    (1) do nothing/see some annoying text (I'm sorry, TA :p)
        usage: make
    (2) Viterbi decode and trim
        usage: make run
2. HMM
    name: viterbi.py
    directory: hmm
    usage: viterbi.py [-h] [--weight <weight>]
                      <test-in> <hmm-model-in> <48-39.map> <pred-out>

    Decode Output of DNN Model for Phone Classification.

    positional arguments:
        <test-in>          testing data file name
        <hmm-model-in>     the hmm model stored with cPickle
        <48-39.map>        48_39.map
        <pred-out>         the output file name you want for the output predictions

    optional arguments:
        -h, --help         show this help message and exit
        --weight <weight>  weight

3. Structured Perceptron
    (1) Train
        name: train.py
        directory: perceptron
        usage: train.py [-h] [--learning-rate <learning-rate>]
                        [--batch-size <batch-size>] [--epochs <max-epochs>]
                        <train-in> <dev-in> <crf-model-out>

        Train CRF Model for Phone Sequence Classification.

        positional arguments:
            <train-in>            training data file name
            <dev-in>              validation data file name
            <crf-model-out>       store crf model with cPickle

        optional arguments:
            -h, --help            show this help message and exit
            --learning-rate <learning-rate>
                                  learning rate for stochastic gradient ascent
            --epochs <max-epochs>
                                  maximum epochs for stochastic gradient ascent
    (2) Test
        name: test.py
        directory: perceptron
        usage: test.py [-h] <test-in> <crf-model-in> <crf-pred>

        Test CRF Model for Phone Sequence Classification.

        positional arguments:
            <test-in>       test data file name
            <crf-model-in>  crf model file name (cPickle format)
            <crf-pred>      output predictions in csv format

        optional arguments:
            -h, --help      show this help message and exit

4. CRF
    (1) Train
        name: train.py
        directory: crf
        usage: train.py [-h] [--learning-rate <learning-rate>]
                        [--batch-size <batch-size>] [--epochs <max-epochs>]
                        <train-in> <dev-in> <crf-model-out>

        Train CRF Model for Phone Sequence Classification.

        positional arguments:
            <train-in>            training data file name
            <dev-in>              validation data file name
            <crf-model-out>       store crf model with cPickle

        optional arguments:
            -h, --help            show this help message and exit
            --learning-rate <learning-rate>
                                  learning rate for stochastic gradient ascent
            --epochs <max-epochs>
                                  maximum epochs for stochastic gradient ascent
    (2) Test
        name: train.py
        directory: crf
        usage: test.py [-h] <test-in> <crf-model-in> <crf-pred>

        Test CRF Model for Phone Sequence Classification.

        positional arguments:
            <test-in>       test data file name
            <crf-model-in>  crf model file name (cPickle format)
            <crf-pred>      output predictions in csv format

        optional arguments:
            -h, --help      show this help message and exit

5. Trim frame predictions to phone predictions
    name: trim.py
    directory: data
    usage: trim.py [-h] <input-csv> <output-csv> <map-file>

    Trim frame prediction file to phone prediction file.

    positional arguments:
        <input-csv>   input frame prediction file
        <output-csv>  output phone prediction file
        <map-file>    48_idx_chr.map_b

    optional arguments:
        -h, --help    show this help message and exit
