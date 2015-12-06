all:
	@echo 'All codes written in python ^^'
run:
	@echo 'Viterbi decoding...'
	@python2 -W ignore hmm/viterbi.py --weight 0.815 data/3lyr_4096nrn_1188in_prob_fixed \
		data/hmm.mdl data/48_39.map hmm/frame_predictions.csv
	@echo 'Trimming frames...'
	@python2 data/trim.py hmm/frame_predictions.csv output.kaggle data/48_idx_chr.map_b
	@echo 'Finished.'
