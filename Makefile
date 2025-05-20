CONFIG_PATH = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_x5.yaml
CONFIG_PATH_OLD = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_my_params.yaml

### mingw32-make test

test:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_fasttext_nopca_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_fasttext_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_w2v_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_rubert_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_w2v_nopca_cat --ds-name=gender --mode=train-test

test_time_text_best:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline --ds-name=gender --mode=test

test_time:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time_plain --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time2vec_part --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time2vec_full --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time2vec_full_scale --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time2vec_part_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time2vec_full_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time_plain_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time2vec_part_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time2vec_full_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time_plain_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time2vec_full_scale_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time2vec_full_withfc --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time2vec_part_withfc --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time2vec_part_rubert --ds-name=gender --mode=train-test


test_text_emb:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_fasttext_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_w2v_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_rubert_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_fasttext --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_w2v --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_rubert --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_rubert_nopca_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_fasttext_nopca_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_w2v_nopca_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_rubert_nopca_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_fasttext_nopca --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_w2v_nopca --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_rubert_nopca --ds-name=gender --mode=train-test

test1:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_sum --ds-name=gender --mode=test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_mean --ds-name=gender --mode=test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_cat --ds-name=gender --mode=test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_mean --ds-name=gender --mode=test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_sum --ds-name=gender --mode=test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_cat --ds-name=gender --mode=test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_sum --ds-name=gender --mode=test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_mean --ds-name=gender --mode=test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_num_emb_dist_common_emb_cat --ds-name=gender --mode=test

test5:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_sum --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_mean --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_cat --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_cat --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_sum --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_mean --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_orig_emb_my_preproc --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_num_emb_dist_common_emb_cat --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_cat --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_mean --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_sum --ds-name=age_bins --mode=train-test

test4:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline --ds-name=x5 --mode=train-test

test3:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline --ds-name=x5 --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_emb16_dist_common_emb_sum --ds-name=x5 --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_emb16_dist_common_emb_cat --ds-name=x5 --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_emb16_disc_common_emb_cat --ds-name=x5 --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_emb16_disc_common_emb_sum --ds-name=x5 --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_num_emb_emb16_dist_common_emb_cat --ds-name=x5 --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_emb16_dist_common_emb_cat --ds-name=x5 --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_emb16_dist_common_emb_sum --ds-name=x5 --mode=train-test

test2:
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=baseline_myparams
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=quant_100_dist_common_emb_myparams_cat
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=quant_num_emb_dist_common_emb_myparams_cat
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=quant_100_dist_common_emb_myparams_sum
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=quant_100_dist_common_emb_myparams_mean
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=st_num_emb_dist_common_emb_myparams_cat
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=st_num_emb_dist_common_emb_myparams_mean
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=st_num_emb_dist_common_emb_myparams_sum
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=deeptlf9_disc_common_emb_myparams_cat
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=deeptlf9_disc_common_emb_myparams_sum
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=deeptlf9_disc_common_emb_myparams_mean