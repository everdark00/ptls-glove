CONFIG_PATH = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_age.yaml
CONFIG_PATH_OLD = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_my_params.yaml

### mingw32-make test

test:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_with_text --ds-name=gender --mode=train

test1:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_mean --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_mean --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_mean --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_num_emb_dist_common_emb_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_orig_emb_my_preproc_3 --ds-name=gender --mode=train-test

test4:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_sum --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_mean --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_cat --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_cat --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_mean --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_sum --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_cat --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_sum --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_mean --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_orig_emb_my_preproc --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_num_emb_dist_common_emb_cat --ds-name=age_bins --mode=train-test

test4:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_orig_emb_my_preproc --ds-name=age_bins --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_num_emb_dist_common_emb_cat --ds-name=age_bins --mode=train-test

test3:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_emb48_dist_common_emb_sum --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_emb48_dist_common_emb_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_emb48_disc_common_emb_mean --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_emb48_disc_common_emb_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_emb48_dist_common_emb_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_num_emb_emb48_dist_common_emb_cat --ds-name=gender --mode=train-test

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