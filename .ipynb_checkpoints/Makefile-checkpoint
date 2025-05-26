CONFIG_PATH_AGE = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_age.yaml
CONFIG_PATH_AGE_S = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_age_supervised.yaml
CONFIG_PATH_GENDER = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_gender.yaml
CONFIG_PATH_GENDER_S = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_gender_supervised.yaml
CONFIG_PATH_X5_S = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_x5_supervised.yaml
CONFIG_PATH_OLD = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_my_params.yaml

### mingw32-make test

final_set1:
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_time2vec_part --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_time_plain --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_time2vec_part --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_time2vec_full --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_time_plain --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_X5) --exp-name=baseline_time2vec_full --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_time2vec_full --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=baseline_8 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=baseline_16 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=baseline_24 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=baseline_48 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=baseline_80 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=baseline_112 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_8 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_16 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_24 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_48 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_80 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_112 --ds-name=gender--mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_8 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_16 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_24 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_48 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_80 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_112 --ds-name=gender--mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_X5) --exp-name=baseline_8 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_X5) --exp-name=baseline_16 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_X5) --exp-name=baseline_24 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_X5) --exp-name=baseline_48 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_X5) --exp-name=baseline_80 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_X5) --exp-name=baseline_112 --ds-name=x5--mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_8 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_16 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_24 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_48 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_80 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_112 --ds-name=x5 --mode=train-test || exit 0"

fs1:
	python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_num_emb_emb48_dist_common_emb_cat --ds-name=gender --mode=test


test:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_fasttext_nopca_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_fasttext_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_w2v_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_rubert_cat --ds-name=gender --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_w2v_nopca_cat --ds-name=gender --mode=train-test

x5_supervised:
	@powershell -Command "Start-Sleep -Seconds 10800"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=quant_num_emb_emb16_dist_common_emb_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=quant_100_emb16_dist_common_emb_sum --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=quant_100_emb16_dist_common_emb_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=st_emb16_dist_common_emb_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=st_emb16_dist_common_emb_sum --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=deeptlf9_emb16_disc_common_emb_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=deeptlf9_emb16_disc_common_emb_sum --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_8 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_16 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_24 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_48 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_80 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_112 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_time2vec_part --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_time_plain --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=quant_100_dist_common_emb_time2vec_full_sum --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=quant_100_dist_common_emb_time2vec_part_sum --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=quant_100_dist_common_emb_time2vec_part_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=quant_100_dist_common_emb_time2vec_full_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_glove --ds-name=x5 --mode=train-test || exit 0"

x5_supervised1:
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_8 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_16 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_24 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_48 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_80 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_11 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_glove_x5_sup_8 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_glove_x5_sup_16 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_glove_x5_sup_24 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_glove_x5_sup_48 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_glove_x5_sup_80 --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_X5_S) --exp-name=baseline_glove_x5_sup_11 --ds-name=x5 --mode=train-test || exit 0"



age_supervised:
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=quant_num_emb_emb16_dist_common_emb_cat --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=quant_100_emb16_dist_common_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=quant_100_emb16_dist_common_emb_cat --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=st_emb16_dist_common_emb_cat --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=st_emb16_dist_common_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=deeptlf9_emb16_disc_common_emb_cat --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=deeptlf9_emb16_disc_common_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_8 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_16 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_24 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_48 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_80 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_glove_age_sup_112 --ds-name=age_bins --mode=train-test || exit 0"

age_supervised_emb_size:
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_8 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_16 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_24 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_48 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_80 --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_AGE_S) --exp-name=baseline_112 --ds-name=age_bins --mode=train-test || exit 0"

emb_size:
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_8 --ds-name=gender --mode=test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_16 --ds-name=gender --mode=test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_24 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_48 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_80 --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_112 --ds-name=gender --mode=train-test || exit 0"

sp_text1:
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_w2v --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_w2v_nopca --ds-name=gender --mode=train-test || exit 0"

sp_text:
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_fasttext_nopca --ds-name=gender --mode=train-test || exit 0"\
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_w2v_nopca --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_rubert_nopca --ds-name=gender --mode=train-test || exit 0"

sp:
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=st_emb48_dist_common_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_num_emb_emb48_dist_common_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_emb48_dist_common_emb_sum --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_emb48_dist_common_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=st_emb48_dist_common_emb_sum --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_time2vec_part --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_time_plain --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_dist_common_emb_time2vec_full_sum --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_dist_common_emb_time2vec_part_sum --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_dist_common_emb_time2vec_part_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_dist_common_emb_time2vec_full_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_glove_emb_sum --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_glove --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_glove_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=deeptlf9_glove_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=deeptlf9_glove_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=st_num_emb_glove_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=st_num_emb_glove_emb_sum --ds-name=gender --mode=train-test || exit 0"


sp_glove:
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_glove_emb_sum --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline_glove --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=quant_100_glove_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=deeptlf9_glove_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=deeptlf9_glove_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=st_num_emb_glove_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=st_num_emb_glove_emb_sum --ds-name=gender --mode=train-test || exit 0"

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

test6:
	python exp_supervised_pipeline.py $(CONFIG_PATH_GENDER_S) --exp-name=baseline --ds-name=gender --mode=train

test4:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time_plain --ds-name=x5 --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_emb16_dist_common_emb_sum --ds-name=x5 --mode=train-test
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_emb16_dist_common_emb_cat --ds-name=x5 --mode=train-test

test3:
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_emb16_disc_common_emb_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_emb16_disc_common_emb_sum --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_num_emb_emb16_dist_common_emb_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_emb16_dist_common_emb_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_emb16_dist_common_emb_sum --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time2vec_part --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline_time_plain --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time2vec_full_sum --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time2vec_part_sum --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time2vec_full_cat --ds-name=x5 --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_time2vec_part_cat --ds-name=x5 --mode=train-test || exit 0"


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

glove:
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=quant_100_glove_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=baseline_glove --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=quant_100_glove_emb_cat --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=deeptlf9_glove_emb_cat --ds-name=age_bins --mode=train-test || exit 0"

glove_gender:
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=quant_100_glove_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=baseline_glove --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=quant_100_glove_emb_cat --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=deeptlf9_glove_emb_cat --ds-name=age_bins --mode=train-test || exit 0"

glove1:
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=deeptlf9_glove_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=st_num_emb_glove_emb_cat --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_AGE) --exp-name=st_num_emb_glove_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=quant_100_glove_emb_sum --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=baseline_glove --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=quant_100_glove_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=deeptlf9_glove_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=deeptlf9_glove_emb_sum --ds-name=age_bins --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=st_num_emb_glove_emb_cat --ds-name=gender --mode=train-test || exit 0"
	@cmd /C "python exp_pipeline.py $(CONFIG_PATH_GENDER) --exp-name=st_num_emb_glove_emb_sum --ds-name=gender --mode=train-test || exit 0"



