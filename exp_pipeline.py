import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import copy
import pickle
import sys
import os
from pdb import set_trace

import pytorch_lightning as pl

import logging
import click
import yaml
from dotsi import Dict

#sys.path.insert(0, os.path.dirname(os.getcwd()) +  '\\ptls')
#set_trace() 
from functools import partial
from ptls.nn import RnnSeqEncoder, TrxEncoder
from ptls.nn.trx_encoder.trx_encoder_glove import TrxEncoderGlove, TrxEncoderCat
from ptls.nn.trx_encoder import GloveEmbedding
from ptls.preprocessing.baseline_discretizer import KDiscretizer, SingleTreeDiscretizer
from ptls.preprocessing.deeptlf.src import DeepTLF
from ptls.frames.coles import CoLESModule
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.frames import TestModule
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.data_load.datasets import AugmentationDataset
from ptls.data_load.augmentations import DropoutTrx
from ptls.preprocessing.deeptlf import DeepTLFDisc
from ptls.preprocessing.time_preprocessing import TimePreprocessor  

import ptls
import torch
from torch import nn
from ptls.preprocessing import PandasDataPreprocessor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from ptls.data_load.datasets import inference_data_loader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def cuda_memory_clear():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def prepare_data_age_bins_scenario():
    data_path = '../data/age_bins'

    source_data = pd.read_csv(os.path.join(data_path, 'transactions_train.csv'))

    df_params = {
        "numeric_cols" : ["amount_rur"],
        "cat_cols" : ["trans_date", "small_group"],
        "order_col" : "trans_date",
        "cat_unique" : [],
        "id_col" : "client_id",
        "target" : "bins"
    } 
 
    for f in df_params["cat_cols"]:
        df_params["cat_unique"].append(source_data[f].unique().shape[0])

    targets = pd.read_csv(os.path.join('../data/age_bins', 'train_target.csv'))    

    return source_data, targets, df_params

def prepare_data_gender_scenario():
    data_path = '../data/gender'

    source_data = pd.read_csv(os.path.join(data_path, 'transactions_d.csv'))
    source_data = source_data.drop(columns=["term_id"]).rename(columns={'customer_id' : 'client_id'})
    if 'Unnamed: 0' in source_data.columns:
        source_data = source_data.drop(columns=['Unnamed: 0'])

    day = [int(i.split()[0]) for i in source_data.tr_datetime.values]
    time = [i.split()[1] for i in source_data.tr_datetime.values]

    padded_time = source_data['tr_datetime'].str.pad(15, 'left', '0')
    day_part = padded_time.str[:6].astype(float)
    time_part = pd.to_datetime(padded_time.str[7:], format='%H:%M:%S').values.astype('int64') // 1e9
    time_part = time_part % (24 * 60 * 60) / (24 * 60 * 60)
    
    source_data.tr_datetime = day_part + time_part
   # source_data.amount = np.sign(source_data.amount) * np.log(np.abs(source_data.amount) + 1.0)

    df_params = {
        "numeric_cols" : ["amount"],
        "cat_cols" : ["mcc_code", "tr_type"],
        "cat_unique" : [],
        "order_col" : "tr_datetime",
        "time_col": "time",
        "text_cols" : ['description'],
        "id_col" : "client_id",
        "target" : "gender"
    }

    for f in df_params["cat_cols"]:
        df_params["cat_unique"].append(source_data[f].unique().shape[0])

    targets = pd.read_csv(os.path.join('../data/gender', 'gender_train.csv')).rename(columns={'customer_id' : 'client_id'})
    targets = source_data[['client_id']].drop_duplicates().merge(targets, on='client_id', how='left').dropna() 
    
    return source_data, targets, df_params

def init_disc(params, df_params, config):
    emb_size = None
    if params.fixed_emb:
        emb_size = config.model.embed_size
        
    if params.type in {'quantile', 'uniform', 'kmeans'}:
        disc = KDiscretizer(
            f_names = df_params['numeric_cols'],
            k_bins = params.k_bins,
            d_type = params.type,
            emb_sz = emb_size
        )
    elif params.type == 'st':
        disc = SingleTreeDiscretizer(
            f_names = df_params['numeric_cols'], 
            target_name = df_params['target'], 
            target_type = params.task_type, 
            k_bins = [params.k_bins],
            emb_sz = emb_size
        )
    elif params.type == 'deeptlf':
        disc = DeepTLFDisc({
          "n_est" : params.n_est,
          "min_freq" : params.min_freq,
          "features" : df_params['numeric_cols'] + df_params['cat_cols'],
          "features_to_split" : df_params['numeric_cols'],
        })
    else:
        raise Exception(f'No discretizer with name {params.type} availible')
    return disc

def get_basic_model_encoder(df_params, config):
    if df_params['target'] == 'gender':
        embeddings={
            'mcc_code': {'in': 200, 'out': 48},
            'tr_type': {'in': 100, 'out': 24}
        }
    else:
        embeddings={
            'trans_date': {'in': 800, 'out': 16},
            'small_group': {'in': 250, 'out': 16}
        }

    trx_encoder_params = dict(
        embeddings_noise=0.003,
        numeric_values=dict([(fe, 'identity') for fe in df_params['numeric_cols']]),
        embeddings=embeddings
    )
    
    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoder(**trx_encoder_params),
        hidden_size=config.model.hidden_size,
        type=config.model.seq_encoder_type,
        bidir=False,
        trainable_starter='static'
    )
    
    return seq_encoder

def get_cat_encoder(df_params, agg_type, config, num_emb_flag=False, text_embeddings_path=None):
    embeddings=dict()
    for i, f in enumerate(df_params["cat_cols"]):
        embeddings[f] = {'in' : df_params["cat_unique"][i], 'out' : config.model.embed_size}       

    trx_encoder_params = dict(
        embeddings=embeddings,
        id_col_name=df_params['id_col'],
        embeddings_noise=0.003,
        agg_type=agg_type,
        numeric_separate=num_emb_flag,
        numeric_features=df_params['numeric_cols'],
        text_embeddings_path=text_embeddings_path
    )
    
    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoderCat(**trx_encoder_params),
        hidden_size=config.model.hidden_size,
        type=config.model.seq_encoder_type,
        bidir=False,
        trainable_starter='static'
    )
    return seq_encoder

def get_trans_encoder(df_params, agg_type, algo, config, numeric_separate=False):
    embeddings=dict()

    trx_encoder_params = dict(
        feature_names=df_params['cat_cols'], 
        in_emb_sizes=df_params["cat_unique"],
        out_emb_size=config.model.embed_size,
        agg_type=agg_type,
        numeric_separate=numeric_separate,
        numeric_features=df_params['numeric_cols']
    )
    
    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoderTrans(**trx_encoder_params),
        hidden_size=config.model.hidden_size,
        type=config.model.seq_encoder_type,
        bidir=False,
        trainable_starter='static'
    )
    return seq_encoder

def get_glove_encoder(df_params, exp, glove_embedding, config):
    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoderGlove(glove_embedding, agg_type=exp['agg_type'], numeric_separate=exp['nsep']),
        hidden_size=config.model.hidden_size,
        type=config.model.seq_encoder_type,
        bidir=False,
        trainable_starter='static'
    )
    return seq_encoder

def get_train_test_age_bins_scenario(df_params, train_embeds, test_embeds, train, test):
    data_path = "../data/age_bins"
    
    df_target = pd.read_csv(os.path.join(data_path, 'train_target.csv'))
    df_target = df_target.set_index(df_params["id_col"])
    df_target.rename(columns={"bins": "target"}, inplace=True)
    
    train_df = pd.DataFrame(data=train_embeds, columns=[f'embed_{i}' for i in range(train_embeds.shape[1])])
    train_df[df_params["id_col"]] = [x[df_params["id_col"]] for x in train]
    train_df = train_df.merge(df_target, how='left', on=df_params["id_col"])
    
    test_df = pd.DataFrame(data=test_embeds, columns=[f'embed_{i}' for i in range(test_embeds.shape[1])])
    test_df[df_params["id_col"]] = [x[df_params["id_col"]] for x in test]
    test_df = test_df.merge(df_target, how='left', on=df_params["id_col"])
    return train_df, test_df

def get_train_test_gender_scenario(df_params, train_embeds, test_embeds, train, test):
    data_path = "../data/gender"

    df_target = pd.read_csv(os.path.join('../data/gender', 'gender_train.csv')).drop(columns=['Unnamed: 0']).rename(columns={'customer_id' : 'client_id'})
    df_target = df_target.set_index(df_params["id_col"])
    df_target.rename(columns={"gender": "target"}, inplace=True)
    
    train_df = pd.DataFrame(data=train_embeds, columns=[f'embed_{i}' for i in range(train_embeds.shape[1])])
    train_df[df_params["id_col"]] = [x[df_params["id_col"]] for x in train]
    train_df = train_df.merge(df_target, how='left', on=df_params["id_col"])
    
    test_df = pd.DataFrame(data=test_embeds, columns=[f'embed_{i}' for i in range(test_embeds.shape[1])])
    test_df[df_params["id_col"]] = [x[df_params["id_col"]] for x in test]
    test_df = test_df.merge(df_target, how='left', on=df_params["id_col"])
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    return train_df, test_df


@click.command()
@click.argument('exp-config-path', type=click.Path(exists=True))
@click.option('--exp-name', default=None)
@click.option('--ds-name', default='gender')
@click.option('--mode', default='train-test')
def main(exp_config_path, exp_name, ds_name, mode):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().setLevel(logging.INFO)

    torch.set_float32_matmul_precision('high')
    
    logging.info(f'experiment {exp_name} started, mode: {mode}')
    with open(exp_config_path) as yf:
        config = Dict(yaml.full_load(yf))

    if ds_name == 'age_bins': 
        data, targets, df_params = prepare_data_age_bins_scenario()
    elif ds_name == 'gender': 
        data, targets, df_params = prepare_data_gender_scenario()
    else:
        raise Exception('Incorrect dataset name provided!')
    logging.info(f"{exp_name}: data loaded")

    exp = config.experiments[exp_name]

    if 'baseline' in exp_name:
        for fe in df_params["numeric_cols"]:
            data[fe] = np.sign(data[fe]) * np.log(np.abs(data[fe]) + 1.0)

    # if 'datetime_feats' in exp:
    #     if ds_name == 'age_bins':
    #         dt_preprocessor = TimePreprocessor(idcol=df_params["id_col"], 
    #                   ordercol=df_params["order_col"],
    #                   mode=exp['datetime_feats'])
    #     elif ds_name == 'gender': 
            
    #     else:
    #     raise Exception('Incorrect dataset name provided!')

    disc = None
    if 'disc' in exp:
        disc = init_disc(exp.disc, df_params, config)
        if exp_name[:2] != "st" and ('ST' not in exp_name) and (disc is not None):
            disc.fit(data)
            data = disc.transform(data, to_embeds=exp['nemb'] if 'nemb' in exp else False)
        elif (disc is not None):
            disc.fit(data.sample(int(2e+5), random_state=42).merge(targets, on=df_params['id_col'], how='inner'))
            data = disc.transform(data, to_embeds=exp['nemb'] if 'nemb' in exp else False)
        logging.info(f"{exp_name}: data discretized")

    if 'nemb' in exp and not exp['nemb']:
        df_params['cat_cols'] =  df_params['numeric_cols'] + df_params['cat_cols']
        df_params["cat_unique"] = (disc.k_bins if (type(disc.k_bins) is list) else [disc.k_bins] * len(df_params['numeric_cols'])) + df_params["cat_unique"]
        df_params['numeric_cols'] = []
    else:
        nn = []
        for fn in df_params['numeric_cols']:
            nn += [fn + '_val', fn + '_pos']
    if not os.path.isfile(f"{config.prep_datasets_path}/{exp_name}_dataset_{ds_name}.pkl"):
        if 'glove_config' in exp:
            if not exp['nsep']:
                embedded_feats = df_params['numeric_cols'] + df_params['cat_cols']
            else:
                embedded_feats = df_params['cat_cols']
            folder_nm = f'{config.emb_path}/{exp_name}'[:-4] if exp['agg_type'] != 'mean' else f'{config.emb_path}/{exp_name}'[:-5]
            glove_embedding = GloveEmbedding(
                feature_names=embedded_feats,
                calculate_cooccur=False,
                embedding_folder=folder_nm,
                glove_params=exp['glove_config']
            )
            glove_embedding.load()
            data = glove_embedding.tokenize_data(data)

        preprocessor = PandasDataPreprocessor(
            col_id=df_params['id_col'],
            col_event_time=df_params['order_col'],
            event_time_transformation='none',
            category_transformation = 'none' if ('glove_config' in exp) else 'frequency',
            cols_category=df_params['cat_cols'],
            cols_numerical= nn if ('nemb' in exp and exp['nemb']) else df_params['numeric_cols'] ,
            cols_identity = df_params['text_cols'],
            return_records=True,
        )
    
        dataset = preprocessor.fit_transform(data)
        
        dataset = sorted(dataset, key=lambda x: x[df_params['id_col']])
    
        with open(f"{config.prep_datasets_path}/{exp_name}_dataset_{ds_name}.pkl", "wb") as fl:
            pickle.dump(dataset , fl)
        logging.info(f"{exp_name}: data preprocessed and saved")
    else:
        with open(f"{config.prep_datasets_path}/{exp_name}_dataset_{ds_name}.pkl", "rb") as fl:
            dataset = pickle.load(fl)
        logging.info(f"{exp_name}: data has been already preprocessed, load data")

    set_trace()
    train, test = train_test_split(dataset, test_size=config.datasets[ds_name].test_split_coef, random_state=config.random_state)
    # with open('../data/train_trx_comp.parquet', 'rb') as fl:
    #     train = pickle.load(fl)
    # with open('../data/test_trx_comp.parquet', 'rb') as fl:
    #     test = pickle.load(fl)

    train, val = train_test_split(train, test_size=config.datasets[ds_name].val_split_coef, random_state=config.random_state)

    text_embeddings_path = None
    if 'text_feats' in exp:
        text_embeddings_path = os.path.isfile(f"{config.raw_data_path}/text_embeddings/{exp_name}_te_{ds_name}.parquet")
        if not text_embeddings_path:
            text_prep = TextPreprocessor(exp.text_feats.method, df_params['text_cols'], enable_pca=True, compressed_dim=48)
    
            train_text_embeds = text_prep.calc_embeddings(train, df_params['id_col'], fit_pca=True)
            val_text_embeds = text_prep.calc_embeddings(val, df_params['id_col'], fit_pca=False)
            test_text_embeds = text_prep.calc_embeddings(test, df_params['id_col'], fit_pca=False)
    
            pd.concat([train_text_embeds, val_text_embeds, test_text_embeds]).to_parquet(text_embeddings_path)

            del text_prep

    del dataset, data

    if mode == 'train' or mode == 'train-test':
        train_dl = PtlsDataModule(
            train_data = ColesDataset(
                   data=AugmentationDataset(
                    f_augmentations=[
                        DropoutTrx(trx_dropout=0.01)
                    ],
                    data=MemoryMapDataset(
                        data=train,
                        i_filters=[
                            SeqLenFilter(min_seq_len=config.train.data_loader.train.min_seq_len),
                        ],
                    )
                   ),
                    splitter=SampleSlices(
                        split_count=config.train.data_loader.train.split_count,
                        cnt_min=config.train.data_loader.train.cnt_min,
                        cnt_max=config.train.data_loader.train.cnt_max,
                    ),
                ),
            train_num_workers=config.train.data_loader.num_workers,
            train_batch_size=config.train.data_loader.train.batch_size,
            valid_data = ColesDataset(
                    MemoryMapDataset(
                        data=val,
                        i_filters=[
                            SeqLenFilter(min_seq_len=config.train.data_loader.val.min_seq_len),
                        ],
                    ),
                    splitter=SampleSlices(
                        split_count=config.train.data_loader.val.split_count,
                        cnt_min=config.train.data_loader.val.cnt_min,
                        cnt_max=config.train.data_loader.val.cnt_max,
                    ),
                ),
            valid_num_workers=config.train.data_loader.num_workers,
            valid_batch_size=config.train.data_loader.val.batch_size,
        )
    
        if exp.trx_encoder_type == 'cat':
            seq_encoder = get_cat_encoder(df_params, agg_type=exp.agg_type, config=config, num_emb_flag=exp.nemb, text_embeddings_path=text_embeddings_path)
        elif exp.trx_encoder_type == 'trans':
            seq_encoder = get_trans_encoder(df_params, agg_type=exp.agg_type, algo=exp.algo, config=config, numeric_separate=exp.nsep)
        elif exp.trx_encoder_type == 'glove':
            seq_encoder = get_glove_encoder(df_params, exp, glove_embedding, config=config)
        elif exp.trx_encoder_type == 'basic':
            seq_encoder = get_basic_model_encoder(df_params, config=config)
        else:
            raise Exception(f"No trx encoder with name {exp.trx_encoder_type}!")
    
        lr_scheduler = None
        if config.train.lr_scheduler.enabled:
            lr_scheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=config.train.lr_scheduler.step_size, gamma=config.train.lr_scheduler.gamma)
        
        model = CoLESModule(
            seq_encoder=seq_encoder,
            optimizer_partial=partial(torch.optim.Adam, lr=config.train.lr, weight_decay=config.train.weight_decay),
            lr_scheduler_partial=lr_scheduler,
        )
    
        callbacks = []
        if config.train.early_stopping.enabled:
            callbacks.append(EarlyStopping(f'valid/{model.metric_name}', mode='max', patience=config.train.early_stopping.patience, min_delta=config.train.early_stopping.min_delta))

        if config.train.save_best_checkpoint:
            callbacks.append(ModelCheckpoint(
                monitor=f'valid/{model.metric_name}',
                dirpath=config.models_path,
                filename=f'{exp_name}_{ds_name}',
                save_top_k=1,
                mode='max'
            ))
        
        trainer = pl.Trainer(
            max_epochs=config.train.max_epochs,
            accelerator=config.train.device,
            callbacks = callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False
        )
    
        logging.info(f"{exp_name}: train starts")
    
        trainer.fit(model, train_dl)
        logging.info(trainer.logged_metrics)

        if not config.train.save_best_checkpoint:
            torch.save(seq_encoder.state_dict(), f"{config.models_path}/{exp_name}_{ds_name}.pt")
    
        logging.info(f"{exp_name}: train ended, model saved")

    if mode == 'test' or mode == 'train-test':
        if 'glove_config' in exp:
            if not exp['nsep']:
                embedded_feats = df_params['numeric_cols'] + df_params['cat_cols']
            else:
                embedded_feats = df_params['cat_cols']
            folder_nm = f'../glove_embeddings/{exp_name}'[:-4] if exp['agg_type'] != 'mean' else f'../glove_embeddings/{exp_name}'[:-5]
            glove_embedding = GloveEmbedding(
                feature_names=embedded_feats,
                calculate_cooccur=False,
                embedding_folder=folder_nm,
                glove_params=exp['glove_config']
            )
            glove_embedding.load()
            
        if exp.trx_encoder_type == 'cat':
            seq_encoder = get_cat_encoder(df_params, agg_type=exp.agg_type, config=config, num_emb_flag=exp.nemb)
        elif exp.trx_encoder_type == 'trans':
            seq_encoder = get_trans_encoder(df_params, agg_type=exp.agg_type, algo=exp.algo, config=config, numeric_separate=exp.nsep)
        elif exp.trx_encoder_type == 'glove':
            seq_encoder = get_glove_encoder(df_params, exp, glove_embedding, config=config)
        elif exp.trx_encoder_type == 'basic':
            seq_encoder = get_basic_model_encoder(df_params, config=config)
        else:
            raise Exception(f"No trx encoder with name {exp.trx_encoder_type}!")

        if config.train.save_best_checkpoint:
            state_dict = torch.load(f"{config.models_path}/{exp_name}_{ds_name}.ckpt")['state_dict']
            new_state_dict = {k.replace('_seq_encoder.', ''): v for k, v in state_dict.items()}
            seq_encoder.load_state_dict(new_state_dict)
        else:
            seq_encoder.load_state_dict(torch.load(f"{config.models_path}/{exp_name}_{ds_name}.pt", weights_only=True))

        res_metrics = []

        if config.test.recall_top_k.enable:
            metric = BatchRecallTopK(config.test.recall_top_k.data_loader.split_count - 1)

            if config.test.recall_top_k.calc_on_train : 
                datasets = [('train', train), ('test', test)]
            else:
                datasets = [('test', test)]
                
            for ds_nm, ds in datasets:
                dl = PtlsDataModule(
                    test_data = ColesDataset(
                            MemoryMapDataset(
                                data=ds,
                                i_filters=[
                                    SeqLenFilter(min_seq_len=config.test.recall_top_k.data_loader.min_seq_len),
                                ],
                            ),
                            splitter=SampleSlices(
                                split_count=config.test.recall_top_k.data_loader.split_count,
                                cnt_min=config.test.recall_top_k.data_loader.cnt_min,
                                cnt_max=config.test.recall_top_k.data_loader.cnt_max,
                            ),
                        ),
                    train_num_workers=config.test.num_workers,
                    train_batch_size=config.test.recall_top_k.data_loader.batch_size,
                )
        
                module = TestModule(
                    model = seq_encoder,
                    metrics = {"recall_top_k" : metric}
                )
            
                predictor = pl.Trainer(
                        accelerator=config.test.device,
                        enable_progress_bar=True,
                        enable_model_summary=False,
                        logger=False
                )
        
                predictor.predict(module, dl)
        
                ds_metrics = module.get_metrics()
        
                for m in ds_metrics:
                    res_metrics.append([exp_name, ds_nm, m, ds_metrics[m]])

        if config.test.proxy_metrics.enable:
            coles_model = CoLESModule(
                seq_encoder=seq_encoder,
            )
        
            inference_runner = pl.Trainer(
                accelerator=config.test.device,
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=False
            )
        
            with torch.no_grad():
                cuda_memory_clear()
                train_dl = inference_data_loader(train + val, num_workers=config.test.num_workers, batch_size=config.test.proxy_metrics.batch_size)
                train_embeds = torch.vstack(inference_runner.predict(coles_model, train_dl))
                cuda_memory_clear()
                test_dl = inference_data_loader(test, num_workers=config.test.num_workers, batch_size=config.test.proxy_metrics.batch_size)
                test_embeds = torch.vstack(inference_runner.predict(coles_model, test_dl))

            if ds_name == 'age_bins': 
                train_df, test_df = get_train_test_age_bins_scenario(df_params, train_embeds, test_embeds, train + val, test)
            elif ds_name == 'gender': 
                train_df, test_df = get_train_test_gender_scenario(df_params, train_embeds, test_embeds, train + val, test)
            else:
                raise Exception(f"No raw dataset with name {ds_name} exists!")

            # with open('../data/embeddings/mles_embeddings.pickle', 'rb') as fl:
            #     embeds = pickle.load(fl)
            
            # embeds.customer_id = embeds.customer_id.astype('int64')
            
            # target = pd.read_csv('../data/gender/gender_train.csv').drop(columns=['Unnamed: 0'])
            # test_ids = pd.read_csv('../data/test_ids.csv')
            # train_ids = pd.DataFrame({'customer_id' : list(set(embeds.customer_id.values) - set(test_ids.customer_id.values))})
            
            # embeds = embeds.merge(target, on='customer_id', how='inner')
            
            # train_df = embeds.merge(train_ids, on='customer_id', how='inner').rename(columns={'gender' : 'target'})
            # test_df = embeds.merge(test_ids, on='customer_id', how='inner').rename(columns={'gender' : 'target'})

            train_df = train_df.set_index(np.arange(train_df.shape[0]))
            test_df = test_df.set_index(np.arange(test_df.shape[0]))

            metrics = {}
            for m in config.test.proxy_metrics.metrics:
                if m == 'accuracy':
                    metrics[m] = accuracy_score
                elif m == 'roc_auc':
                    metrics[m] = roc_auc_score
                else:
                    raise Exception(f"No proxy metric with name {m} exists!")
                    
            logging.info(f"{exp_name}: proxy models eval started")
        
            for model_name, model_config in config.test.proxy_metrics.models.items():
                if 'basic' in model_config:
                    model_config = {}
                if model_name == 'lgbm_boosting':
                    clf = LGBMClassifier(**model_config)
                elif model_name == 'random_forest':
                    clf = RandomForestClassifier(**model_config)
                else:
                    raise Exception(f"No proxy model with name {model_name} exists!")

                kfold = KFold(n_splits=config.test.proxy_metrics.n_folds, shuffle=True, random_state=config.random_state)

                avg_metrics_val = {i : 0 for i in metrics.keys()}
                avg_metrics_test = {i : 0 for i in metrics.keys()}
                embed_columns = [x for x in train_df.columns if x.startswith('emb')]

                for train, val in tqdm(kfold.split(train_df)):
                    x_train, y_train = train_df.loc[train, embed_columns], train_df.loc[train, 'target'].values
                    x_val, y_val = train_df.loc[val, embed_columns], train_df.loc[val, 'target'].values
                    x_test, y_test = test_df[embed_columns], test_df['target'].values

                    transformer = MaxAbsScaler().fit(x_train)
                    x_train = transformer.transform(x_train)
                    x_val = transformer.transform(x_val)
                    x_test = transformer.transform(x_test)

                    #set_trace()
                    
                    clf.fit(x_train, y_train)

                    for m_name, metric in metrics.items():
                        if m_name != 'roc_auc':
                            avg_metrics_test[m_name] += metric(y_test, clf.predict(x_test)) / config.test.proxy_metrics.n_folds
                        else:
                            avg_metrics_test[m_name] += metric(y_test, clf.predict_proba(x_test)[:, 1]) / config.test.proxy_metrics.n_folds
                    if config.test.proxy_metrics.calc_on_val:
                        for m_name, metric in metrics.items():
                            if m_name != 'roc_auc':
                                avg_metrics_val[m_name] += metric(y_val, clf.predict(x_val)) / config.test.proxy_metrics.n_folds
                            else:
                                avg_metrics_val[m_name] += metric(y_val, clf.predict_proba(x_val)[:, 1]) / config.test.proxy_metrics.n_folds

                for m_name, m_value in avg_metrics_test.items():
                    res_metrics.append([exp_name, 'test', f"{m_name}_{model_name}", m_value])
                
                if config.test.proxy_metrics.calc_on_val:
                    for m_name, m_value in avg_metrics_val.items():
                        res_metrics.append([exp_name, 'val', f"{m_name}_{model_name}", m_value])

                logging.info(f"{exp_name}: metrics via {model_name} calculated")
        
        report = pd.DataFrame(res_metrics, columns = ['exp_name', 'dataset', 'metric', 'value'])
        if os.path.isfile(f"{config.report_path}/{config.test.report_name}_{ds_name}.csv"):
            prev_report = pd.read_csv(f"{config.report_path}/{config.test.report_name}_{ds_name}.csv").drop(columns=['Unnamed: 0'])
            pd.concat([prev_report, report]).to_csv(f'{config.report_path}/{config.test.report_name}_{ds_name}.csv')
        else:
            report.to_csv(f'{config.report_path}/{config.test.report_name}_{ds_name}.csv')         
            

if __name__=="__main__":
    main()