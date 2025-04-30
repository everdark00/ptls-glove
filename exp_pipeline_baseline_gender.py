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


from functools import partial
from ptls.nn import RnnSeqEncoder, TrxEncoder
from ptls.frames.coles import CoLESModule
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.data_load.datasets import AugmentationDataset
from ptls.data_load.augmentations import DropoutTrx

import ptls
import torch
from torch import nn
from ptls.preprocessing import PandasDataPreprocessor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from ptls.data_load.datasets import inference_data_loader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def get_basic_model_encoder(df_params, config):
    embeddings={
        'mcc_code': {'in': 200, 'out': 48},
        'tr_type': {'in': 100, 'out': 24}
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
    df_params = {
        'id_col' : 'client_id',
        'numeric_cols' : ['amount']
    }
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().setLevel(logging.INFO)

    torch.set_float32_matmul_precision('high')
    
    logging.info(f'experiment {exp_name} started')
    with open(exp_config_path) as yf:
        config = Dict(yaml.full_load(yf))

    with open('../data/train_trx.parquet', 'rb') as fl:
        train = pickle.load(fl)
    with open('../data/test_trx.parquet', 'rb') as fl:
        test = pickle.load(fl)

    train, val = train_test_split(train, test_size=config.datasets[ds_name].val_split_coef, random_state=config.random_state)

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
    
        seq_encoder = get_basic_model_encoder(df_params, config=config)
    
        lr_scheduler = None
        if config.train.lr_scheduler.enabled:
            lr_scheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=config.train.lr_scheduler.step_size, gamma=config.train.lr_scheduler.gamma)
        
        model = CoLESModule(
            seq_encoder=seq_encoder,
            optimizer_partial=partial(torch.optim.Adam, lr=config.train.lr, weight_decay=config.train.weight_decay),
            lr_scheduler_partial=lr_scheduler,
        )
    
        callbacks = []
        
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

        torch.save(seq_encoder.state_dict(), f"{config.models_path}/{exp_name}_{ds_name}.pt")
    
        logging.info(f"{exp_name}: train ended, model saved")

    if mode == 'test' or mode == 'train-test':
        seq_encoder = get_basic_model_encoder(df_params, config=config)
    
        seq_encoder.load_state_dict(torch.load(f"{config.models_path}/{exp_name}_{ds_name}.pt", weights_only=True))

        res_metrics = []

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
                train_dl = inference_data_loader(train, num_workers=config.test.num_workers, batch_size=config.test.proxy_metrics.batch_size)
                train_embeds = torch.vstack(inference_runner.predict(coles_model, train_dl))
                cuda_memory_clear()
                test_dl = inference_data_loader(test, num_workers=config.test.num_workers, batch_size=config.test.proxy_metrics.batch_size)
                test_embeds = torch.vstack(inference_runner.predict(coles_model, test_dl))

            train_df, test_df = get_train_test_gender_scenario(df_params, train_embeds, test_embeds, train, test)
    
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

                avg_metrics_train = {i : 0 for i in metrics.keys()}
                avg_metrics_test = {i : 0 for i in metrics.keys()}

                for i in tqdm(range(config.test.proxy_metrics.n_trials)):
                    embed_columns = [x for x in train_df.columns if x.startswith('embed')]
                    x_train, y_train = train_df[embed_columns], train_df['target']
                    x_test, y_test = test_df[embed_columns], test_df['target']
                    
                    clf.fit(x_train, y_train)

                    for m_name, metric in metrics.items():
                        avg_metrics_test[m_name] += metric(y_test, clf.predict(x_test)) / config.test.proxy_metrics.n_trials
                    if config.test.proxy_metrics.calc_on_train:
                        for m_name, metric in metrics.items():
                            avg_metrics_train[m_name] += metric(y_train, clf.predict(x_train)) / config.test.proxy_metrics.n_trials

                for m_name, m_value in avg_metrics_test.items():
                    res_metrics.append([exp_name, 'test', f"{m_name}_{model_name}", m_value])
                
                if config.test.proxy_metrics.calc_on_train:
                    for m_name, m_value in avg_metrics_train.items():
                        res_metrics.append([exp_name, 'train', f"{m_name}_{model_name}", m_value])

                logging.info(f"{exp_name}: metrics via {model_name} calculated")
        
        report = pd.DataFrame(res_metrics, columns = ['exp_name', 'dataset', 'metric', 'value'])
        if os.path.isfile(f"{config.report_path}/{config.test.report_name}_{ds_name}.csv"):
            prev_report = pd.read_csv(f"{config.report_path}/{config.test.report_name}_{ds_name}.csv").drop(columns=['Unnamed: 0'])
            pd.concat([prev_report, report]).to_csv(f'{config.report_path}/{config.test.report_name}_{ds_name}.csv')
        else:
            report.to_csv(f'{config.report_path}/{config.test.report_name}_{ds_name}.csv')         
            

if __name__=="__main__":
    main()