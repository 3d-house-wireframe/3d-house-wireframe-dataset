import os
import sys
import numpy as np
import random
import torch
from train import (
    SegmentVQVAE,
    WireframeDataset,
    SegmentVQVAETrainer
)

from argparse import ArgumentParser
from train.config import NestedDictToClass, load_config

# Arguments
program_parser = ArgumentParser(description='Train a segment vqvae model.')
program_parser.add_argument('--vqvae_config', type=str, default='', help='Path to config file.')

args, unknown_args = program_parser.parse_known_args()

cfg = load_config(args.vqvae_config)
args = NestedDictToClass(cfg)
    
train_dataset = WireframeDataset(
    dataset_file_path = args.data.train_set_file_path,
    replica=args.data.replication,
    is_train=True)

# autoencoder

model = SegmentVQVAE(
    encoder_dims_through_depth = args.model.encoder_dims_through_depth,
    decoder_dims_through_depth = args.model.decoder_dims_through_depth,
    codebook_size = args.model.codebook_size,
    attn_encoder_depth = args.model.attn_encoder_depth,
    attn_decoder_depth = args.model.attn_decoder_depth,
)

epochs = args.epochs
batch_size = args.batch_size
num_gpus = args.num_gpus
num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
num_train_steps = epochs * num_step_per_epoch

trainer = SegmentVQVAETrainer(
    model,
    dataset = train_dataset,
    num_train_steps = num_train_steps,
    batch_size = batch_size,
    num_workers = args.num_workers,
    grad_accum_every = 2,
    learning_rate = args.lr,
    max_grad_norm = 1.,
    accelerator_kwargs = dict(
        cpu = False,
        step_scheduler_with_optimizer=False
    ),
    log_every_step = args.log_every_step,
    resume_training=args.resume_training,
    checkpoint_every_step = args.save_every_epoch * num_step_per_epoch,
    checkpoint_folder = args.model.checkpoint_folder,    
    checkpoint_file_name=args.model.checkpoint_file_name,
    use_wandb_tracking = args.use_wandb_tracking,
    num_step_per_epoch = num_step_per_epoch,
)


trainer(project=args.wandb_project_name, run=args.wandb_run_name, hps=cfg)
