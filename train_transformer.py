import os
import sys
from train import (
    SegmentVQVAE,
    WireframeDataset,
    WireframeTransformerTrainer,
    WireframeTransformer,
)
from argparse import ArgumentParser
from train.config import NestedDictToClass, load_config

# Arguments
program_parser = ArgumentParser(description='Train a segment vqvae model.')
program_parser.add_argument('--config', type=str, default='', help='Path to config file.')

args, unknown_args = program_parser.parse_known_args()

cfg = load_config(args.config)
args = NestedDictToClass(cfg)
    
train_dataset = WireframeDataset(
    dataset_file_path = args.data.dataset_file_path,
    replica=args.data.replication,
    is_train=True)

val_dataset = WireframeDataset(
    dataset_file_path = args.data.dataset_file_path,
    replica=args.data.replication,
    is_train=False)

# autoencoder

vqvae = SegmentVQVAE(
    encoder_dims_through_depth = args.model.encoder_dims_through_depth,
    decoder_dims_through_depth = args.model.decoder_dims_through_depth,
    codebook_size = args.model.codebook_size,
    attn_encoder_depth = args.model.attn_encoder_depth,
    attn_decoder_depth = args.model.attn_decoder_depth,
)


transformer = WireframeTransformer(
    vqvae,
    dim = args.model.dim,
    attn_depth = args.model.attn_depth,
    attn_heads = args.model.attn_heads,
    max_seq_len = args.model.max_seq_len,
    coarse_pre_gateloop_depth = args.model.coarse_pre_gateloop_depth,
    fine_pre_gateloop_depth = args.model.fine_pre_gateloop_depth,
)

epochs = args.epochs
batch_size = args.batch_size
num_gpus = args.num_gpus
num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
num_train_steps = epochs * num_step_per_epoch
num_warmup_steps = int(0.1*num_train_steps)
    
trainer = WireframeTransformerTrainer(
    transformer,
    dataset = train_dataset,
    val_dataset = val_dataset,
    num_train_steps = num_train_steps,
    batch_size = batch_size,
    num_workers = args.num_workers,
    grad_accum_every = 2,
    learning_rate = args.lr,
    val_every_step = args.val_every_epoch * num_step_per_epoch,
    checkpoint_every_setp = args.save_every_epoch * num_step_per_epoch,
    log_every_step = args.log_every_step,
    accelerator_kwargs = dict(
        cpu = False,
        step_scheduler_with_optimizer=False
    ),
    use_wandb_tracking = args.use_wandb_tracking,
    checkpoint_folder = args.model.checkpoint_folder,    
    checkpoint_file_name=args.model.checkpoint_file_name,
    resume_training=args.resume_training,
    num_step_per_epoch = num_step_per_epoch,
    vqvae_ckpt_path=args.model.vqvae_ckpt_path,
    from_start=args.from_start,
)

trainer(project=args.wandb_project_name, run=args.wandb_run_name, hps=cfg)