# DynGraphEval patched version of train_link_prediction.py
#
# Changes vs upstream lxd99/TGB_TPNet:
#   - Val/test DataLoader batch_size = 5 for tgbl-wiki / tgbl-review (was 20).
#     20 * 999 negatives still OOMs on 40 GB GPUs after a full training epoch.
#     5 * 999 = 4,995 negatives per forward pass fits comfortably.
#   - stdout progress: epoch start/end, loss every LOG_EVERY_N_BATCHES steps,
#     val MRR printed after each epoch. The upstream logger sends INFO to file
#     only (console handler is WARNING), so nothing was visible during training.
#   - torch.cuda.empty_cache() after training epoch and after each val pass.
#   - GPU memory usage printed after each epoch (not just at run end).

import logging
import time
import sys
import os
from pathlib import Path

os.environ["WANDB_MODE"] = 'disabled'
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "120"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
project_path = Path(__file__).parent.resolve()
os.environ['WANDB_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_CACHE_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_CONFIG_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_DATA_DIR'] = f'{project_path}/wandb'

from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.TPNet import TPNet, RandomProjectionModule
from models.NAT import NAT
from models.modules import LinkPredictor_v1, LinkPredictor_v2
from utils.utils import set_thread, set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import WandbLinkLogger
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.metrics import LossFunction
from tgb.linkproppred.evaluate import Evaluator
import pickle as pk
import wandb

# Print training loss to stdout every this many batches
LOG_EVERY_N_BATCHES = 200

# Validation DataLoader batch size for large datasets (edges per batch × 999 negatives)
VAL_BATCH_SIZE = 10

# Only validate every N epochs (saves time; early stopping still works)
VAL_EVERY_N_EPOCHS = 5

# Use this fraction of val edges for early stopping (1.0 = full val set)
VAL_SUBSET = 0.1


def _resume_ckpt_path(saved_models_dir: str, prefix: str, epoch: int) -> str:
    return os.path.join(saved_models_dir, f"{prefix}_resume_epoch{epoch}.pkl")


def _save_resume_checkpoint(saved_models_dir, prefix, epoch, model, optimizer,
                             memory_bank_backup, early_stopping, model_name):
    """Save everything needed to resume training from the next epoch."""
    path = _resume_ckpt_path(saved_models_dir, prefix, epoch)
    payload = {
        'epoch':           epoch,
        'model':           model.state_dict(),
        'optimizer':       optimizer.state_dict(),
        'early_stopping':  {
            'counter':      early_stopping.counter,
            'best_metrics': early_stopping.best_metrics,
            'best_epoch':   early_stopping.best_epoch,
        },
    }
    if model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
        payload['memory_bank'] = memory_bank_backup
    torch.save(payload, path)
    print(f"  [ckpt] Resume checkpoint saved → {os.path.basename(path)}", flush=True)


def _find_latest_resume_checkpoint(saved_models_dir: str, prefix: str):
    """Return (epoch, path) of the latest resume checkpoint, or (None, None)."""
    import glob
    pattern = os.path.join(saved_models_dir, f"{prefix}_resume_epoch*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None, None
    # extract epoch numbers and pick the highest
    def _epoch(p):
        name = os.path.basename(p)
        return int(name.replace(f"{prefix}_resume_epoch", "").replace(".pkl", ""))
    files.sort(key=_epoch)
    latest = files[-1]
    return _epoch(latest), latest


def _load_resume_checkpoint(path, model, optimizer, early_stopping, model_name, device):
    """Load resume checkpoint and restore model/optimizer/memory/early-stopping state."""
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    es = payload['early_stopping']
    early_stopping.counter      = es['counter']
    early_stopping.best_metrics = es['best_metrics']
    early_stopping.best_epoch   = es['best_epoch']
    mem_backup = payload.get('memory_bank', None)
    return mem_backup  # caller restores this into memory_bank


def _gpu_mem_str(device):
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    return f"  GPU mem: {alloc:.1f}/{reserved:.1f} GB (alloc/reserved)"


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    args = get_link_prediction_args(is_evaluation=False)

    # ── Logger: DEBUG to file, WARNING to console (unchanged from upstream) ──
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        f"./logs/{args.prefix}_link_{args.dataset_name}_{args.model_name}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    # ── Data ─────────────────────────────────────────────────────────────────
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, \
        eval_neg_edge_sampler, eval_metric_name = \
        get_link_prediction_data(dataset_name=args.dataset_name, logger=logger)

    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor, seed=0)
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor, seed=1)

    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids,
        interact_times=train_data.node_interact_times,
        last_observed_time=train_data.node_interact_times[0],
        negative_sample_strategy=args.train_negative_sample_strategy,
        seed=None if args.train_negative_sample_strategy in ('random', 'new_random') else 0)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size, shuffle=False)

    # Use a small val batch size for large datasets to avoid OOM:
    # each batch of B edges is scored against B*999 negatives simultaneously.
    if args.dataset_name in ("tgbl-wiki", "tgbl-review"):
        _val_bs = VAL_BATCH_SIZE
    else:
        _val_bs = args.batch_size

    # Use a subset of val for early stopping — representative but much faster.
    # Full val is only used for the final evaluation after training completes.
    _val_n = int(len(val_data.src_node_ids) * VAL_SUBSET)
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(_val_n)),
        batch_size=_val_bs, shuffle=False)
    val_idx_data_loader_full = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=_val_bs, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=_val_bs, shuffle=False)

    print(f"\nDataset:        {args.dataset_name}")
    print(f"Train edges:    {len(train_data.src_node_ids):,}")
    print(f"Val edges:      {len(val_data.src_node_ids):,}  (using {VAL_SUBSET*100:.0f}% = {_val_n:,} for early stopping)")
    print(f"Train batch:    {args.batch_size}  |  Val batch: {_val_bs}  (× 999 negatives)")
    print(f"Batches/epoch:  train={len(train_idx_data_loader):,}  val={len(val_idx_data_loader):,}")
    print(f"Val frequency:  every {VAL_EVERY_N_EPOCHS} epoch(s)")

    evaluator = Evaluator(name=args.dataset_name)
    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):
        set_random_seed(seed=run, deterministic_alg=args.use_random_projection or args.model_name == 'NAT')
        set_thread(3)
        args.seed = run

        run_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{args.num_runs}  |  model: {args.model_name}  |  dataset: {args.dataset_name}")
        print(f"{'='*60}")
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f'configuration is {args}')

        # ── Build model ───────────────────────────────────────────────────────
        random_projections = None
        if args.use_random_projection:
            random_projections = RandomProjectionModule(
                node_num=node_raw_features.shape[0], edge_num=edge_raw_features.shape[0],
                dim_factor=args.rp_dim_factor, num_layer=args.rp_num_layer,
                time_decay_weight=args.rp_time_decay_weight, device=args.device,
                use_matrix=args.rp_use_matrix, beginning_time=train_data.node_interact_times[0],
                not_scale=args.rp_not_scale, enforce_dim=args.enforce_dim)

        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(
                node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                output_dim=args.output_dim, num_layers=args.num_layers,
                num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
            src_node_mean_time_shift, src_node_std_time_shift, \
                dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(
                    train_data.src_node_ids, train_data.dst_node_ids,
                    train_data.node_interact_times)
            dynamic_backbone = MemoryModel(
                node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler, output_dim=args.output_dim,
                time_feat_dim=args.time_feat_dim, model_name=args.model_name,
                num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                src_node_mean_time_shift=src_node_mean_time_shift,
                src_node_std_time_shift=src_node_std_time_shift,
                dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                dst_node_std_time_shift=dst_node_std_time_shift, device=args.device,
                beta=args.pint_beta, num_hop=args.pint_hop)
        elif args.model_name == 'TPNet':
            dynamic_backbone = TPNet(
                node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                output_dim=args.output_dim,
                random_projections=None if args.encode_not_rp else random_projections,
                num_neighbors=args.num_neighbors, num_layers=args.num_layers,
                dropout=args.dropout, device=args.device, not_embedding=args.not_embedding)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(
                node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                output_dim=args.output_dim, position_feat_dim=args.position_feat_dim,
                walk_length=args.walk_length, num_walk_heads=args.num_walk_heads,
                dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(
                node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(
                node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                output_dim=args.output_dim, num_tokens=args.num_neighbors,
                num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(
                node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                output_dim=args.output_dim, channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size, num_layers=args.num_layers, num_heads=args.num_heads,
                dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length,
                device=args.device)
        elif args.model_name == 'NAT':
            dynamic_backbone = NAT(
                n_feat=node_raw_features, e_feat=edge_raw_features, time_dim=args.time_feat_dim,
                output_dim=args.output_dim, num_neighbors=[1] + args.nat_num_neighbors,
                dropout=args.dropout, n_hops=args.num_layers, ngh_dim=args.nat_ngh_dim,
                device=args.device)
            dynamic_backbone.set_seed(args.seed)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        if args.model_name == 'NAT':
            link_predictor = LinkPredictor_v2(
                input_dim=args.output_dim + dynamic_backbone.self_dim * 2,
                hidden_dim=args.output_dim + dynamic_backbone.self_dim * 2, output_dim=1)
        else:
            link_predictor = LinkPredictor_v1(
                input_dim1=args.output_dim, input_dim2=args.output_dim,
                hidden_dim=args.output_dim, output_dim=1,
                random_projections=None if args.decode_not_rp else random_projections,
                not_encode=args.not_encode)

        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B')

        optimizer = create_optimizer(
            model=model, optimizer_name=args.optimizer,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_path = (
            f"./saved_models/{args.prefix}_link_{args.dataset_name}"
            f"_{args.model_name}_seed{args.seed}.pkl"
        )
        early_stopping = EarlyStopping(
            patience=args.patience, save_model_path=save_model_path,
            logger=logger, model_name=args.model_name)

        loss_func = nn.BCEWithLogitsLoss()
        train_loss_fn = LossFunction(args.train_loss_type)
        wandb_logger = WandbLinkLogger('run', args)
        wandb_logger.watch(model)

        # ── Resume from checkpoint if available ───────────────────────────────
        saved_models_dir = "./saved_models"
        resume_epoch, resume_path = _find_latest_resume_checkpoint(saved_models_dir, args.prefix)
        start_epoch = 0
        resume_memory_backup = None
        if resume_path:
            print(f"\nResuming from checkpoint: {os.path.basename(resume_path)}", flush=True)
            resume_memory_backup = _load_resume_checkpoint(
                resume_path, model, optimizer, early_stopping,
                args.model_name, args.device)
            start_epoch = resume_epoch + 1
            print(f"  Resuming from epoch {start_epoch + 1}/{args.num_epochs} | "
                  f"best_epoch={early_stopping.best_epoch} | "
                  f"patience_counter={early_stopping.counter}/{args.patience}", flush=True)
        else:
            print(f"\nNo resume checkpoint found — starting from epoch 1.", flush=True)

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(start_epoch, args.num_epochs):
            epoch_start = time.time()
            print(f"\n── Epoch {epoch + 1}/{args.num_epochs} ──────────────────────────")

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'TPNet', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'PINT']:
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                if resume_memory_backup is not None:
                    # Restore memory state from resume checkpoint (first resumed epoch only)
                    model[0].memory_bank.reload_memory_bank(resume_memory_backup)
                    resume_memory_backup = None
                    print(f"  [ckpt] Memory bank restored from checkpoint.", flush=True)
                else:
                    model[0].memory_bank.__init_memory_bank__()
            if args.model_name == 'NAT':
                model[0].init_ncache()
            if args.use_random_projection:
                random_projections.reset_random_projections()

            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], \
                    train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], \
                    train_data.edge_ids[train_data_indices]

                batch_neg_src_node_ids, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids) * args.train_neg_num,
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                    current_batch_start_time=batch_node_interact_times[0],
                    current_batch_end_time=batch_node_interact_times[-1])
                batch_neg_node_interact_times = np.repeat(batch_node_interact_times, args.train_neg_num)

                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times, num_neighbors=args.num_neighbors)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_neg_node_interact_times, num_neighbors=args.num_neighbors)
                elif args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_neg_node_interact_times,
                            edge_ids=None, edges_are_positive=False, num_neighbors=args.num_neighbors)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            edge_ids=batch_edge_ids, edges_are_positive=True, num_neighbors=args.num_neighbors)
                elif args.model_name in ['GraphMixer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors, time_gap=args.time_gap)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_neg_node_interact_times,
                            num_neighbors=args.num_neighbors, time_gap=args.time_gap)
                elif args.model_name in ['DyGFormer', 'TPNet']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_neg_node_interact_times)
                elif args.model_name == 'NAT':
                    negative_edge_embeddings = model[0].compute_edge_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_neg_node_interact_times,
                        edge_ids=None, edges_are_positive=False)
                    positive_edge_embeddings = model[0].compute_edge_temporal_embeddings(
                        src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids, edges_are_positive=True)
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                if args.model_name == 'NAT':
                    positive_probabilities = model[1](
                        edge_embeddings=positive_edge_embeddings).squeeze(dim=-1)
                    negative_probabilities = model[1](
                        edge_embeddings=negative_edge_embeddings).squeeze(dim=-1)
                else:
                    positive_probabilities = model[1](
                        src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                        src_node_embeddings=batch_src_node_embeddings,
                        dst_node_embeddings=batch_dst_node_embeddings).squeeze(dim=-1)
                    negative_probabilities = model[1](
                        src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids,
                        src_node_embeddings=batch_neg_src_node_embeddings,
                        dst_node_embeddings=batch_neg_dst_node_embeddings).squeeze(dim=-1)

                if args.use_random_projection:
                    random_projections.update(
                        src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times)

                loss = train_loss_fn.forward(
                    positive_logits=positive_probabilities,
                    negative_logits=negative_probabilities)
                train_losses.append(loss.item())
                input_dict = {
                    "y_pred_pos": positive_probabilities,
                    "y_pred_neg": negative_probabilities.reshape(-1, args.train_neg_num),
                    "eval_metric": [eval_metric_name],
                }
                train_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

                train_idx_data_loader_tqdm.set_description(
                    f'Epoch {epoch + 1} | batch {batch_idx + 1} | loss: {loss.item():.4f}')

                # Print running loss to stdout periodically
                if (batch_idx + 1) % LOG_EVERY_N_BATCHES == 0:
                    elapsed = time.time() - epoch_start
                    mean_loss = np.mean(train_losses)
                    mean_mrr = np.mean([m[eval_metric_name] for m in train_metrics])
                    print(f"  step {batch_idx + 1:>5}/{len(train_idx_data_loader)} "
                          f"| loss: {mean_loss:.4f} | train {eval_metric_name}: {mean_mrr:.4f} "
                          f"| {elapsed:.0f}s{_gpu_mem_str(args.device)}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                    model[0].memory_bank.detach_memory_bank()

            train_elapsed = time.time() - epoch_start
            mean_train_loss = np.mean(train_losses)
            mean_train_mrr = np.mean([m[eval_metric_name] for m in train_metrics])
            print(f"  Train done in {train_elapsed:.0f}s | "
                  f"loss: {mean_train_loss:.4f} | {eval_metric_name}: {mean_train_mrr:.4f}"
                  f"{_gpu_mem_str(args.device)}")

            # Free GPU memory before checkpointing / validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f'Epoch: {epoch + 1}, lr: {optimizer.param_groups[0]["lr"]}, '
                        f'train loss: {mean_train_loss:.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}: '
                            f'{np.mean([m[metric_name] for m in train_metrics]):.4f}')

            # ── Memory backup (needed for both checkpointing and val restore) ─
            mem_backup_for_ckpt = None
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                mem_backup_for_ckpt = model[0].memory_bank.backup_memory_bank()

            # ── Save resume checkpoint after every training epoch ─────────────
            _save_resume_checkpoint(
                saved_models_dir, args.prefix, epoch, model, optimizer,
                mem_backup_for_ckpt, early_stopping, args.model_name)

            # ── Validation (every VAL_EVERY_N_EPOCHS epochs) ──────────────────
            if (epoch + 1) % VAL_EVERY_N_EPOCHS != 0:
                print(f"  Skipping validation this epoch (val every {VAL_EVERY_N_EPOCHS} epochs).")
                continue

            # ── Memory backup before val ──────────────────────────────────────
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                train_backup_memory_bank = mem_backup_for_ckpt
                if args.model_name == 'PINT':
                    train_backup_matrix_memory = model[0].matrix_memory.backup_memory()
            if args.model_name == 'NAT':
                train_backup_ncache = model[0].backup_ncache()
            if args.use_random_projection:
                train_backup_random_projections = random_projections.backup_random_projections()

            print(f"  Validating ({len(val_idx_data_loader)} batches × {_val_bs} edges × 999 negatives, {VAL_SUBSET*100:.0f}% of val)...")
            val_start = time.time()
            val_losses, val_metrics = evaluate_model_link_prediction(
                dataset_name=args.dataset_name, model_name=args.model_name,
                model=model, dtype='val', eval_metric_name=eval_metric_name,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                evaluator=evaluator, evaluate_data=val_data,
                loss_func=loss_func, num_neighbors=args.num_neighbors,
                time_gap=args.time_gap, logger=logger)
            val_elapsed = time.time() - val_start

            mean_val_mrr = np.mean([m[eval_metric_name] for m in val_metrics])
            print(f"  Val done in {val_elapsed:.0f}s | "
                  f"val {eval_metric_name}: {mean_val_mrr:.4f} | "
                  f"val loss: {np.mean(val_losses):.4f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── Restore training memory state ─────────────────────────────────
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)
                del train_backup_memory_bank
                if args.model_name == 'PINT':
                    model[0].matrix_memory.reload_memory(train_backup_matrix_memory)
                    del train_backup_matrix_memory
            if args.model_name == 'NAT':
                model[0].reload_ncache(train_backup_ncache)
                del train_backup_ncache
            if args.use_random_projection:
                random_projections.reload_random_projections(train_backup_random_projections)
                del train_backup_random_projections

            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}: '
                            f'{np.mean([m[metric_name] for m in val_metrics]):.4f}')

            wandb_logger.log_epoch(
                train_losses=train_losses, train_metrics=train_metrics,
                val_losses=val_losses, val_metrics=val_metrics, epoch=epoch)

            val_metric_indicator = [
                (metric_name, np.mean([m[metric_name] for m in val_metrics]), True)
                for metric_name in val_metrics[0].keys()
            ]
            early_stop = early_stopping.step(val_metric_indicator, model, args, epoch + 1)
            if early_stop:
                print(f"  Early stopping triggered (patience={args.patience}).")
                break

        # ── Final evaluation on best checkpoint ───────────────────────────────
        print(f"\nLoading best checkpoint (epoch {early_stopping.best_epoch})...")
        logger.info(f'---------Load the best parameters at epoch {early_stopping.best_epoch}-------')
        early_stopping.load_checkpoint(model)

        logger.info(f'---------get final performance on dataset {args.dataset_name}-------')

        print("Running final val evaluation (full val set)...")
        val_losses, val_metrics = evaluate_model_link_prediction(
            dataset_name=args.dataset_name, model_name=args.model_name,
            model=model, dtype='val', eval_metric_name=eval_metric_name,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=val_idx_data_loader_full,
            evaluate_neg_edge_sampler=eval_neg_edge_sampler,
            evaluator=evaluator, evaluate_data=val_data,
            loss_func=loss_func, num_neighbors=args.num_neighbors,
            time_gap=args.time_gap, logger=logger)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Running final test evaluation...")
        test_losses, test_metrics = evaluate_model_link_prediction(
            dataset_name=args.dataset_name, model_name=args.model_name,
            model=model, dtype='test', eval_metric_name=eval_metric_name,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=test_idx_data_loader,
            evaluate_neg_edge_sampler=eval_neg_edge_sampler,
            evaluator=evaluator, evaluate_data=test_data,
            loss_func=loss_func, num_neighbors=args.num_neighbors,
            time_gap=args.time_gap, logger=logger)

        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            avg = np.mean([m[metric_name] for m in val_metrics])
            logger.info(f'validate {metric_name}: {avg:.4f}')
            val_metric_dict[metric_name] = avg

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            avg = np.mean([m[metric_name] for m in test_metrics])
            logger.info(f'test {metric_name}: {avg:.4f}')
            test_metric_dict[metric_name] = avg

        single_run_time = time.time() - run_start_time
        max_mem_mb = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024 \
            if torch.cuda.is_available() else 0

        print(f"\n{'='*60}")
        print(f"Run {run + 1} complete in {single_run_time/60:.1f} min  |  "
              f"Peak GPU mem: {max_mem_mb:.0f} MB")
        print(f"  Val  {eval_metric_name}: {val_metric_dict.get(eval_metric_name, 0):.4f}")
        print(f"  Test {eval_metric_name}: {test_metric_dict.get(eval_metric_name, 0):.4f}")
        print(f"{'='*60}")

        logger.info(f'Run {run + 1} cost {single_run_time:.2f}s. '
                    f'Max GPU mem: {max_mem_mb:.0f} MB')

        wandb_logger.log_run(
            val_losses=val_losses, val_metrics=val_metrics,
            test_losses=test_losses, test_metrics=test_metrics)
        wandb_logger.finish()

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        result_json = json.dumps({
            "validate metrics": {k: str(v) for k, v in val_metric_dict.items()},
            "test metrics": {k: str(v) for k, v in test_metric_dict.items()},
        }, indent=4)
        save_result_path = (
            f"./saved_results/{args.prefix}_link_{args.dataset_name}"
            f"_{args.model_name}_seed{args.seed}.json"
        )
        with open(save_result_path, 'w') as f:
            f.write(result_json)

    if args.num_runs > 1:
        logger.info(f'-----------metrics over {args.num_runs} runs-----------')
        wandb_logger = WandbLinkLogger('summary', args)
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(
                f'average validate {metric_name}: '
                f'{np.mean([r[metric_name] for r in val_metric_all_runs]):.4f} '
                f'± {np.std([r[metric_name] for r in val_metric_all_runs], ddof=1):.4f}')
            logger.info(
                f'average test {metric_name}: '
                f'{np.mean([r[metric_name] for r in test_metric_all_runs]):.4f} '
                f'± {np.std([r[metric_name] for r in test_metric_all_runs], ddof=1):.4f}')
        wandb_logger.log_final(val_metrics=val_metric_all_runs, test_metrics=test_metric_all_runs)

    sys.exit()
