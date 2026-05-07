import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from preprocess import getDataLoaders
import math
import argparse
import random
import numpy as np
import re
import torch
from train import *

try:
    from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
except ModuleNotFoundError:
    class SummaryWriter(object):
        def __init__(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass
else:
    class SummaryWriter(TorchSummaryWriter):
        def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
            safe_main_tag = re.sub(r'[:*?"<>|]', '_', main_tag)
            return super().add_scalars(safe_main_tag, tag_scalar_dict, global_step, walltime)

def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(data_loader_dict, args, optim_config, cuda, writer, one_subject, seed=3):
    set_seed(seed)
    if args.dataset_name == 'seed3':
        iteration = 7
    elif args.dataset_name == 'seed4':
        iteration = 3
    acc = trainDMMR(data_loader_dict, optim_config, cuda, args, iteration, writer, one_subject)
    return acc

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description='DMMR')

    #config of experiment
    parser.add_argument("--way", type=str, default='DMMR/seed3', help="name of current way")
    parser.add_argument("--index", type=str, default='0', help="tensorboard index")
    parser.add_argument("--max_subjects", type=int, default=None, help="limit the number of target subjects for sanity checks")
    parser.add_argument("--subject_start", type=int, default=0, help="start index of target subjects, inclusive")
    parser.add_argument("--subject_end", type=int, default=None, help="end index of target subjects, exclusive")
    parser.add_argument("--seed", type=int, default=3, help="random seed")

    #config of dataset
    parser.add_argument("--dataset_name", type=str, nargs='?', default='seed3', help="the dataset name, supporting seed3 and seed4")
    parser.add_argument("--session", type=str, nargs='?', default='1', help="selected session")
    parser.add_argument("--subjects", type=int, choices=[15], default=15, help="the number of all subject")
    parser.add_argument("--dim", type=int, default=310, help="dim of input")
    parser.add_argument("--seed3_path", type=str, default="../eeg_data/ExtractedFeatures/", help="path to SEED data root")
    parser.add_argument("--seed4_path", type=str, default="../eeg_data/eeg_feature_smooth/", help="path to SEED-IV data root")
    parser.add_argument("--batch_size", type=int, default=None, help="override batch size")
    parser.add_argument("--time_steps", type=int, default=None, help="override time steps")
    parser.add_argument("--epoch_preTraining", type=int, default=None, help="override epoch of the pre-training phase")
    parser.add_argument("--num_workers_train", type=int, default=None, help="override training dataloader workers")
    parser.add_argument("--num_workers_test", type=int, default=None, help="override test dataloader workers")
    parser.add_argument("--max_train_batches", type=int, default=None, help="limit optimization steps per epoch for quick runs")
    parser.add_argument("--resume", action="store_true", help="resume training from subject checkpoint if available")
    parser.add_argument("--resume_dir", type=str, default="checkpoints", help="directory used to store subject checkpoints")
    parser.add_argument("--ckpt_every_pretrain", type=int, default=1, help="save pretraining checkpoint every N epochs")
    parser.add_argument("--ckpt_every_finetune", type=int, default=1, help="save finetuning checkpoint every N epochs")

    #config of DMMR
    parser.add_argument("--input_dim", type=int, default=310, help="input dim is the same with sample's last dim")
    parser.add_argument("--hid_dim", type=int, default=64, help="hid dim is for hidden layer of lstm")
    parser.add_argument("--n_layers", type=int, default=1, help="num of layers of lstm")
    parser.add_argument("--epoch_fineTuning", type=int, default=500, help="epoch of the fine-tuning phase")
    parser.add_argument("--lr", type=float, default=1e-3, help="epoch of calModel")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    parser.add_argument("--beta", type=float, default=0.05, help="balancing hyperparameter in the loss of pretraining phase")
    parser.add_argument("--use_finetune_ema", type=int, choices=[0, 1], default=0, help="evaluate fine-tuning with EMA-smoothed model weights")
    parser.add_argument("--finetune_ema_decay", type=float, default=0.995, help="EMA decay for fine-tuning model weights")
    parser.add_argument("--finetune_ema_start_epoch", type=int, default=0, help="epoch to start updating fine-tune EMA weights")
    parser.add_argument("--use_contrastive_reg", type=int, choices=[0, 1], default=1, help="enable supervised contrastive regularization in pretraining")
    parser.add_argument("--contrastive_weight", type=float, default=0.1, help="weight for pretraining contrastive regularization")
    parser.add_argument("--contrastive_temperature", type=float, default=0.1, help="temperature for supervised contrastive regularization")
    parser.add_argument("--contrastive_use_proj_head", type=int, choices=[0, 1], default=0, help="use a lightweight projection head before contrastive loss")
    parser.add_argument("--use_rcc", type=int, choices=[0, 1], default=0, help="enable reliability-weighted class-center contrastive loss in fine-tuning")
    parser.add_argument("--rcc_lambda", type=float, default=0.05, help="RCC loss weight")
    parser.add_argument("--rcc_tau", type=float, default=0.2, help="temperature for class-center contrastive logits")
    parser.add_argument("--rcc_reliability_tau", type=float, default=0.5, help="distance temperature for source-subject reliability")
    parser.add_argument("--rcc_warmup_epochs", type=int, default=10, help="epochs before RCC becomes active")
    parser.add_argument("--rcc_ramp_epochs", type=int, default=10, help="epochs to ramp RCC to full weight")
    parser.add_argument("--rcc_ema_momentum", type=float, default=0.9, help="EMA momentum for RCC centers")
    parser.add_argument("--rcc_reliability_min", type=float, default=0.5, help="minimum reliability weight for RCC")
    parser.add_argument("--rcc_reliability_max", type=float, default=1.5, help="maximum reliability weight for RCC")
    parser.add_argument("--rcc_min_valid_samples", type=int, default=4, help="minimum valid samples needed to apply RCC in a batch")
    parser.add_argument("--rcc_update_centers", type=int, choices=[0, 1], default=1, help="whether to update RCC centers online")
    parser.add_argument("--disable_reliability", type=int, choices=[0, 1], default=0, help="disable subject reliability weighting and use plain class-center contrastive loss")
    parser.add_argument("--rcc_init_centers_from_source", type=int, choices=[0, 1], default=0, help="initialize RCC centers from source encoder features before fine-tuning")
    parser.add_argument("--rcc_init_batches", type=int, default=4, help="batches per source subject used to initialize RCC centers; <=0 means all")
    parser.add_argument("--exp_name", type=str, default="default_exp", help="experiment name used in logs and summaries")
    parser.add_argument("--use_source_reliability_weighting", type=int, choices=[0, 1], default=0, help="enable source-wise reliability weighting in fine-tuning")
    parser.add_argument("--srw_tau", type=float, default=0.5, help="temperature for source reliability softmax")
    parser.add_argument("--srw_momentum", type=float, default=0.9, help="EMA momentum for source reliability loss")
    parser.add_argument("--srw_min_weight", type=float, default=0.5, help="minimum source reliability weight")
    parser.add_argument("--srw_max_weight", type=float, default=1.5, help="maximum source reliability weight")
    parser.add_argument("--use_band_mask_recon", type=int, choices=[0, 1], default=0, help="enable band-masked reconstruction auxiliary loss in pretraining")
    parser.add_argument("--band_mask_rate", type=float, default=0.1, help="channel-band mask rate for auxiliary reconstruction")
    parser.add_argument("--band_mask_weight", type=float, default=0.05, help="weight for band-masked reconstruction auxiliary loss")
    parser.add_argument("--use_msr", action="store_true", help="enable Multi-Source Subject Router after encoder in fine-tune/test")
    parser.add_argument("--msr_tau", type=float, default=1.0, help="temperature for source router softmax")
    parser.add_argument("--msr_alpha_init", type=float, default=-1.4, help="raw gate init for MSR alpha; alpha=0.1*sigmoid(raw)")
    parser.add_argument("--msr_hidden_dim", type=int, default=128, help="hidden dimension in MSR router/delta MLPs")
    parser.add_argument("--msr_dropout", type=float, default=0.1, help="dropout in MSR router/delta MLPs")
    parser.add_argument("--msr_memory_init_std", type=float, default=0.02, help="initial std for MSR source memory")
    parser.add_argument("--msr_delta_init_std", type=float, default=1e-3, help="initial std for MSR delta output layer; 0 restores zero-init")
    parser.add_argument("--msr_init_memory_from_prototypes", type=int, choices=[0, 1], default=0, help="initialize MSR memory from source encoder prototypes after pretraining")
    parser.add_argument("--msr_proto_batches", type=int, default=4, help="batches per source subject for MSR prototype initialization; <=0 means all")
    parser.add_argument("--use_class_proto_calib", action="store_true", help="enable fixed source class prototype logits calibration in fine-tune/test")
    parser.add_argument("--proto_alpha", type=float, default=0.1, help="weight for class prototype logits")
    parser.add_argument("--proto_temperature", type=float, default=0.2, help="temperature for class prototype cosine logits")
    parser.add_argument("--proto_learnable_alpha", type=int, choices=[0, 1], default=1, help="whether prototype logit alpha is learnable")
    parser.add_argument("--proto_batches", type=int, default=4, help="batches per source subject to build class prototypes; <=0 means all")
    parser.add_argument("--use_feature_calib", action="store_true", help="enable source feature distribution calibration after encoder")
    parser.add_argument("--feature_calib_alpha", type=float, default=0.5, help="residual strength for feature distribution calibration")
    parser.add_argument("--feature_calib_learnable_alpha", type=int, choices=[0, 1], default=0, help="whether feature calibration alpha is learnable")
    parser.add_argument("--feature_calib_batches", type=int, default=4, help="batches per source subject to estimate feature mean/std; <=0 means all")
    parser.add_argument("--feature_calib_eps", type=float, default=1e-5, help="epsilon for feature distribution calibration std")
    parser.add_argument("--feature_calib_use_std", type=int, choices=[0, 1], default=0, help="use source std in feature calibration; default 0 means source mean-centering only")
    parser.add_argument("--use_rspa", action="store_true", help="enable Reliability-guided Source Prototype Attention after encoder")
    parser.add_argument("--rspa_temperature", type=float, default=0.2, help="temperature for RSPA prototype attention")
    parser.add_argument("--rspa_alpha_init", type=float, default=0.1, help="initial residual alpha for RSPA")
    parser.add_argument("--rspa_alpha_max", type=float, default=0.5, help="maximum bounded residual alpha for RSPA")
    parser.add_argument("--rspa_reliability_tau", type=float, default=1.0, help="distance temperature for RSPA reliability")
    parser.add_argument("--rspa_reliability_min", type=float, default=0.8, help="minimum RSPA prototype reliability")
    parser.add_argument("--rspa_reliability_max", type=float, default=1.2, help="maximum RSPA prototype reliability")
    parser.add_argument("--rspa_hidden_dim", type=int, default=128, help="hidden dimension in RSPA residual MLP")
    parser.add_argument("--rspa_dropout", type=float, default=0.1, help="dropout in RSPA residual MLP")
    parser.add_argument("--rspa_use_warmup", action="store_true", help="warm up RSPA residual injection during fine-tuning")
    parser.add_argument("--rspa_warmup_epochs", type=int, default=2, help="epochs before RSPA residual injection starts")
    parser.add_argument("--rspa_ramp_epochs", type=int, default=4, help="epochs to ramp RSPA injection to full scale")
    parser.add_argument("--rspa_proto_batches", type=int, default=4, help="batches per source subject to build RSPA prototypes; <=0 means all")
    parser.add_argument("--rspa_use_class_hint", action="store_true", help="use current classifier probabilities to class-condition RSPA prototype routing")
    parser.add_argument("--rspa_class_hint_weight", type=float, default=1.0, help="log-probability weight for class-conditioned RSPA routing")
    parser.add_argument("--rspa_class_hint_detach", type=int, choices=[0, 1], default=1, help="detach classifier probabilities before using them as RSPA class hints")
    parser.add_argument("--rspa_filter_low_conf", action="store_true", help="drop low-reliability source-class prototypes from RSPA routing")
    parser.add_argument("--rspa_min_reliability", type=float, default=0.0, help="minimum reliability required when --rspa_filter_low_conf is enabled")
    parser.add_argument("--use_rspa_consistency", action="store_true", help="add KL consistency between classifier predictions before and after RSPA residual")
    parser.add_argument("--rspa_consistency_weight", type=float, default=0.02, help="weight for RSPA pre/post prediction consistency loss")
    parser.add_argument("--rspa_source_balance", action="store_true", help="cap per-source attention mass in RSPA prototype routing")
    parser.add_argument("--rspa_source_cap", type=float, default=0.12, help="maximum attention mass allowed for one source subject before renormalization")
    parser.add_argument("--rspa_adaptive_gate", action="store_true", help="enable sample-wise adaptive residual gate in RSPA")
    parser.add_argument("--rspa_adaptive_gate_min", type=float, default=0.0, help="minimum sample-wise RSPA adaptive gate")
    parser.add_argument("--rspa_adaptive_gate_max", type=float, default=1.0, help="maximum sample-wise RSPA adaptive gate")
    parser.add_argument("--rspa_centered_adaptive_gate", action="store_true", help="enable unity-centered sample-wise RSPA gate: 1 + delta*tanh(net)")
    parser.add_argument("--rspa_centered_gate_delta", type=float, default=0.2, help="maximum deviation around 1.0 for centered adaptive RSPA gate")
    parser.add_argument("--rspa_gate_output_init_std", type=float, default=0.0, help="std for centered adaptive gate output layer; 0 keeps exact unity initialization")
    parser.add_argument("--rspa_logit_blend_weight", type=float, default=0.0, help="blend post-RSPA logits with pre-RSPA logits: logits=(1-w)*post+w*pre")
    parser.add_argument("--use_parallel_tcn", action="store_true", help="enable parallel TCN branch fused with LSTM feature")
    parser.add_argument("--ptcn_hidden_dim", type=int, default=64, help="hidden channels in parallel TCN branch")
    parser.add_argument("--ptcn_layers", type=int, default=2, help="number of dilated temporal conv blocks in parallel TCN branch")
    parser.add_argument("--ptcn_kernel_size", type=int, default=3, help="kernel size for parallel TCN blocks")
    parser.add_argument("--ptcn_dropout", type=float, default=0.1, help="dropout in parallel TCN branch")
    parser.add_argument("--ptcn_alpha_init", type=float, default=0.1, help="initial bounded fusion alpha for parallel TCN branch")
    parser.add_argument("--ptcn_alpha_max", type=float, default=0.3, help="maximum bounded fusion alpha for parallel TCN branch")
    parser.add_argument("--ptcn_delta_init_std", type=float, default=0.01, help="initial std for parallel TCN fusion output layer")
    parser.add_argument("--use_attn_lstm_readout", action="store_true", help="replace last-step LSTM readout with gated temporal attention readout")
    parser.add_argument("--attn_lstm_alpha_init", type=float, default=0.3, help="initial alpha for attentive LSTM readout")
    parser.add_argument("--attn_lstm_alpha_max", type=float, default=1.0, help="maximum alpha for attentive LSTM readout")
    parser.add_argument("--attn_lstm_dropout", type=float, default=0.1, help="dropout in attentive LSTM readout scoring MLP")
    parser.add_argument("--use_mamba_lite", action="store_true", help="enable pure-PyTorch Bi-SSM/Mamba-lite branch fused after encoder")
    parser.add_argument("--mamba_d_model", type=int, default=128, help="hidden dimension in Mamba-lite branch")
    parser.add_argument("--mamba_layers", type=int, default=1, help="number of Mamba-lite selective SSM blocks")
    parser.add_argument("--mamba_kernel_size", type=int, default=3, help="depthwise temporal kernel size in Mamba-lite branch")
    parser.add_argument("--mamba_dropout", type=float, default=0.1, help="dropout in Mamba-lite branch")
    parser.add_argument("--mamba_alpha_init", type=float, default=0.2, help="initial fusion alpha in Mamba-lite branch")
    parser.add_argument("--mamba_alpha_max", type=float, default=0.8, help="maximum bounded fusion alpha in Mamba-lite branch")
    parser.add_argument("--mamba_delta_init_std", type=float, default=0.01, help="initial std for Mamba-lite fusion output layer")
    parser.add_argument("--use_eeg_conformer", action="store_true", help="enable EEG-Conformer-style side branch fused after encoder")
    parser.add_argument("--eeg_conf_node_dim", type=int, default=32, help="node token dim in EEG-Conformer branch")
    parser.add_argument("--eeg_conf_d_model", type=int, default=128, help="Transformer dim in EEG-Conformer branch")
    parser.add_argument("--eeg_conf_heads", type=int, default=4, help="attention heads in EEG-Conformer branch")
    parser.add_argument("--eeg_conf_layers", type=int, default=1, help="Transformer layers in EEG-Conformer branch")
    parser.add_argument("--eeg_conf_dropout", type=float, default=0.1, help="dropout in EEG-Conformer branch")
    parser.add_argument("--eeg_conf_alpha_init", type=float, default=0.25, help="initial fusion alpha in EEG-Conformer branch")
    parser.add_argument("--eeg_conf_alpha_max", type=float, default=0.8, help="maximum bounded fusion alpha in EEG-Conformer branch")
    parser.add_argument("--eeg_conf_delta_init_std", type=float, default=0.02, help="initial std for EEG-Conformer fusion output layer")
    parser.add_argument("--eeg_conf_use_cls_pool", action="store_true", help="use learnable CLS token pooling in EEG-Conformer branch")
    parser.add_argument("--eeg_conf_warmup_finetune_only", action="store_true", help="disable EEG-Conformer residual in pretrain and warm it up during fine-tune")
    parser.add_argument("--eeg_conf_warmup_epochs", type=int, default=2, help="fine-tune epochs before EEG-Conformer injection starts")
    parser.add_argument("--eeg_conf_ramp_epochs", type=int, default=2, help="fine-tune epochs to ramp EEG-Conformer injection")
    parser.add_argument("--use_patch_transformer", action="store_true", help="enable PatchTST-style temporal patch Transformer branch")
    parser.add_argument("--patch_len", type=int, default=6, help="temporal patch length")
    parser.add_argument("--patch_stride", type=int, default=3, help="temporal patch stride")
    parser.add_argument("--patch_d_model", type=int, default=128, help="Transformer dim in patch branch")
    parser.add_argument("--patch_heads", type=int, default=4, help="attention heads in patch branch")
    parser.add_argument("--patch_layers", type=int, default=1, help="Transformer layers in patch branch")
    parser.add_argument("--patch_dropout", type=float, default=0.1, help="dropout in patch branch")
    parser.add_argument("--patch_alpha_init", type=float, default=0.25, help="initial fusion alpha in patch branch")
    parser.add_argument("--patch_alpha_max", type=float, default=0.8, help="maximum bounded fusion alpha in patch branch")
    parser.add_argument("--patch_delta_init_std", type=float, default=0.02, help="initial std for patch branch fusion output layer")
    parser.add_argument("--use_tgb", action="store_true", help="enable TemporalGraphBlock after ABP")
    parser.add_argument("--use_mst", action="store_true", help="enable Multi-Scale Spatiotemporal Block after ABP")
    parser.add_argument("--use_csgformer", action="store_true", help="enable CSGFormerBlock after ABP")
    parser.add_argument("--use_emt_lite_v2", action="store_true", help="enable EmT-lite-v2 encoder after ABP")
    parser.add_argument("--use_dmmr_hemi", action="store_true", help="enable DMMR-Hemi side branches with weak asymmetry fusion")
    parser.add_argument("--hemi_gate_init", type=float, default=-2.2, help="raw gate init for hemi side-path alpha")
    parser.add_argument("--hemi_dropout", type=float, default=0.1, help="dropout in hemi side MLP")
    parser.add_argument("--hemi_hidden_dim", type=int, default=128, help="hidden dim in hemi side MLP")
    parser.add_argument("--lambda_static", type=float, default=0.7, help="static graph mixing coefficient in CSGFormer")
    parser.add_argument("--gamma_init", type=float, default=0.1, help="initial residual gate gamma in CSGFormer graph block")
    parser.add_argument("--channel_embed_dim", type=int, default=8, help="channel embedding dim in CSFA")
    parser.add_argument("--node_dim", type=int, default=16, help="node feature dim after lift in CSGFormer")
    parser.add_argument("--d_model", type=int, default=128, help="transformer token dim in CSGFormer")
    parser.add_argument("--transformer_layers", type=int, default=1, help="number of temporal transformer layers in CSGFormer")
    parser.add_argument("--transformer_heads", type=int, default=4, help="number of temporal transformer heads in CSGFormer")
    parser.add_argument("--use_temporal_attn_pool", type=int, choices=[0, 1], default=1, help="use temporal attention pooling instead of mean pooling in EmT-lite-v2")
    parser.add_argument("--transformer_friendly_shuffle", type=int, choices=[0, 1], default=1, help="use transformer-friendly time shuffle (no fixed last step)")
    parser.add_argument("--mst_alpha_init", type=float, default=0.1, help="initial residual gate alpha in MST block")
    parser.add_argument("--tgb_num_channels", type=int, default=62, help="channel count in TemporalGraphBlock")
    parser.add_argument("--tgb_num_bands", type=int, default=5, help="band count in TemporalGraphBlock")
    parser.add_argument("--tgb_dropout", type=float, default=0.1, help="dropout rate in TemporalGraphBlock")
    parser.add_argument("--tgb_kernel_size", type=int, default=3, help="temporal depthwise kernel size in TemporalGraphBlock")
    parser.add_argument("--tgb_alpha_init", type=float, default=0.1, help="initial residual gate alpha in TemporalGraphBlock")
    parser.add_argument("--use_gcn_residual", action="store_true", help="enable gated residual for graph aggregation")
    parser.add_argument("--gcn_alpha_init", type=float, default=0.1, help="initial residual gate alpha in graph aggregation")
    parser.add_argument("--gcn_learnable_alpha", type=int, choices=[0, 1], default=1, help="whether graph residual alpha is learnable")
    parser.add_argument("--use_self_loop_prior", action="store_true", help="enable self-loop prior for graph adjacency")
    parser.add_argument("--self_loop_weight", type=float, default=0.1, help="self-loop prior weight added before softmax")
    parser.add_argument("--use_pre_lstm_dropout", action="store_true", help="enable dropout before feeding TGB output to LSTM")
    parser.add_argument("--pre_lstm_dropout_p", type=float, default=0.1, help="dropout probability before LSTM")
    parser.add_argument("--use_sspb_v2", action="store_true", help="enable SSPB-v2 after encoder in fine-tune/test")
    parser.add_argument("--use_sspb", action="store_true", help="enable SSPB after encoder in fine-tune/test")
    parser.add_argument("--num_subjects_total", type=int, default=None, help="total number of subjects for prompt bank")
    parser.add_argument("--prompt_tau", type=float, default=2.0, help="temperature for SSPB-v2 prompt attention")
    parser.add_argument("--prompt_alpha_max", type=float, default=0.2, help="max bounded alpha in SSPB-v2")
    parser.add_argument("--prompt_beta_max", type=float, default=0.3, help="max bounded beta in SSPB-v2")
    parser.add_argument("--prompt_alpha_init", type=float, default=0.1, help="initial residual gate alpha in prompt bank")
    parser.add_argument("--prompt_beta_init", type=float, default=0.1, help="initial residual gate beta in prompt bank")
    parser.add_argument("--prompt_dropout", type=float, default=0.0, help="dropout for prompt context")
    parser.add_argument("--use_zero_init_prompt_residual", type=int, choices=[0, 1], default=1, help="use zero-init residual fusion in SSPB-v2")
    parser.add_argument("--prompt_fusion_dropout", type=float, default=0.1, help="dropout inside SSPB fusion net")
    parser.add_argument("--stable_adj_alpha", type=float, default=0.2, help="stable adjacency mixing ratio for TGB")
    parser.add_argument("--use_prompt_ortho_loss", type=int, choices=[0, 1], default=1, help="add prompt orthogonality loss in fine-tuning")
    parser.add_argument("--prompt_ortho_weight", type=float, default=0.01, help="weight for prompt orthogonality loss")
    parser.add_argument("--use_sspb_differential_lr", type=int, choices=[0, 1], default=0, help="use param-group learning rates in fine-tuning")
    parser.add_argument("--backbone_lr", type=float, default=3e-4, help="learning rate for DMMR backbone param group in fine-tuning")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="learning rate for SSPB+classifier param group in fine-tuning")
    parser.add_argument("--freeze_encoder_first_epoch", type=int, choices=[0, 1], default=0, help="freeze encoder only for first fine-tuning epoch")
    parser.add_argument("--sspb_lr", type=float, default=None, help="learning rate for SSPB param group; defaults to 10x base lr when differential lr enabled")
    parser.add_argument("--prompt_gate_init", type=float, default=0.01, help="initial gate value for SSPB residual fusion")
    parser.add_argument("--use_prompt_gate_warmup", action="store_true", help="enable SSPB injection warmup/ramp in fine-tuning")
    parser.add_argument("--prompt_warmup_epochs", type=int, default=2, help="epochs before SSPB injection starts")
    parser.add_argument("--prompt_ramp_epochs", type=int, default=2, help="epochs to ramp SSPB injection to full scale")
    parser.add_argument("--use_prompt_fusion_detach", action="store_true", help="detach eeg feature in SSPB fusion branch")
    parser.add_argument("--use_hyp_contrast", action="store_true", help="enable hyperbolic contrastive auxiliary head in fine-tuning")
    parser.add_argument("--hyp_proj_dim", type=int, default=32, help="projection dimension for hyperbolic contrastive head")
    parser.add_argument("--hyp_temperature", type=float, default=0.1, help="temperature for supervised hyperbolic contrastive loss")
    parser.add_argument("--hyp_loss_weight", type=float, default=0.1, help="weight for hyperbolic contrastive auxiliary loss")
    parser.add_argument("--hyp_curvature", type=float, default=1.0, help="curvature for Poincare distance")


    args = parser.parse_args()
    if args.use_sspb:
        args.use_sspb_v2 = True
    if args.use_dmmr_hemi:
        args.use_tgb = False
        args.use_mst = False
        args.use_csgformer = False
        args.use_emt_lite_v2 = False
        args.use_sspb_v2 = False
        args.use_hyp_contrast = False
    args.source_subjects = args.subjects-1
    if args.num_subjects_total is None:
        args.num_subjects_total = args.subjects
    if args.num_workers_train is None:
        args.num_workers_train = 4 if cuda else 0
    if args.num_workers_test is None:
        args.num_workers_test = 2 if cuda else 0
    if args.dataset_name == "seed3":
        args.path = args.seed3_path
        args.cls_classes = 3
        if args.time_steps is None:
            args.time_steps = 30
        if args.batch_size is None:
            args.batch_size = 512  #batch_size
        if args.epoch_preTraining is None:
            args.epoch_preTraining = 300  #epoch of the pre-training phase
    elif args.dataset_name == "seed4":
        args.path = args.seed4_path
        args.cls_classes = 4
        if args.time_steps is None:
            args.time_steps = 10
        if args.batch_size is None:
            args.batch_size = 256  #batch_size
        if args.epoch_preTraining is None:
            args.epoch_preTraining = 400  #epoch of the pre-training phase
    else:
        print("need to define the input dataset")
    optim_config = {"lr": args.lr, "weight_decay": args.weight_decay}
    # leave-one-subject-out cross-validation
    acc_list=[]
    writer = SummaryWriter("data/session"+args.session+"/"+args.way+"/" + args.index)
    subject_total = args.subjects if args.max_subjects is None else min(args.max_subjects, args.subjects)
    subject_start = max(0, args.subject_start)
    subject_end = subject_total if args.subject_end is None else min(args.subject_end, subject_total)
    for one_subject in range(subject_start, subject_end):
        # 1.data preparation
        source_loaders, test_loader = getDataLoaders(one_subject, args)
        data_loader_dict = {"source_loader": source_loaders, "test_loader":test_loader}
        # 2. main
        acc = main(data_loader_dict, args, optim_config, cuda, writer, one_subject, seed=args.seed)
        writer.add_scalars('single experiment acc: ',
                           {'test acc': acc}, one_subject + 1)
        writer.flush()
        acc_list.append(acc)
    writer.add_text('final acc avg', str(np.average(acc_list)))
    writer.add_text('final acc std', str(np.std(acc_list)))
    acc_list_str = [str(x) for x in acc_list]
    writer.add_text('final each acc', ",".join(acc_list_str))
    writer.add_scalars('final experiment acc scala: /avg',
                       {'test acc': np.average(acc_list)})
    writer.add_scalars('final experiment acc scala: /std',
                       {'test acc': np.std(acc_list)})
    print("final acc avg:", np.average(acc_list))
    print("final acc std:", np.std(acc_list))
    print("final each acc:", ",".join(acc_list_str))
    if bool(getattr(args, "use_rcc", 0)):
        os.makedirs("logs", exist_ok=True)
        summary_path = os.path.join("logs", "rcc_quick_summary.txt")
        with open(summary_path, "a", encoding="utf-8") as fp:
            fp.write(
                "exp_name={exp} subjects={start}-{end} use_rcc={use_rcc} disable_reliability={disable_rel} "
                "rcc_lambda={lam} tau={tau} each_acc=[{each}] avg_acc={avg:.6f} std_acc={std:.6f}\n".format(
                    exp=args.exp_name,
                    start=subject_start,
                    end=subject_end,
                    use_rcc=int(bool(getattr(args, "use_rcc", 0))),
                    disable_rel=int(bool(getattr(args, "disable_reliability", 0))),
                    lam=float(getattr(args, "rcc_lambda", 0.05)),
                    tau=float(getattr(args, "rcc_tau", 0.2)),
                    each=",".join(acc_list_str),
                    avg=float(np.average(acc_list)),
                    std=float(np.std(acc_list)),
                )
            )
        print("rcc_summary_path:", summary_path)
    writer.close()
