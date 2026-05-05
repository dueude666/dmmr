import os
import time
from model import *
import numpy as np
from test import *
import torch
import torch.nn.functional as F
from collections import defaultdict
import random
from contrastive_regularization import SubjectInvariantContrastiveLoss
from rcc_loss import RCCLoss
try:
    from sklearn.manifold import TSNE
except ModuleNotFoundError:
    TSNE = None
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

def initialize_msr_memory_from_source_prototypes(preTrainModel, source_loader, cuda, args, writer, one_subject):
    preTrainModel.eval()
    proto_sums = []
    proto_counts = []
    max_batches = getattr(args, "msr_proto_batches", None)
    if max_batches is not None and int(max_batches) <= 0:
        max_batches = None

    with torch.no_grad():
        for source_idx, loader in enumerate(source_loader):
            feat_sum = None
            sample_count = 0
            batch_count = 0
            for source_data, _ in loader:
                if max_batches is not None and batch_count >= int(max_batches):
                    break
                if cuda:
                    source_data = source_data.cuda()
                x = preTrainModel.attentionLayer(source_data, source_data.shape[0], preTrainModel.time_steps)
                if getattr(preTrainModel, "use_dmmr_hemi", False):
                    feat, _, _ = preTrainModel._encode_with_hemi(x)
                else:
                    encoded = preTrainModel._encode_sequence(x)
                    feat = encoded[0] if isinstance(encoded, tuple) else encoded
                feat = feat.detach()
                cur_sum = feat.sum(dim=0)
                feat_sum = cur_sum if feat_sum is None else feat_sum + cur_sum
                sample_count += int(feat.shape[0])
                batch_count += 1
            if feat_sum is None or sample_count == 0:
                raise RuntimeError("failed to build MSR prototype for source index {}".format(source_idx))
            proto_sums.append(feat_sum / float(sample_count))
            proto_counts.append(sample_count)

    prototypes = torch.stack(proto_sums, dim=0)
    preTrainModel.multiSourceSubjectRouter.set_subject_memory(prototypes)
    memory_norm = float(prototypes.norm(dim=-1).mean().detach().cpu().item())
    print("msr_memory_init: source_prototypes")
    print("msr_proto_batches:", max_batches if max_batches is not None else "all")
    print("msr_proto_sample_counts:", ",".join(str(v) for v in proto_counts))
    print("msr_proto_memory_norm_mean:", memory_norm)
    writer.add_text("subject {} msr memory init".format(one_subject + 1), "source_prototypes")
    writer.add_text("subject {} msr proto batches".format(one_subject + 1), str(max_batches if max_batches is not None else "all"))
    writer.add_text("subject {} msr proto sample counts".format(one_subject + 1), ",".join(str(v) for v in proto_counts))
    writer.add_text("subject {} msr proto memory norm mean".format(one_subject + 1), str(memory_norm))

def initialize_class_prototypes(fineTuneModel, source_loader, cuda, args, writer, one_subject):
    fineTuneModel.eval()
    num_classes = int(args.cls_classes)
    feature_dim = 64
    proto_sums = None
    proto_counts = torch.zeros(num_classes)
    max_batches = getattr(args, "proto_batches", None)
    if max_batches is not None and int(max_batches) <= 0:
        max_batches = None

    with torch.no_grad():
        for loader in source_loader:
            batch_count = 0
            for source_data, source_label in loader:
                if max_batches is not None and batch_count >= int(max_batches):
                    break
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                x = fineTuneModel.attentionLayer(source_data, source_data.shape[0], fineTuneModel.time_steps)
                if getattr(fineTuneModel, "use_dmmr_hemi", False):
                    feat = fineTuneModel._encode_with_hemi(x)
                else:
                    feat = fineTuneModel._encode_sequence(x)
                labels = source_label.squeeze().long()
                if proto_sums is None:
                    proto_sums = torch.zeros(num_classes, feature_dim, device=feat.device, dtype=feat.dtype)
                    proto_counts = torch.zeros(num_classes, device=feat.device, dtype=feat.dtype)
                for cls_idx in range(num_classes):
                    mask = labels == cls_idx
                    if mask.any():
                        proto_sums[cls_idx] += feat[mask].sum(dim=0)
                        proto_counts[cls_idx] += mask.sum().to(dtype=feat.dtype)
                batch_count += 1

    if proto_sums is None:
        raise RuntimeError("failed to build class prototypes: no source batches were read")
    safe_counts = proto_counts.clamp_min(1.0).unsqueeze(1)
    prototypes = proto_sums / safe_counts
    fineTuneModel.classPrototypeCalibrator.set_prototypes(prototypes, proto_counts)

    proto_norm_mean = float(prototypes.norm(dim=-1).mean().detach().cpu().item())
    counts_list = [int(v) for v in proto_counts.detach().cpu().tolist()]
    print("class_proto_init: source_class_prototypes")
    print("class_proto_batches:", max_batches if max_batches is not None else "all")
    print("class_proto_counts:", ",".join(str(v) for v in counts_list))
    print("class_proto_norm_mean:", proto_norm_mean)
    print("class_proto_alpha_init:", fineTuneModel.classPrototypeCalibrator.get_alpha_value())
    writer.add_text("subject {} class proto init".format(one_subject + 1), "source_class_prototypes")
    writer.add_text("subject {} class proto batches".format(one_subject + 1), str(max_batches if max_batches is not None else "all"))
    writer.add_text("subject {} class proto counts".format(one_subject + 1), ",".join(str(v) for v in counts_list))
    writer.add_text("subject {} class proto norm mean".format(one_subject + 1), str(proto_norm_mean))
    writer.add_text("subject {} class proto alpha init".format(one_subject + 1), str(fineTuneModel.classPrototypeCalibrator.get_alpha_value()))

def initialize_feature_distribution_calibrator(fineTuneModel, source_loader, cuda, args, writer, one_subject):
    fineTuneModel.eval()
    max_batches = getattr(args, "feature_calib_batches", None)
    if max_batches is not None and int(max_batches) <= 0:
        max_batches = None
    sum_feat = None
    sum_sq_feat = None
    sample_count = 0
    with torch.no_grad():
        for loader in source_loader:
            batch_count = 0
            for source_data, _ in loader:
                if max_batches is not None and batch_count >= int(max_batches):
                    break
                if cuda:
                    source_data = source_data.cuda()
                x = fineTuneModel.attentionLayer(source_data, source_data.shape[0], fineTuneModel.time_steps)
                if getattr(fineTuneModel, "use_dmmr_hemi", False):
                    feat = fineTuneModel._encode_with_hemi(x)
                else:
                    feat = fineTuneModel._encode_sequence(x)
                feat = feat.detach()
                cur_sum = feat.sum(dim=0)
                cur_sq = (feat * feat).sum(dim=0)
                sum_feat = cur_sum if sum_feat is None else sum_feat + cur_sum
                sum_sq_feat = cur_sq if sum_sq_feat is None else sum_sq_feat + cur_sq
                sample_count += int(feat.shape[0])
                batch_count += 1
    if sum_feat is None or sample_count == 0:
        raise RuntimeError("failed to initialize feature distribution calibrator")
    mean = sum_feat / float(sample_count)
    var = (sum_sq_feat / float(sample_count)) - mean * mean
    std = torch.sqrt(var.clamp_min(float(getattr(args, "feature_calib_eps", 1e-5))))
    fineTuneModel.featureDistributionCalibrator.set_stats(mean, std)
    print("feature_calib_init: source_feature_mean_std")
    print("feature_calib_batches:", max_batches if max_batches is not None else "all")
    print("feature_calib_sample_count:", sample_count)
    print("feature_calib_alpha_init:", fineTuneModel.featureDistributionCalibrator.get_alpha_value())
    print("feature_calib_use_std:", bool(getattr(args, "feature_calib_use_std", 0)))
    print("feature_calib_mean_norm:", float(mean.norm().detach().cpu().item()))
    print("feature_calib_std_mean:", float(std.mean().detach().cpu().item()))
    print("feature_calib_std_min:", float(std.min().detach().cpu().item()))
    writer.add_text("subject {} feature calib init".format(one_subject + 1), "source_feature_mean_std")
    writer.add_text("subject {} feature calib sample count".format(one_subject + 1), str(sample_count))
    writer.add_text("subject {} feature calib alpha init".format(one_subject + 1), str(fineTuneModel.featureDistributionCalibrator.get_alpha_value()))
    writer.add_text("subject {} feature calib mean norm".format(one_subject + 1), str(float(mean.norm().detach().cpu().item())))
    writer.add_text("subject {} feature calib std mean".format(one_subject + 1), str(float(std.mean().detach().cpu().item())))

def initialize_rspa_prototypes(fineTuneModel, source_loader, cuda, args, writer, one_subject):
    fineTuneModel.eval()
    num_subjects = len(source_loader)
    num_classes = int(args.cls_classes)
    feature_dim = 64
    max_batches = getattr(args, "rspa_proto_batches", 4)
    if max_batches is not None and int(max_batches) <= 0:
        max_batches = None
    proto_sums = torch.zeros(num_subjects, num_classes, feature_dim)
    proto_counts = torch.zeros(num_subjects, num_classes)
    with torch.no_grad():
        for source_idx, loader in enumerate(source_loader):
            batch_count = 0
            for source_data, source_label in loader:
                if max_batches is not None and batch_count >= int(max_batches):
                    break
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                x = fineTuneModel.attentionLayer(source_data, source_data.shape[0], fineTuneModel.time_steps)
                if getattr(fineTuneModel, "use_dmmr_hemi", False):
                    feat = fineTuneModel._encode_with_hemi(x)
                else:
                    feat = fineTuneModel._encode_sequence(x)
                labels = source_label.squeeze().long()
                if proto_sums.device != feat.device:
                    proto_sums = proto_sums.to(feat.device)
                    proto_counts = proto_counts.to(feat.device)
                for cls_idx in range(num_classes):
                    mask = labels == cls_idx
                    if mask.any():
                        proto_sums[source_idx, cls_idx] += feat[mask].sum(dim=0)
                        proto_counts[source_idx, cls_idx] += mask.sum().to(dtype=proto_counts.dtype)
                batch_count += 1
    safe_counts = proto_counts.clamp_min(1.0).unsqueeze(-1)
    prototypes = proto_sums / safe_counts
    fineTuneModel.reliabilitySourcePrototypeAttention.set_prototypes(prototypes, proto_counts)
    valid_count = int((proto_counts > 0).sum().detach().cpu().item())
    proto_norm = float(prototypes.norm(dim=-1).mean().detach().cpu().item())
    print("rspa_proto_init: source_class_prototypes")
    print("rspa_proto_batches:", max_batches if max_batches is not None else "all")
    print("rspa_valid_source_class_count:", valid_count)
    print("rspa_proto_norm_mean:", proto_norm)
    print("rspa_alpha_init:", fineTuneModel.reliabilitySourcePrototypeAttention.get_alpha_value())
    writer.add_text("subject {} rspa proto init".format(one_subject + 1), "source_class_prototypes")
    writer.add_text("subject {} rspa valid source class count".format(one_subject + 1), str(valid_count))
    writer.add_text("subject {} rspa proto norm mean".format(one_subject + 1), str(proto_norm))

def initialize_rcc_centers(rcc_loss_fn, fineTuneModel, source_loader, cuda, args, writer, one_subject):
    if rcc_loss_fn is None:
        return
    was_training = fineTuneModel.training
    fineTuneModel.eval()
    max_batches = getattr(args, "rcc_init_batches", 4)
    if max_batches is not None and int(max_batches) <= 0:
        max_batches = None
    update_steps = 0
    sample_count = 0
    with torch.no_grad():
        for source_idx, loader in enumerate(source_loader):
            batch_count = 0
            for source_data, source_label in loader:
                if max_batches is not None and batch_count >= int(max_batches):
                    break
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                subject_ids = torch.full(
                    (source_data.shape[0],),
                    source_idx,
                    dtype=torch.long,
                    device=source_data.device,
                )
                x = fineTuneModel.attentionLayer(source_data, source_data.shape[0], fineTuneModel.time_steps)
                if getattr(fineTuneModel, "use_dmmr_hemi", False):
                    feat = fineTuneModel._encode_with_hemi(x)
                else:
                    feat = fineTuneModel._encode_sequence(x)
                rcc_loss_fn.update_centers(
                    features=feat.detach(),
                    labels=source_label.squeeze().detach(),
                    subject_ids=subject_ids,
                )
                update_steps += 1
                sample_count += int(source_data.shape[0])
                batch_count += 1
    if was_training:
        fineTuneModel.train()
    class_count = int(rcc_loss_fn.class_initialized.sum().detach().cpu().item())
    subject_class_count = int(rcc_loss_fn.subject_class_initialized.sum().detach().cpu().item())
    center_norm = float(rcc_loss_fn.class_centers.norm(dim=-1).mean().detach().cpu().item())
    print("rcc_init_centers_from_source:", True)
    print("rcc_init_batches:", max_batches if max_batches is not None else "all")
    print("rcc_init_update_steps:", update_steps)
    print("rcc_init_sample_count:", sample_count)
    print("rcc_init_class_centers:", class_count)
    print("rcc_init_subject_class_centers:", subject_class_count)
    print("rcc_init_center_norm_mean:", center_norm)
    writer.add_text("subject {} rcc init centers".format(one_subject + 1), "source_features")
    writer.add_text("subject {} rcc init update steps".format(one_subject + 1), str(update_steps))
    writer.add_text("subject {} rcc init sample count".format(one_subject + 1), str(sample_count))

def _copy_trainable_state(model):
    return {
        name: param.detach().clone()
        for name, param in model.state_dict().items()
        if torch.is_floating_point(param)
    }

def _update_ema_state(ema_state, model, decay):
    with torch.no_grad():
        model_state = model.state_dict()
        for name, param in model_state.items():
            if name not in ema_state or not torch.is_floating_point(param):
                continue
            ema_state[name].mul_(decay).add_(param.detach(), alpha=(1.0 - decay))

def _build_model_with_state(model, state):
    ema_model = copy.deepcopy(model)
    ema_state_dict = ema_model.state_dict()
    for name, param in state.items():
        if name in ema_state_dict:
            ema_state_dict[name].copy_(param)
    ema_model.load_state_dict(ema_state_dict)
    return ema_model

def trainDMMR(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    # data of source subjects, which is used as the training set
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = DMMRPreTrainingModel(
        cuda,
        number_of_source=len(source_loader),
        number_of_category=args.cls_classes,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
        use_tgb=getattr(args, "use_tgb", False),
        use_mst=getattr(args, "use_mst", False),
        use_csgformer=getattr(args, "use_csgformer", False),
        use_emt_lite_v2=getattr(args, "use_emt_lite_v2", False),
        use_dmmr_hemi=getattr(args, "use_dmmr_hemi", False),
        lambda_static=getattr(args, "lambda_static", 0.7),
        gamma_init=getattr(args, "gamma_init", 0.1),
        channel_embed_dim=getattr(args, "channel_embed_dim", 8),
        node_dim=getattr(args, "node_dim", 16),
        d_model=getattr(args, "d_model", 128),
        transformer_layers=getattr(args, "transformer_layers", 1),
        transformer_heads=getattr(args, "transformer_heads", 4),
        use_temporal_attn_pool=getattr(args, "use_temporal_attn_pool", True),
        transformer_friendly_shuffle=getattr(args, "transformer_friendly_shuffle", True),
        hemi_gate_init=getattr(args, "hemi_gate_init", -2.2),
        hemi_dropout=getattr(args, "hemi_dropout", 0.1),
        hemi_hidden_dim=getattr(args, "hemi_hidden_dim", 128),
        use_band_mask_recon=bool(getattr(args, "use_band_mask_recon", 0)),
        band_mask_rate=getattr(args, "band_mask_rate", 0.1),
        use_msr=bool(getattr(args, "use_msr", False)),
        msr_tau=getattr(args, "msr_tau", 1.0),
        msr_alpha_init=getattr(args, "msr_alpha_init", -1.4),
        msr_hidden_dim=getattr(args, "msr_hidden_dim", 128),
        msr_dropout=getattr(args, "msr_dropout", 0.1),
        msr_memory_init_std=getattr(args, "msr_memory_init_std", 0.02),
        msr_delta_init_std=getattr(args, "msr_delta_init_std", 1e-3),
        use_class_proto_calib=bool(getattr(args, "use_class_proto_calib", False)),
        proto_alpha=getattr(args, "proto_alpha", 0.1),
        proto_temperature=getattr(args, "proto_temperature", 0.2),
        proto_learnable_alpha=bool(getattr(args, "proto_learnable_alpha", 1)),
        use_feature_calib=bool(getattr(args, "use_feature_calib", False)),
        feature_calib_alpha=getattr(args, "feature_calib_alpha", 0.5),
        feature_calib_learnable_alpha=bool(getattr(args, "feature_calib_learnable_alpha", 0)),
        feature_calib_eps=getattr(args, "feature_calib_eps", 1e-5),
        feature_calib_use_std=bool(getattr(args, "feature_calib_use_std", 0)),
        use_rspa=bool(getattr(args, "use_rspa", False)),
        rspa_temperature=getattr(args, "rspa_temperature", 0.2),
        rspa_alpha_init=getattr(args, "rspa_alpha_init", 0.1),
        rspa_alpha_max=getattr(args, "rspa_alpha_max", 0.5),
        rspa_reliability_tau=getattr(args, "rspa_reliability_tau", 1.0),
        rspa_reliability_min=getattr(args, "rspa_reliability_min", 0.8),
        rspa_reliability_max=getattr(args, "rspa_reliability_max", 1.2),
        rspa_hidden_dim=getattr(args, "rspa_hidden_dim", 128),
        rspa_dropout=getattr(args, "rspa_dropout", 0.1),
        rspa_use_warmup=bool(getattr(args, "rspa_use_warmup", False)),
        rspa_warmup_epochs=getattr(args, "rspa_warmup_epochs", 2),
        rspa_ramp_epochs=getattr(args, "rspa_ramp_epochs", 4),
        rspa_use_class_hint=bool(getattr(args, "rspa_use_class_hint", False)),
        rspa_class_hint_weight=getattr(args, "rspa_class_hint_weight", 1.0),
        rspa_class_hint_detach=bool(getattr(args, "rspa_class_hint_detach", 1)),
        rspa_filter_low_conf=bool(getattr(args, "rspa_filter_low_conf", False)),
        rspa_min_reliability=getattr(args, "rspa_min_reliability", 0.0),
        rspa_source_balance=bool(getattr(args, "rspa_source_balance", False)),
        rspa_source_cap=getattr(args, "rspa_source_cap", 0.12),
        rspa_adaptive_gate=bool(getattr(args, "rspa_adaptive_gate", False)),
        rspa_adaptive_gate_min=getattr(args, "rspa_adaptive_gate_min", 0.0),
        rspa_adaptive_gate_max=getattr(args, "rspa_adaptive_gate_max", 1.0),
        rspa_centered_adaptive_gate=bool(getattr(args, "rspa_centered_adaptive_gate", False)),
        rspa_centered_gate_delta=getattr(args, "rspa_centered_gate_delta", 0.2),
        rspa_gate_output_init_std=getattr(args, "rspa_gate_output_init_std", 0.0),
        rspa_logit_blend_weight=getattr(args, "rspa_logit_blend_weight", 0.0),
        use_parallel_tcn=bool(getattr(args, "use_parallel_tcn", False)),
        ptcn_hidden_dim=getattr(args, "ptcn_hidden_dim", 64),
        ptcn_layers=getattr(args, "ptcn_layers", 2),
        ptcn_kernel_size=getattr(args, "ptcn_kernel_size", 3),
        ptcn_dropout=getattr(args, "ptcn_dropout", 0.1),
        ptcn_alpha_init=getattr(args, "ptcn_alpha_init", 0.1),
        ptcn_alpha_max=getattr(args, "ptcn_alpha_max", 0.3),
        ptcn_delta_init_std=getattr(args, "ptcn_delta_init_std", 0.01),
        use_attn_lstm_readout=bool(getattr(args, "use_attn_lstm_readout", False)),
        attn_lstm_alpha_init=getattr(args, "attn_lstm_alpha_init", 0.3),
        attn_lstm_alpha_max=getattr(args, "attn_lstm_alpha_max", 1.0),
        attn_lstm_dropout=getattr(args, "attn_lstm_dropout", 0.1),
        use_mamba_lite=bool(getattr(args, "use_mamba_lite", False)),
        mamba_d_model=getattr(args, "mamba_d_model", 128),
        mamba_layers=getattr(args, "mamba_layers", 1),
        mamba_kernel_size=getattr(args, "mamba_kernel_size", 3),
        mamba_dropout=getattr(args, "mamba_dropout", 0.1),
        mamba_alpha_init=getattr(args, "mamba_alpha_init", 0.2),
        mamba_alpha_max=getattr(args, "mamba_alpha_max", 0.8),
        mamba_delta_init_std=getattr(args, "mamba_delta_init_std", 0.01),
        use_eeg_conformer=bool(getattr(args, "use_eeg_conformer", False)),
        eeg_conf_node_dim=getattr(args, "eeg_conf_node_dim", 32),
        eeg_conf_d_model=getattr(args, "eeg_conf_d_model", 128),
        eeg_conf_heads=getattr(args, "eeg_conf_heads", 4),
        eeg_conf_layers=getattr(args, "eeg_conf_layers", 1),
        eeg_conf_dropout=getattr(args, "eeg_conf_dropout", 0.1),
        eeg_conf_alpha_init=getattr(args, "eeg_conf_alpha_init", 0.25),
        eeg_conf_alpha_max=getattr(args, "eeg_conf_alpha_max", 0.8),
        eeg_conf_delta_init_std=getattr(args, "eeg_conf_delta_init_std", 0.02),
        eeg_conf_use_cls_pool=bool(getattr(args, "eeg_conf_use_cls_pool", False)),
        eeg_conf_warmup_finetune_only=bool(getattr(args, "eeg_conf_warmup_finetune_only", False)),
        eeg_conf_warmup_epochs=getattr(args, "eeg_conf_warmup_epochs", 2),
        eeg_conf_ramp_epochs=getattr(args, "eeg_conf_ramp_epochs", 2),
        use_patch_transformer=bool(getattr(args, "use_patch_transformer", False)),
        patch_len=getattr(args, "patch_len", 6),
        patch_stride=getattr(args, "patch_stride", 3),
        patch_d_model=getattr(args, "patch_d_model", 128),
        patch_heads=getattr(args, "patch_heads", 4),
        patch_layers=getattr(args, "patch_layers", 1),
        patch_dropout=getattr(args, "patch_dropout", 0.1),
        patch_alpha_init=getattr(args, "patch_alpha_init", 0.25),
        patch_alpha_max=getattr(args, "patch_alpha_max", 0.8),
        patch_delta_init_std=getattr(args, "patch_delta_init_std", 0.02),
        mst_alpha_init=getattr(args, "mst_alpha_init", 0.1),
        tgb_num_channels=getattr(args, "tgb_num_channels", 62),
        tgb_num_bands=getattr(args, "tgb_num_bands", 5),
        tgb_dropout=getattr(args, "tgb_dropout", 0.1),
        tgb_kernel_size=getattr(args, "tgb_kernel_size", 3),
        tgb_alpha_init=getattr(args, "tgb_alpha_init", 0.1),
        use_gcn_residual=getattr(args, "use_gcn_residual", False),
        gcn_alpha_init=getattr(args, "gcn_alpha_init", 0.1),
        gcn_learnable_alpha=bool(getattr(args, "gcn_learnable_alpha", 1)),
        use_self_loop_prior=getattr(args, "use_self_loop_prior", False),
        self_loop_weight=getattr(args, "self_loop_weight", 0.1),
        use_pre_lstm_dropout=getattr(args, "use_pre_lstm_dropout", False),
        pre_lstm_dropout_p=getattr(args, "pre_lstm_dropout_p", 0.1),
        stable_adj_alpha=getattr(args, "stable_adj_alpha", 1.0),
        use_sspb_v2=getattr(args, "use_sspb_v2", False),
        num_subjects_total=getattr(args, "num_subjects_total", args.subjects),
        target_subject=one_subject,
        prompt_tau=getattr(args, "prompt_tau", 2.0),
        prompt_alpha_max=getattr(args, "prompt_alpha_max", 0.2),
        prompt_beta_max=getattr(args, "prompt_beta_max", 0.3),
        prompt_alpha_init=getattr(args, "prompt_alpha_init", 0.1),
        prompt_beta_init=getattr(args, "prompt_beta_init", 0.1),
        prompt_dropout=getattr(args, "prompt_dropout", 0.0),
        use_zero_init_prompt_residual=bool(getattr(args, "use_zero_init_prompt_residual", 1)),
        prompt_fusion_dropout=getattr(args, "prompt_fusion_dropout", 0.1),
        prompt_gate_init=getattr(args, "prompt_gate_init", 0.01),
        use_prompt_gate_warmup=getattr(args, "use_prompt_gate_warmup", False),
        prompt_warmup_epochs=getattr(args, "prompt_warmup_epochs", 2),
        prompt_ramp_epochs=getattr(args, "prompt_ramp_epochs", 2),
        use_prompt_fusion_detach=getattr(args, "use_prompt_fusion_detach", False),
        use_hyp_contrast=getattr(args, "use_hyp_contrast", False),
        hyp_proj_dim=getattr(args, "hyp_proj_dim", 32),
        hyp_temperature=getattr(args, "hyp_temperature", 0.1),
        hyp_curvature=getattr(args, "hyp_curvature", 1.0),
    )
    if cuda:
        preTrainModel = preTrainModel.cuda()
    if getattr(args, "use_gcn_residual", False):
        gcn_alpha_init = preTrainModel.temporalGraphBlock.get_gcn_alpha_value()
        print("gcn_alpha_init:", gcn_alpha_init)
        writer.add_text("subject {} gcn alpha init".format(one_subject + 1), str(gcn_alpha_init))
    if getattr(args, "use_mst", False):
        mst_alpha_init = preTrainModel.multiScaleTemporalBlock.get_alpha_value()
        print("mst_alpha_init:", mst_alpha_init)
        writer.add_text("subject {} mst alpha init".format(one_subject + 1), str(mst_alpha_init))
    if getattr(args, "use_parallel_tcn", False):
        ptcn_alpha_init = preTrainModel.parallelTCNBranch.get_alpha_value()
        print("parallel_tcn_enabled:", True)
        print("parallel_tcn_alpha_init:", ptcn_alpha_init)
        print("parallel_tcn_hidden_dim:", getattr(args, "ptcn_hidden_dim", 64))
        print("parallel_tcn_layers:", getattr(args, "ptcn_layers", 2))
        print("parallel_tcn_delta_init_std:", getattr(args, "ptcn_delta_init_std", 0.01))
        writer.add_text("subject {} parallel tcn enabled".format(one_subject + 1), "True")
        writer.add_text("subject {} parallel tcn alpha init".format(one_subject + 1), str(ptcn_alpha_init))
    if getattr(args, "use_attn_lstm_readout", False):
        attn_alpha_init = preTrainModel.attentiveSharedEncoder.get_alpha_value()
        print("attn_lstm_readout_enabled:", True)
        print("attn_lstm_alpha_init:", attn_alpha_init)
        writer.add_text("subject {} attn lstm enabled".format(one_subject + 1), "True")
        writer.add_text("subject {} attn lstm alpha init".format(one_subject + 1), str(attn_alpha_init))
    if getattr(args, "use_mamba_lite", False):
        mamba_alpha_init = preTrainModel.mambaLiteBranch.get_alpha_value()
        print("mamba_lite_enabled:", True)
        print("mamba_lite_alpha_init:", mamba_alpha_init)
        print("mamba_lite_d_model:", getattr(args, "mamba_d_model", 128))
        print("mamba_lite_layers:", getattr(args, "mamba_layers", 1))
        print("mamba_lite_delta_init_std:", getattr(args, "mamba_delta_init_std", 0.01))
        writer.add_text("subject {} mamba lite enabled".format(one_subject + 1), "True")
        writer.add_text("subject {} mamba lite alpha init".format(one_subject + 1), str(mamba_alpha_init))
    if getattr(args, "use_eeg_conformer", False):
        eeg_conf_alpha_init = preTrainModel.eegConformerBranch.get_alpha_value()
        print("eeg_conformer_enabled:", True)
        print("eeg_conformer_alpha_init:", eeg_conf_alpha_init)
        print("eeg_conformer_node_dim:", getattr(args, "eeg_conf_node_dim", 32))
        print("eeg_conformer_d_model:", getattr(args, "eeg_conf_d_model", 128))
        print("eeg_conformer_layers:", getattr(args, "eeg_conf_layers", 1))
        print("eeg_conformer_delta_init_std:", getattr(args, "eeg_conf_delta_init_std", 0.02))
        print("eeg_conformer_use_cls_pool:", bool(getattr(args, "eeg_conf_use_cls_pool", False)))
        print("eeg_conformer_warmup_finetune_only:", bool(getattr(args, "eeg_conf_warmup_finetune_only", False)))
        print("eeg_conformer_warmup_epochs:", getattr(args, "eeg_conf_warmup_epochs", 2))
        print("eeg_conformer_ramp_epochs:", getattr(args, "eeg_conf_ramp_epochs", 2))
        writer.add_text("subject {} eeg conformer enabled".format(one_subject + 1), "True")
        writer.add_text("subject {} eeg conformer alpha init".format(one_subject + 1), str(eeg_conf_alpha_init))
    if getattr(args, "use_patch_transformer", False):
        patch_alpha_init = preTrainModel.patchTransformerBranch.get_alpha_value()
        print("patch_transformer_enabled:", True)
        print("patch_transformer_alpha_init:", patch_alpha_init)
        print("patch_len:", getattr(args, "patch_len", 6))
        print("patch_stride:", getattr(args, "patch_stride", 3))
        print("patch_d_model:", getattr(args, "patch_d_model", 128))
        print("patch_layers:", getattr(args, "patch_layers", 1))
        writer.add_text("subject {} patch transformer enabled".format(one_subject + 1), "True")
        writer.add_text("subject {} patch transformer alpha init".format(one_subject + 1), str(patch_alpha_init))
    if getattr(args, "use_msr", False):
        msr_alpha_init = preTrainModel.multiSourceSubjectRouter.get_alpha_value()
        print("msr_enabled:", True)
        print("msr_num_sources:", len(source_loader))
        print("msr_tau:", getattr(args, "msr_tau", 1.0))
        print("msr_alpha_init:", msr_alpha_init)
        print("msr_delta_init_std:", getattr(args, "msr_delta_init_std", 1e-3))
        writer.add_text("subject {} msr enabled".format(one_subject + 1), "True")
        writer.add_text("subject {} msr num sources".format(one_subject + 1), str(len(source_loader)))
        writer.add_text("subject {} msr tau".format(one_subject + 1), str(getattr(args, "msr_tau", 1.0)))
        writer.add_text("subject {} msr alpha init".format(one_subject + 1), str(msr_alpha_init))
        writer.add_text("subject {} msr delta init std".format(one_subject + 1), str(getattr(args, "msr_delta_init_std", 1e-3)))
    if getattr(args, "use_csgformer", False):
        gamma_init = preTrainModel.csgformerBlock.get_gamma_value()
        static_stats = preTrainModel.csgformerBlock.get_static_adj_stats()
        print("csg_gamma_init:", gamma_init)
        print("a_static_diag_mean:", static_stats["diag_mean"])
        print("a_static_offdiag_mean:", static_stats["offdiag_mean"])
        writer.add_text("subject {} csg gamma init".format(one_subject + 1), str(gamma_init))
        writer.add_text("subject {} a static diag mean".format(one_subject + 1), str(static_stats["diag_mean"]))
        writer.add_text("subject {} a static offdiag mean".format(one_subject + 1), str(static_stats["offdiag_mean"]))
    if getattr(args, "use_emt_lite_v2", False):
        gamma_graph_init = preTrainModel.emtLiteV2Encoder.get_gamma_graph_value()
        static_stats = preTrainModel.emtLiteV2Encoder.get_static_adj_stats()
        print("emt_gamma_graph_init:", gamma_graph_init)
        print("emt_a_static_diag_mean:", static_stats["diag_mean"])
        print("emt_a_static_offdiag_mean:", static_stats["offdiag_mean"])
        writer.add_text("subject {} emt gamma graph init".format(one_subject + 1), str(gamma_graph_init))
        writer.add_text("subject {} emt a static diag mean".format(one_subject + 1), str(static_stats["diag_mean"]))
        writer.add_text("subject {} emt a static offdiag mean".format(one_subject + 1), str(static_stats["offdiag_mean"]))
    if getattr(args, "use_dmmr_hemi", False):
        print("hemi_channel_order:", preTrainModel.hemi_channel_order_name)
        print("hemi_left_count:", preTrainModel.hemi_left_count)
        print("hemi_right_count:", preTrainModel.hemi_right_count)
        print("hemi_midline_count:", preTrainModel.hemi_midline_count)
        print("hemi_alpha_init:", preTrainModel.hemiFusion.get_alpha_value())
        writer.add_text("subject {} hemi channel order".format(one_subject + 1), str(preTrainModel.hemi_channel_order_name))
        writer.add_text("subject {} hemi left count".format(one_subject + 1), str(preTrainModel.hemi_left_count))
        writer.add_text("subject {} hemi right count".format(one_subject + 1), str(preTrainModel.hemi_right_count))
        writer.add_text("subject {} hemi midline count".format(one_subject + 1), str(preTrainModel.hemi_midline_count))
        writer.add_text("subject {} hemi alpha init".format(one_subject + 1), str(preTrainModel.hemiFusion.get_alpha_value()))
    if getattr(args, "use_tgb", False):
        print("stable_adj_alpha:", getattr(args, "stable_adj_alpha", 1.0))
        writer.add_text("subject {} stable adj alpha".format(one_subject + 1), str(getattr(args, "stable_adj_alpha", 1.0)))
        adj_stats_init = preTrainModel.temporalGraphBlock.get_adj_norm_stats()
        print("adj_norm_diag_mean_init:", adj_stats_init["diag_mean"])
        print("adj_norm_offdiag_mean_init:", adj_stats_init["offdiag_mean"])
        writer.add_text("subject {} adj norm diag mean init".format(one_subject + 1), str(adj_stats_init["diag_mean"]))
        writer.add_text("subject {} adj norm offdiag mean init".format(one_subject + 1), str(adj_stats_init["offdiag_mean"]))
    if getattr(args, "use_self_loop_prior", False):
        print("self_loop_weight:", args.self_loop_weight)
        writer.add_text("subject {} self loop weight".format(one_subject + 1), str(args.self_loop_weight))
    if getattr(args, "use_pre_lstm_dropout", False):
        print("pre_lstm_dropout_p:", args.pre_lstm_dropout_p)
        writer.add_text("subject {} pre lstm dropout p".format(one_subject + 1), str(args.pre_lstm_dropout_p))
    if getattr(args, "use_sspb_v2", False):
        alpha_init = preTrainModel.sourcePromptBank.get_alpha_value()
        beta_init = preTrainModel.sourcePromptBank.get_beta_value()
        prompt_gate_init = preTrainModel.sourcePromptBank.get_prompt_gate_value()
        zero_init_ok = preTrainModel.sourcePromptBank.is_fusion_last_zero_initialized()
        print("target_subject:", preTrainModel.target_subject)
        print("source_prompt_count:", preTrainModel.source_prompt_count)
        print("prompt_alpha_init:", alpha_init)
        print("prompt_beta_init:", beta_init)
        print("prompt_gate_init:", prompt_gate_init)
        print("prompt_fusion_last_zero_init:", zero_init_ok)
        writer.add_text("subject {} target subject".format(one_subject + 1), str(preTrainModel.target_subject))
        writer.add_text("subject {} source prompt count".format(one_subject + 1), str(preTrainModel.source_prompt_count))
        writer.add_text("subject {} prompt alpha init".format(one_subject + 1), str(alpha_init))
        writer.add_text("subject {} prompt beta init".format(one_subject + 1), str(beta_init))
        writer.add_text("subject {} prompt gate init".format(one_subject + 1), str(prompt_gate_init))
        writer.add_text("subject {} prompt fusion last zero init".format(one_subject + 1), str(zero_init_ok))
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    contrastive_criterion = None
    if bool(getattr(args, "use_contrastive_reg", True)):
        contrastive_criterion = SubjectInvariantContrastiveLoss(
            temperature=getattr(args, "contrastive_temperature", 0.1),
            use_proj_head=bool(getattr(args, "contrastive_use_proj_head", 0)),
            feature_dim=64,
        )
        if cuda:
            contrastive_criterion = contrastive_criterion.cuda()
        print("contrastive_reg_enabled:", True)
        print("contrastive_weight:", getattr(args, "contrastive_weight", 0.1))
        print("contrastive_temperature:", getattr(args, "contrastive_temperature", 0.1))
        print("contrastive_use_proj_head:", bool(getattr(args, "contrastive_use_proj_head", 0)))
    pretrain_params = list(preTrainModel.parameters())
    if contrastive_criterion is not None:
        pretrain_params += list(contrastive_criterion.parameters())
    optimizer_PreTraining = torch.optim.Adam(pretrain_params, **optimizer_config)

    acc_final = 0
    best_epoch = -1
    last_test_acc = 0.0
    max_train_batches = getattr(args, "max_train_batches", None)
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        if contrastive_criterion is not None:
            contrastive_criterion.train()
        data_set_all = 0
        train_steps = 0
        original_pretrain_loss_sum = 0.0
        contrastive_loss_sum = 0.0
        total_pretrain_loss_sum = 0.0
        contrastive_feature_norm_sum = 0.0
        contrastive_positive_pair_sum = 0
        contrastive_valid_ratio_sum = 0.0
        contrastive_non_finite_count = 0
        band_mask_loss_sum = 0.0
        for i in range(1, iteration + 1):
            if max_train_batches is not None and train_steps >= max_train_batches:
                break
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1 # for the gradient reverse layer (GRL)
            batch_dict = defaultdict(list) #Pre-fetch a batch of data for each subject in advance and store them in this dictionary.
            data_dict = defaultdict(list) #Store the data of each subject in the current batch
            label_dict = defaultdict(list) #Store the labels corresponding to the data of each subject in the current batch
            label_data_dict = defaultdict(set)
            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                if max_train_batches is not None and train_steps >= max_train_batches:
                    break
                # Assign a unique ID to each source subject
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()
                #the input of the model
                source_data, source_label = batch_dict[j]
                # Prepare corresponding new batch of each subject, the new batch has same label with current batch.
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                # Store the corresponding new batch of each subject, providing the supervision for different decoders
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data = corres_batch_data.cuda()
                data_set_all += len(source_label)
                optimizer_PreTraining.zero_grad()
                # Call the pretraining model
                if contrastive_criterion is not None:
                    rec_loss, sim_loss, contrastive_feature = preTrainModel(
                        source_data, corres_batch_data, subject_id, m, mark=j, return_feature=True
                    )
                else:
                    rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, m, mark=j)
                    contrastive_feature = None
                # The loss of the pre-training phase, beta is the balancing hyperparameter
                original_pretrain_loss = rec_loss + args.beta * sim_loss
                contrastive_loss = rec_loss.new_zeros(())
                contrastive_stats = None
                if contrastive_criterion is not None:
                    contrastive_loss, contrastive_stats = contrastive_criterion(
                        contrastive_feature, source_label.squeeze(), subject_id
                    )
                band_mask_loss = getattr(preTrainModel, "last_band_mask_loss", rec_loss.new_zeros(()))
                loss_pretrain = original_pretrain_loss + float(getattr(args, "contrastive_weight", 0.1)) * contrastive_loss
                if bool(getattr(args, "use_band_mask_recon", 0)):
                    loss_pretrain = loss_pretrain + float(getattr(args, "band_mask_weight", 0.05)) * band_mask_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
                original_pretrain_loss_sum += float(original_pretrain_loss.detach().cpu().item())
                contrastive_loss_sum += float(contrastive_loss.detach().cpu().item())
                total_pretrain_loss_sum += float(loss_pretrain.detach().cpu().item())
                band_mask_loss_sum += float(band_mask_loss.detach().cpu().item())
                if contrastive_stats is not None:
                    contrastive_feature_norm_sum += float(contrastive_stats["feature_norm_mean"])
                    contrastive_positive_pair_sum += int(contrastive_stats["positive_pair_count"])
                    contrastive_valid_ratio_sum += float(contrastive_stats["valid_positive_ratio"])
                    contrastive_non_finite_count += int(contrastive_stats["non_finite"])
                train_steps += 1
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        step_count = max(train_steps, 1)
        print("original_pretrain_loss_avg:", original_pretrain_loss_sum / step_count)
        print("contrastive_loss_avg:", contrastive_loss_sum / step_count)
        print("band_mask_loss_avg:", band_mask_loss_sum / step_count)
        print("total_pretrain_loss_avg:", total_pretrain_loss_sum / step_count)
        if bool(getattr(args, "use_band_mask_recon", 0)):
            print("band_mask_weight:", getattr(args, "band_mask_weight", 0.05))
            print("band_mask_rate:", getattr(args, "band_mask_rate", 0.1))
            writer.add_scalar('subject: '+str(one_subject+1)+' '+'train DMMR/band mask loss avg',
                              band_mask_loss_sum / step_count, epoch + 1)
        if contrastive_criterion is not None:
            print("contrastive_feature_norm_mean_avg:", contrastive_feature_norm_sum / step_count)
            print("contrastive_positive_pair_count_avg:", contrastive_positive_pair_sum / step_count)
            print("contrastive_valid_positive_ratio_avg:", contrastive_valid_ratio_sum / step_count)
            print("contrastive_non_finite_count:", contrastive_non_finite_count)
            writer.add_scalar('subject: '+str(one_subject+1)+' '+'train DMMR/original pretrain loss avg',
                              original_pretrain_loss_sum / step_count, epoch + 1)
            writer.add_scalar('subject: '+str(one_subject+1)+' '+'train DMMR/contrastive loss avg',
                              contrastive_loss_sum / step_count, epoch + 1)
            writer.add_scalar('subject: '+str(one_subject+1)+' '+'train DMMR/total pretrain loss avg',
                              total_pretrain_loss_sum / step_count, epoch + 1)
            writer.add_scalar('subject: '+str(one_subject+1)+' '+'train DMMR/contrastive feature norm mean',
                              contrastive_feature_norm_sum / step_count, epoch + 1)
            writer.add_scalar('subject: '+str(one_subject+1)+' '+'train DMMR/contrastive positive pair count',
                              contrastive_positive_pair_sum / step_count, epoch + 1)
            writer.add_scalar('subject: '+str(one_subject+1)+' '+'train DMMR/contrastive valid positive ratio',
                              contrastive_valid_ratio_sum / step_count, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    if hasattr(preTrainModel, "last_band_mask_loss"):
        preTrainModel.last_band_mask_loss = None
    if getattr(args, "use_msr", False) and bool(getattr(args, "msr_init_memory_from_prototypes", 0)):
        initialize_msr_memory_from_source_prototypes(preTrainModel, source_loader, cuda, args, writer, one_subject)
    #Load the ABP module, the encoder from pretrained model and build a new model for the fine-tuning phase
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    if getattr(args, "use_sspb_v2", False) and bool(getattr(args, "use_sspb_differential_lr", 0)):
        backbone_modules = [fineTuneModel.attentionLayer, fineTuneModel.sharedEncoder]
        if getattr(fineTuneModel, "use_tgb", False):
            backbone_modules.append(fineTuneModel.temporalGraphBlock)
        head_modules = [fineTuneModel.sourcePromptBank, fineTuneModel.cls_fc]

        backbone_params = []
        for module in backbone_modules:
            backbone_params.extend(list(module.parameters()))
        head_params = []
        for module in head_modules:
            head_params.extend(list(module.parameters()))

        backbone_param_ids = set(map(id, backbone_params))
        head_params = [p for p in head_params if id(p) not in backbone_param_ids]
        backbone_params = [p for p in backbone_params if p.requires_grad]
        head_params = [p for p in head_params if p.requires_grad]

        backbone_lr = float(getattr(args, "backbone_lr", optimizer_config["lr"]))
        head_lr = float(getattr(args, "head_lr", optimizer_config["lr"]))
        backbone_param_count = int(sum(p.numel() for p in backbone_params))
        head_param_count = int(sum(p.numel() for p in head_params))
        print("sspb_differential_lr_enabled:", True)
        print("backbone_lr:", backbone_lr)
        print("head_lr:", head_lr)
        print("backbone_param_count:", backbone_param_count)
        print("head_param_count:", head_param_count)
        writer.add_text("subject {} sspb differential lr enabled".format(one_subject + 1), "True")
        writer.add_text("subject {} backbone lr".format(one_subject + 1), str(backbone_lr))
        writer.add_text("subject {} head lr".format(one_subject + 1), str(head_lr))
        writer.add_text("subject {} backbone param count".format(one_subject + 1), str(backbone_param_count))
        writer.add_text("subject {} head param count".format(one_subject + 1), str(head_param_count))
        optimizer_FineTuning = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": optimizer_config["weight_decay"]},
                {"params": head_params, "lr": head_lr, "weight_decay": optimizer_config["weight_decay"]},
            ]
        )
    else:
        optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    rcc_loss_fn = None
    if bool(getattr(args, "use_rcc", 0)):
        rcc_loss_fn = RCCLoss(
            num_classes=args.cls_classes,
            num_subjects=len(source_loader),
            feature_dim=64,
            rcc_lambda=float(getattr(args, "rcc_lambda", 0.05)),
            tau=float(getattr(args, "rcc_tau", 0.2)),
            reliability_tau=float(getattr(args, "rcc_reliability_tau", 0.5)),
            warmup_epochs=int(getattr(args, "rcc_warmup_epochs", 10)),
            ramp_epochs=int(getattr(args, "rcc_ramp_epochs", 10)),
            ema_momentum=float(getattr(args, "rcc_ema_momentum", 0.9)),
            reliability_min=float(getattr(args, "rcc_reliability_min", 0.5)),
            reliability_max=float(getattr(args, "rcc_reliability_max", 1.5)),
            min_valid_samples=int(getattr(args, "rcc_min_valid_samples", 4)),
            use_reliability=(not bool(getattr(args, "disable_reliability", 0))),
            update_centers=bool(getattr(args, "rcc_update_centers", 1)),
        )
        if cuda:
            rcc_loss_fn = rcc_loss_fn.cuda()
        print("rcc_enabled:", True)
        print("rcc_lambda:", getattr(args, "rcc_lambda", 0.05))
        print("rcc_tau:", getattr(args, "rcc_tau", 0.2))
        print("rcc_reliability_tau:", getattr(args, "rcc_reliability_tau", 0.5))
        print("rcc_disable_reliability:", bool(getattr(args, "disable_reliability", 0)))
        print("rcc_warmup_epochs:", getattr(args, "rcc_warmup_epochs", 10))
        print("rcc_ramp_epochs:", getattr(args, "rcc_ramp_epochs", 10))
        if bool(getattr(args, "rcc_init_centers_from_source", 0)):
            initialize_rcc_centers(rcc_loss_fn, fineTuneModel, source_loader, cuda, args, writer, one_subject)
    if getattr(args, "use_class_proto_calib", False):
        initialize_class_prototypes(fineTuneModel, source_loader, cuda, args, writer, one_subject)
    if getattr(args, "use_feature_calib", False):
        initialize_feature_distribution_calibrator(fineTuneModel, source_loader, cuda, args, writer, one_subject)
    if getattr(args, "use_rspa", False):
        initialize_rspa_prototypes(fineTuneModel, source_loader, cuda, args, writer, one_subject)
    use_srw = bool(getattr(args, "use_source_reliability_weighting", 0))
    source_loss_ema = [1.0 for _ in range(len(source_loader))]
    if use_srw:
        print("source_reliability_weighting_enabled:", True)
        print("srw_tau:", getattr(args, "srw_tau", 0.5))
        print("srw_momentum:", getattr(args, "srw_momentum", 0.9))
        print("srw_min_weight:", getattr(args, "srw_min_weight", 0.5))
        print("srw_max_weight:", getattr(args, "srw_max_weight", 1.5))
    use_finetune_ema = bool(getattr(args, "use_finetune_ema", 0))
    finetune_ema_decay = float(getattr(args, "finetune_ema_decay", 0.995))
    finetune_ema_start_epoch = int(getattr(args, "finetune_ema_start_epoch", 0))
    finetune_ema_state = None
    if use_finetune_ema:
        finetune_ema_state = _copy_trainable_state(fineTuneModel)
        print("finetune_ema_enabled:", True)
        print("finetune_ema_decay:", finetune_ema_decay)
        print("finetune_ema_start_epoch:", finetune_ema_start_epoch)
        print("finetune_ema_param_count:", sum(v.numel() for v in finetune_ema_state.values()))
        writer.add_text("subject {} finetune ema enabled".format(one_subject + 1), "True")
        writer.add_text("subject {} finetune ema decay".format(one_subject + 1), str(finetune_ema_decay))
    for epoch in range(args.epoch_fineTuning):
        if bool(getattr(args, "freeze_encoder_first_epoch", 0)):
            freeze_encoder_now = (epoch == 0)
            for p in fineTuneModel.sharedEncoder.parameters():
                p.requires_grad = (not freeze_encoder_now)
            if freeze_encoder_now:
                print("encoder_freeze_status: frozen (epoch 0)")
                writer.add_text("subject {} encoder freeze epoch {}".format(one_subject + 1, epoch), "frozen")
            else:
                print("encoder_freeze_status: unfrozen (epoch {})".format(epoch))
                writer.add_text("subject {} encoder freeze epoch {}".format(one_subject + 1, epoch), "unfrozen")
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        train_steps = 0
        ortho_loss_sum = 0.0
        ortho_loss_count = 0
        prompt_sim_offdiag_sum = 0.0
        prompt_sim_count = 0
        cls_loss_sum = 0.0
        hyp_loss_sum = 0.0
        total_loss_sum = 0.0
        rcc_loss_raw_sum = 0.0
        rcc_loss_weighted_sum = 0.0
        rcc_lambda_now_sum = 0.0
        rcc_reliability_mean_sum = 0.0
        rcc_reliability_min_epoch = float("inf")
        rcc_reliability_max_epoch = 0.0
        rcc_center_norm_mean_sum = 0.0
        rcc_valid_samples_sum = 0
        rcc_steps = 0
        hyp_count = 0
        z_hyp_norm_mean_sum = 0.0
        z_hyp_norm_max = 0.0
        hyp_non_finite_count = 0
        srw_weight_sum = 0.0
        srw_weight_count = 0
        srw_weight_min = float("inf")
        srw_weight_max = 0.0
        rspa_consistency_sum = 0.0
        rspa_consistency_count = 0
        for i in range(1, iteration + 1):
            if max_train_batches is not None and train_steps >= max_train_batches:
                break
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                if max_train_batches is not None and train_steps >= max_train_batches:
                    break
                source_data, source_label = batch_dict[j]
                batch_subject_id = torch.full((source_data.shape[0],), j, dtype=torch.long)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    batch_subject_id = batch_subject_id.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                # Call the fine-tuning model
                if getattr(args, "use_hyp_contrast", False):
                    x_pred, x_logits, cls_loss, attn, hyp_loss, hyp_stats, shared_feature = fineTuneModel(
                        source_data,
                        source_label,
                        current_epoch=epoch,
                        return_attn=True,
                        return_hyp=True,
                        return_feature=True,
                    )
                else:
                    x_pred, x_logits, cls_loss, attn, shared_feature = fineTuneModel(
                        source_data,
                        source_label,
                        current_epoch=epoch,
                        return_attn=True,
                        return_feature=True,
                    )
                    hyp_loss = cls_loss.new_zeros(())
                    hyp_stats = None
                total_loss = cls_loss
                if getattr(args, "use_hyp_contrast", False):
                    total_loss = total_loss + float(getattr(args, "hyp_loss_weight", 0.1)) * hyp_loss
                if (
                    bool(getattr(args, "use_rspa_consistency", False))
                    and getattr(args, "use_rspa", False)
                    and getattr(fineTuneModel, "rspa_pre_logits", None) is not None
                ):
                    pre_probs = F.softmax(fineTuneModel.rspa_pre_logits.detach(), dim=1)
                    post_log_probs = F.log_softmax(x_logits, dim=1)
                    rspa_consistency_loss = F.kl_div(post_log_probs, pre_probs, reduction="batchmean")
                    total_loss = total_loss + float(getattr(args, "rspa_consistency_weight", 0.02)) * rspa_consistency_loss
                    rspa_consistency_sum += float(rspa_consistency_loss.detach().cpu().item())
                    rspa_consistency_count += 1
                rcc_loss = cls_loss.new_zeros(())
                rcc_stats = None
                if rcc_loss_fn is not None:
                    rcc_loss, rcc_stats = rcc_loss_fn(
                        features=shared_feature,
                        labels=source_label.squeeze(),
                        subject_ids=batch_subject_id,
                        epoch=epoch,
                    )
                    total_loss = total_loss + float(rcc_stats["rcc_lambda_now"]) * rcc_loss
                if getattr(args, "use_sspb_v2", False) and bool(getattr(args, "use_prompt_ortho_loss", 1)):
                    prompts = fineTuneModel.sourcePromptBank.prompt_bank
                    prompts_norm = F.normalize(prompts, p=2, dim=-1)
                    sim_matrix = torch.matmul(prompts_norm, prompts_norm.transpose(0, 1))
                    identity_target = torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
                    ortho_loss = F.mse_loss(sim_matrix, identity_target)
                    total_loss = total_loss + float(getattr(args, "prompt_ortho_weight", 0.01)) * ortho_loss
                    ortho_loss_sum += float(ortho_loss.detach().cpu().item())
                    ortho_loss_count += 1
                    offdiag_mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
                    prompt_sim_offdiag_sum += float(sim_matrix[offdiag_mask].mean().detach().cpu().item())
                    prompt_sim_count += 1
                srw_weight = 1.0
                if use_srw:
                    ema_tensor = torch.tensor(source_loss_ema, dtype=torch.float32)
                    raw_weights = torch.softmax(-ema_tensor / max(float(getattr(args, "srw_tau", 0.5)), 1e-6), dim=0)
                    raw_weights = raw_weights * len(source_loss_ema)
                    raw_weight = float(raw_weights[j].item())
                    srw_weight = min(
                        max(raw_weight, float(getattr(args, "srw_min_weight", 0.5))),
                        float(getattr(args, "srw_max_weight", 1.5)),
                    )
                    total_loss = total_loss * srw_weight
                total_loss.backward()
                optimizer_FineTuning.step()
                if use_finetune_ema and epoch >= finetune_ema_start_epoch:
                    _update_ema_state(finetune_ema_state, fineTuneModel, finetune_ema_decay)
                if rcc_loss_fn is not None:
                    rcc_loss_fn.update_centers(
                        features=shared_feature.detach(),
                        labels=source_label.squeeze().detach(),
                        subject_ids=batch_subject_id.detach(),
                    )
                if use_srw:
                    cls_loss_value = float(cls_loss.detach().cpu().item())
                    srw_momentum = float(getattr(args, "srw_momentum", 0.9))
                    source_loss_ema[j] = srw_momentum * source_loss_ema[j] + (1.0 - srw_momentum) * cls_loss_value
                    srw_weight_sum += srw_weight
                    srw_weight_count += 1
                    srw_weight_min = min(srw_weight_min, srw_weight)
                    srw_weight_max = max(srw_weight_max, srw_weight)
                cls_loss_sum += float(cls_loss.detach().cpu().item())
                hyp_loss_sum += float(hyp_loss.detach().cpu().item())
                total_loss_sum += float(total_loss.detach().cpu().item())
                if rcc_stats is not None:
                    rcc_loss_raw_sum += float(rcc_stats["rcc_loss_raw"])
                    rcc_loss_weighted_sum += float(rcc_stats["rcc_loss_weighted"])
                    rcc_lambda_now_sum += float(rcc_stats["rcc_lambda_now"])
                    rcc_reliability_mean_sum += float(rcc_stats["reliability_mean"])
                    rcc_reliability_min_epoch = min(rcc_reliability_min_epoch, float(rcc_stats["reliability_min"]))
                    rcc_reliability_max_epoch = max(rcc_reliability_max_epoch, float(rcc_stats["reliability_max"]))
                    rcc_center_norm_mean_sum += float(rcc_stats["center_norm_mean"])
                    rcc_valid_samples_sum += int(rcc_stats["valid_rcc_samples"])
                    rcc_steps += 1
                    if rcc_stats.get("warning") and epoch == 0 and j == 0 and train_steps < 3:
                        print("rcc_warning (step {}): {}".format(train_steps, rcc_stats["warning"]))
                if hyp_stats is not None:
                    hyp_count += 1
                    z_hyp_norm_mean_sum += float(hyp_stats["z_hyp_norm_mean"])
                    z_hyp_norm_max = max(z_hyp_norm_max, float(hyp_stats["z_hyp_norm_max"]))
                    if bool(hyp_stats["has_nan_or_inf"]):
                        hyp_non_finite_count += 1
                if getattr(args, "use_sspb_v2", False) and epoch == 0 and j == 0 and train_steps < 3:
                    debug_info = fineTuneModel.sourcePromptBank.get_last_attention_debug()
                    print("prompt attn mean (step {}): {}".format(train_steps, debug_info["attn_mean"]))
                    print("prompt attn top3 (step {}): {}".format(train_steps, debug_info["top3_prompt_indices"]))
                    print("fusion_out_norm (step {}): {}".format(train_steps, debug_info["fusion_out_norm"]))
                    print("feature_delta_norm (step {}): {}".format(train_steps, debug_info["feature_delta_norm"]))
                if getattr(args, "use_csgformer", False) and epoch == 0 and j == 0 and train_steps < 3:
                    dyn_stats = fineTuneModel.csgformerBlock.get_last_dyn_stats()
                    pool_stats = fineTuneModel.csgformerBlock.get_last_pool_stats()
                    print("csg_a_dyn_min (step {}): {}".format(train_steps, dyn_stats["min"]))
                    print("csg_a_dyn_max (step {}): {}".format(train_steps, dyn_stats["max"]))
                    print("csg_a_dyn_has_nan_or_inf (step {}): {}".format(train_steps, dyn_stats["has_nan_or_inf"]))
                    print("csg_alpha_max_mean (step {}): {}".format(train_steps, pool_stats["alpha_max_mean"]))
                    print("csg_alpha_entropy_mean (step {}): {}".format(train_steps, pool_stats["alpha_entropy_mean"]))
                if getattr(args, "use_emt_lite_v2", False) and epoch == 0 and j == 0 and train_steps < 3:
                    emt_stats = fineTuneModel.emtLiteV2Encoder.get_last_stats()
                    print("emt_gamma_min (step {}): {}".format(train_steps, emt_stats["gamma_min"]))
                    print("emt_gamma_max (step {}): {}".format(train_steps, emt_stats["gamma_max"]))
                    print("emt_beta_min (step {}): {}".format(train_steps, emt_stats["beta_min"]))
                    print("emt_beta_max (step {}): {}".format(train_steps, emt_stats["beta_max"]))
                    print("emt_a_dyn_min (step {}): {}".format(train_steps, emt_stats["a_dyn_min"]))
                    print("emt_a_dyn_max (step {}): {}".format(train_steps, emt_stats["a_dyn_max"]))
                    print("emt_a_dyn_has_nan_or_inf (step {}): {}".format(train_steps, emt_stats["a_dyn_has_nan_or_inf"]))
                    print("emt_node_alpha_max_mean (step {}): {}".format(train_steps, emt_stats["node_alpha_max_mean"]))
                    print("emt_node_alpha_entropy_mean (step {}): {}".format(train_steps, emt_stats["node_alpha_entropy_mean"]))
                    print("emt_temp_alpha_max_mean (step {}): {}".format(train_steps, emt_stats["temp_alpha_max_mean"]))
                    print("emt_temp_alpha_entropy_mean (step {}): {}".format(train_steps, emt_stats["temp_alpha_entropy_mean"]))
                if getattr(args, "use_dmmr_hemi", False) and epoch == 0 and j == 0 and train_steps < 3:
                    hemi_stats = fineTuneModel.hemiFusion.get_last_stats()
                    print("hemi_alpha (step {}): {}".format(train_steps, hemi_stats["alpha"]))
                    print("hemi_full_norm_mean (step {}): {}".format(train_steps, hemi_stats["full_norm_mean"]))
                    print("hemi_left_norm_mean (step {}): {}".format(train_steps, hemi_stats["left_norm_mean"]))
                    print("hemi_right_norm_mean (step {}): {}".format(train_steps, hemi_stats["right_norm_mean"]))
                    print("hemi_side_norm_mean (step {}): {}".format(train_steps, hemi_stats["side_norm_mean"]))
                    print("hemi_fused_norm_mean (step {}): {}".format(train_steps, hemi_stats["fused_norm_mean"]))
                    print("hemi_left_right_cos_mean (step {}): {}".format(train_steps, hemi_stats["left_right_cos_mean"]))
                if getattr(args, "use_msr", False) and epoch == 0 and j == 0 and train_steps < 3:
                    msr_stats = fineTuneModel.multiSourceSubjectRouter.get_last_stats()
                    print("msr_alpha (step {}): {}".format(train_steps, msr_stats.get("alpha")))
                    print("msr_router_entropy_mean (step {}): {}".format(train_steps, msr_stats.get("router_entropy_mean")))
                    print("msr_router_entropy_norm_mean (step {}): {}".format(train_steps, msr_stats.get("router_entropy_norm_mean")))
                    print("msr_router_max_mean (step {}): {}".format(train_steps, msr_stats.get("router_max_mean")))
                    print("msr_context_norm_mean (step {}): {}".format(train_steps, msr_stats.get("context_norm_mean")))
                    print("msr_delta_norm_mean (step {}): {}".format(train_steps, msr_stats.get("delta_norm_mean")))
                    print("msr_feature_delta_norm_mean (step {}): {}".format(train_steps, msr_stats.get("feature_delta_norm_mean")))
                    print("msr_top1_counts (step {}): {}".format(train_steps, msr_stats.get("top1_counts")))
                    print("msr_has_nan_or_inf (step {}): {}".format(train_steps, msr_stats.get("has_nan_or_inf")))
                if getattr(args, "use_class_proto_calib", False) and epoch == 0 and j == 0 and train_steps < 3:
                    proto_stats = fineTuneModel.classPrototypeCalibrator.get_last_stats()
                    print("class_proto_alpha (step {}): {}".format(train_steps, proto_stats.get("alpha")))
                    print("class_proto_logit_mean (step {}): {}".format(train_steps, proto_stats.get("proto_logit_mean")))
                    print("class_proto_logit_std (step {}): {}".format(train_steps, proto_stats.get("proto_logit_std")))
                    print("class_proto_entropy_norm_mean (step {}): {}".format(train_steps, proto_stats.get("proto_entropy_norm_mean")))
                    print("class_proto_counts (step {}): {}".format(train_steps, proto_stats.get("prototype_counts")))
                    print("class_proto_has_nan_or_inf (step {}): {}".format(train_steps, proto_stats.get("has_nan_or_inf")))
                if getattr(args, "use_feature_calib", False) and epoch == 0 and j == 0 and train_steps < 3:
                    calib_stats = fineTuneModel.featureDistributionCalibrator.get_last_stats()
                    print("feature_calib_alpha (step {}): {}".format(train_steps, calib_stats.get("alpha")))
                    print("feature_calib_source_mean_norm (step {}): {}".format(train_steps, calib_stats.get("source_mean_norm")))
                    print("feature_calib_source_std_mean (step {}): {}".format(train_steps, calib_stats.get("source_std_mean")))
                    print("feature_calib_use_std (step {}): {}".format(train_steps, calib_stats.get("use_std")))
                    print("feature_calib_feature_norm_mean (step {}): {}".format(train_steps, calib_stats.get("feature_norm_mean")))
                    print("feature_calib_calibrated_norm_mean (step {}): {}".format(train_steps, calib_stats.get("calibrated_norm_mean")))
                    print("feature_calib_delta_norm_mean (step {}): {}".format(train_steps, calib_stats.get("feature_delta_norm_mean")))
                    print("feature_calib_has_nan_or_inf (step {}): {}".format(train_steps, calib_stats.get("has_nan_or_inf")))
                if getattr(args, "use_rspa", False) and epoch == 0 and j == 0 and train_steps < 3:
                    rspa_stats = fineTuneModel.reliabilitySourcePrototypeAttention.get_last_stats()
                    print("rspa_alpha (step {}): {}".format(train_steps, rspa_stats.get("alpha")))
                    print("rspa_valid_prototypes (step {}): {}".format(train_steps, rspa_stats.get("valid_prototypes")))
                    print("rspa_attn_entropy_norm_mean (step {}): {}".format(train_steps, rspa_stats.get("attn_entropy_norm_mean")))
                    print("rspa_attn_max_mean (step {}): {}".format(train_steps, rspa_stats.get("attn_max_mean")))
                    print("rspa_reliability_mean (step {}): {}".format(train_steps, rspa_stats.get("reliability_mean")))
                    print("rspa_inject_scale (step {}): {}".format(train_steps, rspa_stats.get("inject_scale")))
                    print("rspa_use_class_hint (step {}): {}".format(train_steps, rspa_stats.get("use_class_hint")))
                    print("rspa_class_hint_conf_mean (step {}): {}".format(train_steps, rspa_stats.get("class_hint_conf_mean")))
                    print("rspa_filter_low_conf (step {}): {}".format(train_steps, rspa_stats.get("filter_low_conf")))
                    print("rspa_filtered_prototypes (step {}): {}".format(train_steps, rspa_stats.get("filtered_prototypes")))
                    print("rspa_source_balance (step {}): {}".format(train_steps, rspa_stats.get("source_balance")))
                    print("rspa_source_mass_max_mean (step {}): {}".format(train_steps, rspa_stats.get("source_mass_max_mean")))
                    print("rspa_adaptive_gate (step {}): {}".format(train_steps, rspa_stats.get("adaptive_gate")))
                    print("rspa_centered_adaptive_gate (step {}): {}".format(train_steps, rspa_stats.get("centered_adaptive_gate")))
                    print("rspa_adaptive_gate_mean (step {}): {}".format(train_steps, rspa_stats.get("adaptive_gate_mean")))
                    print("rspa_feature_delta_norm_mean (step {}): {}".format(train_steps, rspa_stats.get("feature_delta_norm_mean")))
                    print("rspa_has_nan_or_inf (step {}): {}".format(train_steps, rspa_stats.get("has_nan_or_inf")))
                if getattr(args, "use_parallel_tcn", False) and epoch == 0 and j == 0 and train_steps < 3:
                    ptcn_stats = fineTuneModel.parallelTCNBranch.get_last_stats()
                    print("parallel_tcn_alpha (step {}): {}".format(train_steps, ptcn_stats.get("alpha")))
                    print("parallel_tcn_lstm_feat_norm_mean (step {}): {}".format(train_steps, ptcn_stats.get("lstm_feat_norm_mean")))
                    print("parallel_tcn_tcn_feat_norm_mean (step {}): {}".format(train_steps, ptcn_stats.get("tcn_feat_norm_mean")))
                    print("parallel_tcn_delta_norm_mean (step {}): {}".format(train_steps, ptcn_stats.get("delta_norm_mean")))
                    print("parallel_tcn_feature_delta_norm_mean (step {}): {}".format(train_steps, ptcn_stats.get("feature_delta_norm_mean")))
                    print("parallel_tcn_attn_entropy_norm_mean (step {}): {}".format(train_steps, ptcn_stats.get("attn_entropy_norm_mean")))
                    print("parallel_tcn_attn_max_mean (step {}): {}".format(train_steps, ptcn_stats.get("attn_max_mean")))
                    print("parallel_tcn_has_nan_or_inf (step {}): {}".format(train_steps, ptcn_stats.get("has_nan_or_inf")))
                if getattr(args, "use_attn_lstm_readout", False) and epoch == 0 and j == 0 and train_steps < 3:
                    attn_stats = fineTuneModel.attentiveSharedEncoder.get_last_stats()
                    print("attn_lstm_alpha (step {}): {}".format(train_steps, attn_stats.get("alpha")))
                    print("attn_lstm_last_norm_mean (step {}): {}".format(train_steps, attn_stats.get("last_norm_mean")))
                    print("attn_lstm_pooled_norm_mean (step {}): {}".format(train_steps, attn_stats.get("pooled_norm_mean")))
                    print("attn_lstm_feature_delta_norm_mean (step {}): {}".format(train_steps, attn_stats.get("feature_delta_norm_mean")))
                    print("attn_lstm_attn_entropy_norm_mean (step {}): {}".format(train_steps, attn_stats.get("attn_entropy_norm_mean")))
                    print("attn_lstm_attn_max_mean (step {}): {}".format(train_steps, attn_stats.get("attn_max_mean")))
                    print("attn_lstm_has_nan_or_inf (step {}): {}".format(train_steps, attn_stats.get("has_nan_or_inf")))
                if getattr(args, "use_mamba_lite", False) and epoch == 0 and j == 0 and train_steps < 3:
                    mamba_stats = fineTuneModel.mambaLiteBranch.get_last_stats()
                    print("mamba_lite_alpha (step {}): {}".format(train_steps, mamba_stats.get("alpha")))
                    print("mamba_lite_base_feat_norm_mean (step {}): {}".format(train_steps, mamba_stats.get("base_feat_norm_mean")))
                    print("mamba_lite_mamba_feat_norm_mean (step {}): {}".format(train_steps, mamba_stats.get("mamba_feat_norm_mean")))
                    print("mamba_lite_delta_norm_mean (step {}): {}".format(train_steps, mamba_stats.get("delta_norm_mean")))
                    print("mamba_lite_feature_delta_norm_mean (step {}): {}".format(train_steps, mamba_stats.get("feature_delta_norm_mean")))
                    print("mamba_lite_attn_entropy_norm_mean (step {}): {}".format(train_steps, mamba_stats.get("attn_entropy_norm_mean")))
                    print("mamba_lite_attn_max_mean (step {}): {}".format(train_steps, mamba_stats.get("attn_max_mean")))
                    print("mamba_lite_last_state_norm_mean (step {}): {}".format(train_steps, mamba_stats.get("last_state_norm_mean")))
                    print("mamba_lite_has_nan_or_inf (step {}): {}".format(train_steps, mamba_stats.get("has_nan_or_inf")))
                if getattr(args, "use_eeg_conformer", False) and epoch == 0 and j == 0 and train_steps < 3:
                    conf_stats = fineTuneModel.eegConformerBranch.get_last_stats()
                    print("eeg_conformer_alpha (step {}): {}".format(train_steps, conf_stats.get("alpha")))
                    print("eeg_conformer_inject_scale (step {}): {}".format(train_steps, conf_stats.get("inject_scale")))
                    print("eeg_conformer_base_feat_norm_mean (step {}): {}".format(train_steps, conf_stats.get("base_feat_norm_mean")))
                    print("eeg_conformer_feat_norm_mean (step {}): {}".format(train_steps, conf_stats.get("conformer_feat_norm_mean")))
                    print("eeg_conformer_delta_norm_mean (step {}): {}".format(train_steps, conf_stats.get("delta_norm_mean")))
                    print("eeg_conformer_feature_delta_norm_mean (step {}): {}".format(train_steps, conf_stats.get("feature_delta_norm_mean")))
                    print("eeg_conformer_node_attn_entropy_norm_mean (step {}): {}".format(train_steps, conf_stats.get("node_attn_entropy_norm_mean")))
                    print("eeg_conformer_node_attn_max_mean (step {}): {}".format(train_steps, conf_stats.get("node_attn_max_mean")))
                    print("eeg_conformer_time_attn_entropy_norm_mean (step {}): {}".format(train_steps, conf_stats.get("time_attn_entropy_norm_mean")))
                    print("eeg_conformer_time_attn_max_mean (step {}): {}".format(train_steps, conf_stats.get("time_attn_max_mean")))
                    print("eeg_conformer_has_nan_or_inf (step {}): {}".format(train_steps, conf_stats.get("has_nan_or_inf")))
                if getattr(args, "use_patch_transformer", False) and epoch == 0 and j == 0 and train_steps < 3:
                    patch_stats = fineTuneModel.patchTransformerBranch.get_last_stats()
                    print("patch_transformer_alpha (step {}): {}".format(train_steps, patch_stats.get("alpha")))
                    print("patch_transformer_base_feat_norm_mean (step {}): {}".format(train_steps, patch_stats.get("base_feat_norm_mean")))
                    print("patch_transformer_patch_feat_norm_mean (step {}): {}".format(train_steps, patch_stats.get("patch_feat_norm_mean")))
                    print("patch_transformer_delta_norm_mean (step {}): {}".format(train_steps, patch_stats.get("delta_norm_mean")))
                    print("patch_transformer_feature_delta_norm_mean (step {}): {}".format(train_steps, patch_stats.get("feature_delta_norm_mean")))
                    print("patch_transformer_patch_attn_entropy_norm_mean (step {}): {}".format(train_steps, patch_stats.get("patch_attn_entropy_norm_mean")))
                    print("patch_transformer_patch_attn_max_mean (step {}): {}".format(train_steps, patch_stats.get("patch_attn_max_mean")))
                    print("patch_transformer_num_patches (step {}): {}".format(train_steps, patch_stats.get("num_patches")))
                    print("patch_transformer_has_nan_or_inf (step {}): {}".format(train_steps, patch_stats.get("has_nan_or_inf")))
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
                train_steps += 1
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        step_count = max(train_steps, 1)
        cls_loss_avg = cls_loss_sum / step_count
        hyp_loss_avg = hyp_loss_sum / step_count
        total_loss_avg = total_loss_sum / step_count
        print("cls_loss_avg:", cls_loss_avg)
        print("hyp_contrastive_loss_avg:", hyp_loss_avg)
        print("total_loss_avg:", total_loss_avg)
        if rcc_steps > 0:
            rcc_loss_raw_avg = rcc_loss_raw_sum / rcc_steps
            rcc_loss_weighted_avg = rcc_loss_weighted_sum / rcc_steps
            rcc_lambda_now_avg = rcc_lambda_now_sum / rcc_steps
            rcc_reliability_mean_avg = rcc_reliability_mean_sum / rcc_steps
            rcc_center_norm_mean_avg = rcc_center_norm_mean_sum / rcc_steps
            print("loss_rcc_raw_avg:", rcc_loss_raw_avg)
            print("loss_rcc_weighted_avg:", rcc_loss_weighted_avg)
            print("rcc_lambda_now_avg:", rcc_lambda_now_avg)
            print("reliability_mean_avg:", rcc_reliability_mean_avg)
            print("reliability_min_epoch:", rcc_reliability_min_epoch)
            print("reliability_max_epoch:", rcc_reliability_max_epoch)
            print("center_norm_mean_avg:", rcc_center_norm_mean_avg)
            print("valid_rcc_samples_sum:", rcc_valid_samples_sum)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/rcc loss raw avg', rcc_loss_raw_avg, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/rcc loss weighted avg', rcc_loss_weighted_avg, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/rcc lambda avg', rcc_lambda_now_avg, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/rcc reliability mean', rcc_reliability_mean_avg, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/rcc reliability min', rcc_reliability_min_epoch, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/rcc reliability max', rcc_reliability_max_epoch, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/rcc center norm mean', rcc_center_norm_mean_avg, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/rcc valid samples', rcc_valid_samples_sum, epoch + 1)
        if use_srw and srw_weight_count > 0:
            srw_weight_avg = srw_weight_sum / srw_weight_count
            print("srw_weight_avg:", srw_weight_avg)
            print("srw_weight_min:", srw_weight_min)
            print("srw_weight_max:", srw_weight_max)
            print("srw_source_loss_ema:", ",".join(["{:.6f}".format(v) for v in source_loss_ema]))
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/srw weight avg', srw_weight_avg, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/srw weight min', srw_weight_min, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/srw weight max', srw_weight_max, epoch + 1)
        if rspa_consistency_count > 0:
            rspa_consistency_avg = rspa_consistency_sum / rspa_consistency_count
            print("rspa_consistency_loss_avg:", rspa_consistency_avg)
            print("rspa_consistency_weight:", float(getattr(args, "rspa_consistency_weight", 0.02)))
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/rspa consistency loss avg', rspa_consistency_avg, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/cls loss avg', cls_loss_avg, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/hyp contrastive loss avg', hyp_loss_avg, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/total loss avg', total_loss_avg, epoch + 1)
        if hyp_count > 0:
            z_hyp_norm_mean_avg = z_hyp_norm_mean_sum / hyp_count
            print("z_hyp_norm_mean_avg:", z_hyp_norm_mean_avg)
            print("z_hyp_norm_max:", z_hyp_norm_max)
            print("hyp_non_finite_count:", hyp_non_finite_count)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/z hyp norm mean', z_hyp_norm_mean_avg, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/z hyp norm max', z_hyp_norm_max, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/hyp non finite count', hyp_non_finite_count, epoch + 1)
        if ortho_loss_count > 0:
            ortho_loss_avg = ortho_loss_sum / ortho_loss_count
            prompt_sim_offdiag_avg = prompt_sim_offdiag_sum / prompt_sim_count
            print("prompt_ortho_loss_avg:", ortho_loss_avg)
            print("prompt_sim_offdiag_avg:", prompt_sim_offdiag_avg)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/prompt ortho loss', ortho_loss_avg, epoch + 1)
            writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/prompt sim offdiag', prompt_sim_offdiag_avg, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))
        # test the fine-tuned model with the data of unseen target subject
        eval_fineTuneModel = fineTuneModel
        if use_finetune_ema and epoch >= finetune_ema_start_epoch:
            eval_fineTuneModel = _build_model_with_state(fineTuneModel, finetune_ema_state)
            print("finetune_ema_eval:", True)
        testModel = DMMRTestModel(eval_fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        last_test_acc = float(acc_DMMR)
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_epoch = epoch
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    # save models
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    print("fold_id:", one_subject + 1)
    print("best_acc:", acc_final)
    print("final_acc:", last_test_acc)
    print("best_epoch:", best_epoch)
    if getattr(args, "use_sspb_v2", False):
        alpha_final = fineTuneModel.sourcePromptBank.get_alpha_value()
        beta_final = fineTuneModel.sourcePromptBank.get_beta_value()
        prompt_gate_final = fineTuneModel.sourcePromptBank.get_prompt_gate_value()
        print("prompt_alpha_final:", alpha_final)
        print("prompt_beta_final:", beta_final)
        print("prompt_gate_final:", prompt_gate_final)
        writer.add_text("subject {} prompt alpha final".format(one_subject + 1), str(alpha_final))
        writer.add_text("subject {} prompt beta final".format(one_subject + 1), str(beta_final))
        writer.add_text("subject {} prompt gate final".format(one_subject + 1), str(prompt_gate_final))
    if getattr(args, "use_gcn_residual", False):
        gcn_alpha_final = fineTuneModel.temporalGraphBlock.get_gcn_alpha_value()
        print("gcn_alpha_final:", gcn_alpha_final)
        writer.add_text("subject {} gcn alpha final".format(one_subject + 1), str(gcn_alpha_final))
    if getattr(args, "use_tgb", False):
        adj_stats_final = fineTuneModel.temporalGraphBlock.get_adj_norm_stats()
        print("adj_norm_diag_mean_final:", adj_stats_final["diag_mean"])
        print("adj_norm_offdiag_mean_final:", adj_stats_final["offdiag_mean"])
        writer.add_text("subject {} adj norm diag mean final".format(one_subject + 1), str(adj_stats_final["diag_mean"]))
        writer.add_text("subject {} adj norm offdiag mean final".format(one_subject + 1), str(adj_stats_final["offdiag_mean"]))
    if getattr(args, "use_mst", False):
        mst_alpha_final = fineTuneModel.multiScaleTemporalBlock.get_alpha_value()
        print("mst_alpha_final:", mst_alpha_final)
        writer.add_text("subject {} mst alpha final".format(one_subject + 1), str(mst_alpha_final))
    if getattr(args, "use_parallel_tcn", False):
        ptcn_stats_final = fineTuneModel.parallelTCNBranch.get_last_stats()
        ptcn_alpha_final = fineTuneModel.parallelTCNBranch.get_alpha_value()
        print("parallel_tcn_alpha_final:", ptcn_alpha_final)
        print("parallel_tcn_lstm_feat_norm_mean_final:", ptcn_stats_final.get("lstm_feat_norm_mean"))
        print("parallel_tcn_tcn_feat_norm_mean_final:", ptcn_stats_final.get("tcn_feat_norm_mean"))
        print("parallel_tcn_delta_norm_mean_final:", ptcn_stats_final.get("delta_norm_mean"))
        print("parallel_tcn_feature_delta_norm_mean_final:", ptcn_stats_final.get("feature_delta_norm_mean"))
        print("parallel_tcn_attn_entropy_norm_mean_final:", ptcn_stats_final.get("attn_entropy_norm_mean"))
        print("parallel_tcn_attn_max_mean_final:", ptcn_stats_final.get("attn_max_mean"))
        print("parallel_tcn_has_nan_or_inf_final:", ptcn_stats_final.get("has_nan_or_inf"))
        writer.add_text("subject {} parallel tcn alpha final".format(one_subject + 1), str(ptcn_alpha_final))
        writer.add_text("subject {} parallel tcn feature delta norm final".format(one_subject + 1), str(ptcn_stats_final.get("feature_delta_norm_mean")))
    if getattr(args, "use_attn_lstm_readout", False):
        attn_stats_final = fineTuneModel.attentiveSharedEncoder.get_last_stats()
        attn_alpha_final = fineTuneModel.attentiveSharedEncoder.get_alpha_value()
        print("attn_lstm_alpha_final:", attn_alpha_final)
        print("attn_lstm_last_norm_mean_final:", attn_stats_final.get("last_norm_mean"))
        print("attn_lstm_pooled_norm_mean_final:", attn_stats_final.get("pooled_norm_mean"))
        print("attn_lstm_feature_delta_norm_mean_final:", attn_stats_final.get("feature_delta_norm_mean"))
        print("attn_lstm_attn_entropy_norm_mean_final:", attn_stats_final.get("attn_entropy_norm_mean"))
        print("attn_lstm_attn_max_mean_final:", attn_stats_final.get("attn_max_mean"))
        print("attn_lstm_has_nan_or_inf_final:", attn_stats_final.get("has_nan_or_inf"))
        writer.add_text("subject {} attn lstm alpha final".format(one_subject + 1), str(attn_alpha_final))
        writer.add_text("subject {} attn lstm feature delta norm final".format(one_subject + 1), str(attn_stats_final.get("feature_delta_norm_mean")))
    if getattr(args, "use_mamba_lite", False):
        mamba_stats_final = fineTuneModel.mambaLiteBranch.get_last_stats()
        mamba_alpha_final = fineTuneModel.mambaLiteBranch.get_alpha_value()
        print("mamba_lite_alpha_final:", mamba_alpha_final)
        print("mamba_lite_base_feat_norm_mean_final:", mamba_stats_final.get("base_feat_norm_mean"))
        print("mamba_lite_mamba_feat_norm_mean_final:", mamba_stats_final.get("mamba_feat_norm_mean"))
        print("mamba_lite_delta_norm_mean_final:", mamba_stats_final.get("delta_norm_mean"))
        print("mamba_lite_feature_delta_norm_mean_final:", mamba_stats_final.get("feature_delta_norm_mean"))
        print("mamba_lite_attn_entropy_norm_mean_final:", mamba_stats_final.get("attn_entropy_norm_mean"))
        print("mamba_lite_attn_max_mean_final:", mamba_stats_final.get("attn_max_mean"))
        print("mamba_lite_last_state_norm_mean_final:", mamba_stats_final.get("last_state_norm_mean"))
        print("mamba_lite_has_nan_or_inf_final:", mamba_stats_final.get("has_nan_or_inf"))
        writer.add_text("subject {} mamba lite alpha final".format(one_subject + 1), str(mamba_alpha_final))
        writer.add_text("subject {} mamba lite feature delta norm final".format(one_subject + 1), str(mamba_stats_final.get("feature_delta_norm_mean")))
    if getattr(args, "use_eeg_conformer", False):
        conf_stats_final = fineTuneModel.eegConformerBranch.get_last_stats()
        conf_alpha_final = fineTuneModel.eegConformerBranch.get_alpha_value()
        print("eeg_conformer_alpha_final:", conf_alpha_final)
        print("eeg_conformer_inject_scale_final:", conf_stats_final.get("inject_scale"))
        print("eeg_conformer_base_feat_norm_mean_final:", conf_stats_final.get("base_feat_norm_mean"))
        print("eeg_conformer_feat_norm_mean_final:", conf_stats_final.get("conformer_feat_norm_mean"))
        print("eeg_conformer_delta_norm_mean_final:", conf_stats_final.get("delta_norm_mean"))
        print("eeg_conformer_feature_delta_norm_mean_final:", conf_stats_final.get("feature_delta_norm_mean"))
        print("eeg_conformer_node_attn_entropy_norm_mean_final:", conf_stats_final.get("node_attn_entropy_norm_mean"))
        print("eeg_conformer_node_attn_max_mean_final:", conf_stats_final.get("node_attn_max_mean"))
        print("eeg_conformer_time_attn_entropy_norm_mean_final:", conf_stats_final.get("time_attn_entropy_norm_mean"))
        print("eeg_conformer_time_attn_max_mean_final:", conf_stats_final.get("time_attn_max_mean"))
        print("eeg_conformer_has_nan_or_inf_final:", conf_stats_final.get("has_nan_or_inf"))
        writer.add_text("subject {} eeg conformer alpha final".format(one_subject + 1), str(conf_alpha_final))
        writer.add_text("subject {} eeg conformer feature delta norm final".format(one_subject + 1), str(conf_stats_final.get("feature_delta_norm_mean")))
    if getattr(args, "use_patch_transformer", False):
        patch_stats_final = fineTuneModel.patchTransformerBranch.get_last_stats()
        patch_alpha_final = fineTuneModel.patchTransformerBranch.get_alpha_value()
        print("patch_transformer_alpha_final:", patch_alpha_final)
        print("patch_transformer_base_feat_norm_mean_final:", patch_stats_final.get("base_feat_norm_mean"))
        print("patch_transformer_patch_feat_norm_mean_final:", patch_stats_final.get("patch_feat_norm_mean"))
        print("patch_transformer_delta_norm_mean_final:", patch_stats_final.get("delta_norm_mean"))
        print("patch_transformer_feature_delta_norm_mean_final:", patch_stats_final.get("feature_delta_norm_mean"))
        print("patch_transformer_patch_attn_entropy_norm_mean_final:", patch_stats_final.get("patch_attn_entropy_norm_mean"))
        print("patch_transformer_patch_attn_max_mean_final:", patch_stats_final.get("patch_attn_max_mean"))
        print("patch_transformer_num_patches_final:", patch_stats_final.get("num_patches"))
        print("patch_transformer_has_nan_or_inf_final:", patch_stats_final.get("has_nan_or_inf"))
        writer.add_text("subject {} patch transformer alpha final".format(one_subject + 1), str(patch_alpha_final))
        writer.add_text("subject {} patch transformer feature delta norm final".format(one_subject + 1), str(patch_stats_final.get("feature_delta_norm_mean")))
    if getattr(args, "use_msr", False):
        msr_stats_final = fineTuneModel.multiSourceSubjectRouter.get_last_stats()
        msr_alpha_final = fineTuneModel.multiSourceSubjectRouter.get_alpha_value()
        print("msr_alpha_final:", msr_alpha_final)
        print("msr_router_entropy_mean_final:", msr_stats_final.get("router_entropy_mean"))
        print("msr_router_entropy_norm_mean_final:", msr_stats_final.get("router_entropy_norm_mean"))
        print("msr_router_max_mean_final:", msr_stats_final.get("router_max_mean"))
        print("msr_context_norm_mean_final:", msr_stats_final.get("context_norm_mean"))
        print("msr_delta_norm_mean_final:", msr_stats_final.get("delta_norm_mean"))
        print("msr_feature_delta_norm_mean_final:", msr_stats_final.get("feature_delta_norm_mean"))
        print("msr_memory_norm_mean_final:", msr_stats_final.get("memory_norm_mean"))
        print("msr_top1_counts_final:", msr_stats_final.get("top1_counts"))
        print("msr_has_nan_or_inf_final:", msr_stats_final.get("has_nan_or_inf"))
        writer.add_text("subject {} msr alpha final".format(one_subject + 1), str(msr_alpha_final))
        writer.add_text("subject {} msr router entropy final".format(one_subject + 1), str(msr_stats_final.get("router_entropy_mean")))
        writer.add_text("subject {} msr router max final".format(one_subject + 1), str(msr_stats_final.get("router_max_mean")))
        writer.add_text("subject {} msr feature delta norm final".format(one_subject + 1), str(msr_stats_final.get("feature_delta_norm_mean")))
    if getattr(args, "use_class_proto_calib", False):
        proto_stats_final = fineTuneModel.classPrototypeCalibrator.get_last_stats()
        proto_alpha_final = fineTuneModel.classPrototypeCalibrator.get_alpha_value()
        print("class_proto_alpha_final:", proto_alpha_final)
        print("class_proto_temperature_final:", proto_stats_final.get("temperature"))
        print("class_proto_logit_mean_final:", proto_stats_final.get("proto_logit_mean"))
        print("class_proto_logit_std_final:", proto_stats_final.get("proto_logit_std"))
        print("class_proto_entropy_norm_mean_final:", proto_stats_final.get("proto_entropy_norm_mean"))
        print("class_proto_norm_mean_final:", proto_stats_final.get("prototype_norm_mean"))
        print("class_proto_counts_final:", proto_stats_final.get("prototype_counts"))
        print("class_proto_has_nan_or_inf_final:", proto_stats_final.get("has_nan_or_inf"))
        writer.add_text("subject {} class proto alpha final".format(one_subject + 1), str(proto_alpha_final))
        writer.add_text("subject {} class proto entropy norm final".format(one_subject + 1), str(proto_stats_final.get("proto_entropy_norm_mean")))
        writer.add_text("subject {} class proto logit std final".format(one_subject + 1), str(proto_stats_final.get("proto_logit_std")))
    if getattr(args, "use_feature_calib", False):
        calib_stats_final = fineTuneModel.featureDistributionCalibrator.get_last_stats()
        calib_alpha_final = fineTuneModel.featureDistributionCalibrator.get_alpha_value()
        print("feature_calib_alpha_final:", calib_alpha_final)
        print("feature_calib_source_mean_norm_final:", calib_stats_final.get("source_mean_norm"))
        print("feature_calib_source_std_mean_final:", calib_stats_final.get("source_std_mean"))
        print("feature_calib_source_std_min_final:", calib_stats_final.get("source_std_min"))
        print("feature_calib_use_std_final:", calib_stats_final.get("use_std"))
        print("feature_calib_feature_norm_mean_final:", calib_stats_final.get("feature_norm_mean"))
        print("feature_calib_calibrated_norm_mean_final:", calib_stats_final.get("calibrated_norm_mean"))
        print("feature_calib_delta_norm_mean_final:", calib_stats_final.get("feature_delta_norm_mean"))
        print("feature_calib_has_nan_or_inf_final:", calib_stats_final.get("has_nan_or_inf"))
        writer.add_text("subject {} feature calib alpha final".format(one_subject + 1), str(calib_alpha_final))
        writer.add_text("subject {} feature calib delta norm final".format(one_subject + 1), str(calib_stats_final.get("feature_delta_norm_mean")))
    if getattr(args, "use_rspa", False):
        rspa_stats_final = fineTuneModel.reliabilitySourcePrototypeAttention.get_last_stats()
        rspa_alpha_final = fineTuneModel.reliabilitySourcePrototypeAttention.get_alpha_value()
        print("rspa_alpha_final:", rspa_alpha_final)
        print("rspa_valid_prototypes_final:", rspa_stats_final.get("valid_prototypes"))
        print("rspa_attn_entropy_norm_mean_final:", rspa_stats_final.get("attn_entropy_norm_mean"))
        print("rspa_attn_max_mean_final:", rspa_stats_final.get("attn_max_mean"))
        print("rspa_reliability_mean_final:", rspa_stats_final.get("reliability_mean"))
        print("rspa_reliability_min_final:", rspa_stats_final.get("reliability_min"))
        print("rspa_reliability_max_final:", rspa_stats_final.get("reliability_max"))
        print("rspa_inject_scale_final:", rspa_stats_final.get("inject_scale"))
        print("rspa_use_class_hint_final:", rspa_stats_final.get("use_class_hint"))
        print("rspa_class_hint_weight_final:", rspa_stats_final.get("class_hint_weight"))
        print("rspa_class_hint_conf_mean_final:", rspa_stats_final.get("class_hint_conf_mean"))
        print("rspa_filter_low_conf_final:", rspa_stats_final.get("filter_low_conf"))
        print("rspa_min_reliability_final:", rspa_stats_final.get("min_reliability"))
        print("rspa_filtered_prototypes_final:", rspa_stats_final.get("filtered_prototypes"))
        print("rspa_source_balance_final:", rspa_stats_final.get("source_balance"))
        print("rspa_source_cap_final:", rspa_stats_final.get("source_cap"))
        print("rspa_source_mass_max_mean_final:", rspa_stats_final.get("source_mass_max_mean"))
        print("rspa_adaptive_gate_final:", rspa_stats_final.get("adaptive_gate"))
        print("rspa_centered_adaptive_gate_final:", rspa_stats_final.get("centered_adaptive_gate"))
        print("rspa_centered_gate_delta_final:", rspa_stats_final.get("centered_gate_delta"))
        print("rspa_gate_output_init_std_final:", rspa_stats_final.get("gate_output_init_std"))
        print("rspa_adaptive_gate_mean_final:", rspa_stats_final.get("adaptive_gate_mean"))
        print("rspa_adaptive_gate_min_value_final:", rspa_stats_final.get("adaptive_gate_min_value"))
        print("rspa_adaptive_gate_max_value_final:", rspa_stats_final.get("adaptive_gate_max_value"))
        print("rspa_context_norm_mean_final:", rspa_stats_final.get("context_norm_mean"))
        print("rspa_delta_norm_mean_final:", rspa_stats_final.get("delta_norm_mean"))
        print("rspa_feature_delta_norm_mean_final:", rspa_stats_final.get("feature_delta_norm_mean"))
        print("rspa_has_nan_or_inf_final:", rspa_stats_final.get("has_nan_or_inf"))
        writer.add_text("subject {} rspa alpha final".format(one_subject + 1), str(rspa_alpha_final))
        writer.add_text("subject {} rspa feature delta norm final".format(one_subject + 1), str(rspa_stats_final.get("feature_delta_norm_mean")))
    if getattr(args, "use_csgformer", False):
        gamma_final = fineTuneModel.csgformerBlock.get_gamma_value()
        dyn_stats_final = fineTuneModel.csgformerBlock.get_last_dyn_stats()
        pool_stats_final = fineTuneModel.csgformerBlock.get_last_pool_stats()
        print("csg_gamma_final:", gamma_final)
        print("csg_a_dyn_final_min:", dyn_stats_final["min"])
        print("csg_a_dyn_final_max:", dyn_stats_final["max"])
        print("csg_a_dyn_final_has_nan_or_inf:", dyn_stats_final["has_nan_or_inf"])
        print("csg_alpha_max_mean_final:", pool_stats_final["alpha_max_mean"])
        print("csg_alpha_entropy_mean_final:", pool_stats_final["alpha_entropy_mean"])
        writer.add_text("subject {} csg gamma final".format(one_subject + 1), str(gamma_final))
        writer.add_text("subject {} csg a dyn final min".format(one_subject + 1), str(dyn_stats_final["min"]))
        writer.add_text("subject {} csg a dyn final max".format(one_subject + 1), str(dyn_stats_final["max"]))
        writer.add_text("subject {} csg a dyn final has nan".format(one_subject + 1), str(dyn_stats_final["has_nan_or_inf"]))
        writer.add_text("subject {} csg alpha max mean final".format(one_subject + 1), str(pool_stats_final["alpha_max_mean"]))
        writer.add_text("subject {} csg alpha entropy mean final".format(one_subject + 1), str(pool_stats_final["alpha_entropy_mean"]))
    if getattr(args, "use_emt_lite_v2", False):
        gamma_graph_final = fineTuneModel.emtLiteV2Encoder.get_gamma_graph_value()
        emt_stats_final = fineTuneModel.emtLiteV2Encoder.get_last_stats()
        print("emt_gamma_graph_final:", gamma_graph_final)
        print("emt_gamma_range_final:", (emt_stats_final["gamma_min"], emt_stats_final["gamma_max"]))
        print("emt_beta_range_final:", (emt_stats_final["beta_min"], emt_stats_final["beta_max"]))
        print("emt_a_dyn_final_min:", emt_stats_final["a_dyn_min"])
        print("emt_a_dyn_final_max:", emt_stats_final["a_dyn_max"])
        print("emt_a_dyn_final_has_nan_or_inf:", emt_stats_final["a_dyn_has_nan_or_inf"])
        print("emt_node_alpha_max_mean_final:", emt_stats_final["node_alpha_max_mean"])
        print("emt_node_alpha_entropy_mean_final:", emt_stats_final["node_alpha_entropy_mean"])
        print("emt_temp_alpha_max_mean_final:", emt_stats_final["temp_alpha_max_mean"])
        print("emt_temp_alpha_entropy_mean_final:", emt_stats_final["temp_alpha_entropy_mean"])
        writer.add_text("subject {} emt gamma graph final".format(one_subject + 1), str(gamma_graph_final))
        writer.add_text("subject {} emt gamma min final".format(one_subject + 1), str(emt_stats_final["gamma_min"]))
        writer.add_text("subject {} emt gamma max final".format(one_subject + 1), str(emt_stats_final["gamma_max"]))
        writer.add_text("subject {} emt beta min final".format(one_subject + 1), str(emt_stats_final["beta_min"]))
        writer.add_text("subject {} emt beta max final".format(one_subject + 1), str(emt_stats_final["beta_max"]))
        writer.add_text("subject {} emt a dyn min final".format(one_subject + 1), str(emt_stats_final["a_dyn_min"]))
        writer.add_text("subject {} emt a dyn max final".format(one_subject + 1), str(emt_stats_final["a_dyn_max"]))
        writer.add_text("subject {} emt a dyn has nan final".format(one_subject + 1), str(emt_stats_final["a_dyn_has_nan_or_inf"]))
        writer.add_text("subject {} emt node alpha max mean final".format(one_subject + 1), str(emt_stats_final["node_alpha_max_mean"]))
        writer.add_text("subject {} emt node alpha entropy mean final".format(one_subject + 1), str(emt_stats_final["node_alpha_entropy_mean"]))
        writer.add_text("subject {} emt temp alpha max mean final".format(one_subject + 1), str(emt_stats_final["temp_alpha_max_mean"]))
        writer.add_text("subject {} emt temp alpha entropy mean final".format(one_subject + 1), str(emt_stats_final["temp_alpha_entropy_mean"]))
    if getattr(args, "use_dmmr_hemi", False):
        hemi_stats_final = fineTuneModel.hemiFusion.get_last_stats()
        hemi_alpha_final = fineTuneModel.hemiFusion.get_alpha_value()
        print("hemi_alpha_final:", hemi_alpha_final)
        print("hemi_full_norm_mean_final:", hemi_stats_final["full_norm_mean"])
        print("hemi_left_norm_mean_final:", hemi_stats_final["left_norm_mean"])
        print("hemi_right_norm_mean_final:", hemi_stats_final["right_norm_mean"])
        print("hemi_side_norm_mean_final:", hemi_stats_final["side_norm_mean"])
        print("hemi_fused_norm_mean_final:", hemi_stats_final["fused_norm_mean"])
        print("hemi_left_right_cos_mean_final:", hemi_stats_final["left_right_cos_mean"])
        writer.add_text("subject {} hemi alpha final".format(one_subject + 1), str(hemi_alpha_final))
        writer.add_text("subject {} hemi full norm mean final".format(one_subject + 1), str(hemi_stats_final["full_norm_mean"]))
        writer.add_text("subject {} hemi left norm mean final".format(one_subject + 1), str(hemi_stats_final["left_norm_mean"]))
        writer.add_text("subject {} hemi right norm mean final".format(one_subject + 1), str(hemi_stats_final["right_norm_mean"]))
        writer.add_text("subject {} hemi side norm mean final".format(one_subject + 1), str(hemi_stats_final["side_norm_mean"]))
        writer.add_text("subject {} hemi fused norm mean final".format(one_subject + 1), str(hemi_stats_final["fused_norm_mean"]))
        writer.add_text("subject {} hemi left right cos mean final".format(one_subject + 1), str(hemi_stats_final["left_right_cos_mean"]))
    return acc_final

############## Ablation studies ##############
# w/o mix
def trainDMMR_WithoutMix(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithoutMix(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final
# w/o noise
def trainDMMR_WithoutNoise(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithoutNoise(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/"+args.way+"/"+args.index+"/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir+str(one_subject)+'_pretrain_model.pth')
    torch.save(best_tune_model, modelDir+str(one_subject)+'_tune_model.pth')
    torch.save(best_test_model, modelDir+str(one_subject)+'_test_model.pth')
    return acc_final
# w/o both
def trainDMMR_WithoutBothMixAndNoise(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithoutBothMixAndNoise(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                                number_of_category=args.cls_classes, batch_size=args.batch_size,
                                                time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

############## Other noise injection methods ##############
def trainDMMR_Noise_MaskChannels(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithMaskChannels(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, args, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

def trainDMMR_Noise_MaskTimeSteps(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithMaskTimeSteps(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, args, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

def trainDMMR_Noise_ChannelsShuffling(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithChannelsShuffling(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, args, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

def trainDMMR_Noise_Dropout(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithDropout(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps, dropout_rate=0.2)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, args, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

############## T-SNE plots ##############
class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, features, labels):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.labels = labels

    def plot_tsne(self, save_filename, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        if TSNE is None or plt is None:
            raise ImportError("T-SNE plotting requires scikit-learn and matplotlib.")
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        for i in range(data.shape[0]):
            colors = plt.cm.tab20.colors
            plt.scatter(data[i, 0], data[i, 1], color=colors[self.labels[i]])
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('tsne.eps', dpi=600, format='eps')
        plt.savefig(save_filename, dpi=600)
        plt.show()
def TSNEForDMMR(data_loader_dict, cuda, args):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    target_loader = data_loader_dict["test_loader"]
    preTrainModel = DMMRPreTrainingModel(cuda,
                                         number_of_source=len(source_loader),
                                         number_of_category=args.cls_classes,
                                         batch_size=args.batch_size,
                                         time_steps=args.time_steps)
    #load the pretrained model
    preTrainModel.load_state_dict(torch.load("T-SNE/model/1_pretrain_model.pth", map_location='cpu'))
    preTrainModel.eval()
    pretrainReturnFeature = ModelReturnFeatures(preTrainModel, time_steps=args.time_steps)
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes,
                                    batch_size=args.batch_size,
                                    time_steps=args.time_steps)
    # load the fine-tuned model
    fineTuneModel.load_state_dict(torch.load("T-SNE/model/1_tune_model.pth", map_location='cpu'))
    fineTuneModel.eval()
    fineTuneModelReturnFeauters = ModelReturnFeatures(fineTuneModel, time_steps=args.time_steps)
    fineTuneModelReturnFeauters.eval()

    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))

    origin_features_list = []
    origin_subject_id_list = []
    label_list = []
    pretrain_shared_features_list = []
    shared_features_list = []
    for i in range(1, 2):
        for j in range(len(source_iters)):
            try:
                source_data, source_label = next(source_iters[j])
            except:
                source_iters[j] = iter(source_loader[j])
                source_data, source_label = next(source_iters[j])
            subject_id = torch.ones(args.batch_size)
            subject_id = subject_id * j
            subject_id = subject_id.long()

            _, pretrain_shared_feature = pretrainReturnFeature(source_data)
            _, shared_feature = fineTuneModelReturnFeauters(source_data)

            num_samples = 50
            # 50 samples are taken from each individual subject data
            source_data_narray = source_data.numpy()
            label_data_narray = source_label.squeeze().numpy()
            # Reshape for sampling
            source_data_narray = source_data_narray.reshape(512, 30 * 310)
            # Randomly select 50 samples from it to obtain a tensor of size (50, 310).
            random_indices = np.random.choice(source_data_narray.shape[0], num_samples, replace=False)
            source_data_narray_50 = source_data_narray[random_indices]
            subject_narray = np.full((num_samples,), j)
            label_data_narray_50 = label_data_narray[random_indices]
            #origin feature
            origin_features_list.append(source_data_narray_50)
            origin_subject_id_list.append(subject_narray)
            label_list.append(label_data_narray_50)

            # pretrained feature
            pretrain_shared_feature_narray = pretrain_shared_feature.detach().numpy()
            pretrain_shared_feature_narray_50 = pretrain_shared_feature_narray[random_indices]
            pretrain_shared_features_list.append(pretrain_shared_feature_narray_50)
            #fine-tuned feature
            shared_feature_narray = shared_feature.detach().numpy()
            shared_feature_narray_50 = shared_feature_narray[random_indices]
            shared_features_list.append(shared_feature_narray_50)

        #generate target data
        target_data, target_label = next(iter(target_loader))
        _, target_pretrain_shared_feature = pretrainReturnFeature(target_data)
        _, target_shared_feature = fineTuneModelReturnFeauters(target_data)
        target_data_narray = target_data.numpy()
        target_label = target_label.squeeze().numpy()
        target_data_narray = target_data_narray.reshape(512, 30 * 310)
        random_indices_target = np.random.choice(target_data_narray.shape[0], num_samples, replace=False)
        target_data_narray_50 = target_data_narray[random_indices_target]
        target_subject_id = np.full((num_samples,), 14)
        target_label_narray_50 = target_label[random_indices]


        #add target subject data
        origin_features_list.append(target_data_narray_50)
        origin_subject_id_list.append(target_subject_id)
        label_list.append(target_label_narray_50)

        target_pretrain_shared_feature_narray = target_pretrain_shared_feature.detach().numpy()
        target_pretrain_shared_feature_narray_50 = target_pretrain_shared_feature_narray[random_indices]
        pretrain_shared_features_list.append(target_pretrain_shared_feature_narray_50)

        target_shared_feature_narray = target_shared_feature.detach().numpy()
        target_shared_feature_narray_50 = target_shared_feature_narray[random_indices]
        shared_features_list.append(target_shared_feature_narray_50)


        #concat for later norm
        origin_stacked_feature = np.concatenate(origin_features_list, axis=0)
        stacked_subject_id = np.concatenate(origin_subject_id_list, axis=0)
        stacked_label = np.concatenate(label_list, axis=0)

        # T-SNE
        #origin data
        vis_pretrain_shared = FeatureVisualize(origin_stacked_feature, stacked_subject_id)
        vis_pretrain_shared.plot_tsne('T-SNE/plot/origin_subject.jpg',save_eps=False)
        vis_pretrain_shared = FeatureVisualize(origin_stacked_feature, stacked_label)
        vis_pretrain_shared.plot_tsne("T-SNE/plot/origin_label.jpg",save_eps=False)

        # pretrained feature
        pretrain_shared_stacked_feature = np.concatenate(pretrain_shared_features_list, axis=0)
        vis_pretrain_shared = FeatureVisualize(pretrain_shared_stacked_feature, stacked_subject_id)
        vis_pretrain_shared.plot_tsne('T-SNE/plot/pretrain_subject.jpg',save_eps=False)
        vis_pretrain_shared = FeatureVisualize(pretrain_shared_stacked_feature, stacked_label)
        vis_pretrain_shared.plot_tsne("T-SNE/plot/pretrain_label.jpg",save_eps=False)
        # fine tuned data
        shared_stacked_feature = np.concatenate(shared_features_list, axis=0)
        vis_shared = FeatureVisualize(shared_stacked_feature, stacked_subject_id)
        vis_shared.plot_tsne("T-SNE/plot/tune_subject.jpg",save_eps=False)
        vis_shared_label = FeatureVisualize(shared_stacked_feature, stacked_label)
        vis_shared_label.plot_tsne("T-SNE/plot/tune_label.jpg",save_eps=False)
        return 0
