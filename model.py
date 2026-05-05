import torch
import torch.nn as nn
import torch.nn.functional as F
from GradientReverseLayer import ReverseLayerF
from temporal_graph_block import TemporalGraphBlock
from multiscale_temporal_block import MultiScaleSpatiotemporalBlock
from csgformer_block import CSGFormerBlock
from emt_lite_v2_encoder import EmTLiteV2Encoder
from source_subject_prompt_bank import SourceSubjectPromptBank
from hyperbolic_contrast_head import HyperbolicContrastiveHead
from hemi_fusion import HemiAsymmetryFusion
from multi_source_subject_router import MultiSourceSubjectRouter
from class_prototype_calibrator import ClassPrototypeCalibrator
from feature_distribution_calibrator import FeatureDistributionCalibrator
from parallel_tcn_branch import ParallelTCNBranch
from mamba_lite_branch import MambaLiteFusionBranch
from eeg_conformer_branch import EEGConformerFusionBranch
from patch_transformer_branch import PatchTransformerFusionBranch
from reliability_source_prototype_attention import ReliabilitySourcePrototypeAttention
import random
import copy

# The ABP module
class Attention(nn.Module):
    def __init__(self, cuda, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        if cuda:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim).cuda())
            self.u_linear = nn.Parameter(torch.randn(input_dim).cuda())
        else:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim))
            self.u_linear = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, batch_size, time_steps):
        x_reshape = torch.Tensor.reshape(x, [-1, self.input_dim])
        attn_softmax = F.softmax(torch.mm(x_reshape, self.w_linear)+ self.u_linear,1)
        res = torch.mul(attn_softmax, x_reshape)
        res = torch.Tensor.reshape(res, [batch_size, time_steps, self.input_dim])
        return res

class LSTM(nn.Module):
    def __init__(self, input_dim=310, output_dim=64, layers=2, location=-1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=layers, batch_first=True)
        self.location = location
    def forward(self, x):
        # self.lstm.flatten_parameters()
        feature, (hn, cn) = self.lstm(x)
        return feature[:, self.location, :], hn, cn

class Encoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2):
        super(Encoder, self).__init__()
        self.theta = LSTM(input_dim, hid_dim, n_layers)
    def forward(self, x):
        x_h = self.theta(x)
        return x_h

class AttentiveLSTMEncoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=1, alpha_init=0.3, alpha_max=1.0, dropout=0.1):
        super(AttentiveLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hid_dim, num_layers=n_layers, batch_first=True)
        self.attn_score = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 1),
        )
        self.alpha_max = float(alpha_max)
        self.alpha_raw = nn.Parameter(self._bounded_raw(float(alpha_init), self.alpha_max))
        self.last_stats = {}

    @staticmethod
    def _bounded_raw(target_value, max_value):
        eps = 1e-6
        ratio = min(max(target_value / max(max_value, eps), eps), 1.0 - eps)
        return torch.log(torch.tensor(ratio / (1.0 - ratio)))

    def get_alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    def get_alpha_value(self):
        return float(self.get_alpha().detach().cpu().item())

    def get_last_stats(self):
        return dict(self.last_stats)

    def forward(self, x):
        outputs, (hn, cn) = self.lstm(x)
        last_feat = outputs[:, -1, :]
        scores = self.attn_score(outputs).squeeze(-1)
        attn = F.softmax(scores, dim=-1)
        pooled_feat = torch.sum(outputs * attn.unsqueeze(-1), dim=1)
        alpha = self.get_alpha()
        feat = last_feat + alpha * (pooled_feat - last_feat)
        with torch.no_grad():
            safe_attn = attn.clamp_min(1e-12)
            entropy = -(safe_attn * safe_attn.log()).sum(dim=-1)
            self.last_stats = {
                "alpha": float(alpha.detach().cpu().item()),
                "last_norm_mean": float(last_feat.norm(dim=-1).mean().detach().cpu().item()),
                "pooled_norm_mean": float(pooled_feat.norm(dim=-1).mean().detach().cpu().item()),
                "feature_delta_norm_mean": float((feat - last_feat).norm(dim=-1).mean().detach().cpu().item()),
                "attn_entropy_norm_mean": float((entropy / torch.log(torch.tensor(float(max(attn.shape[-1], 2)), device=attn.device))).mean().detach().cpu().item()),
                "attn_max_mean": float(attn.max(dim=-1).values.mean().detach().cpu().item()),
                "has_nan_or_inf": bool(
                    torch.isnan(feat).any().item()
                    or torch.isinf(feat).any().item()
                    or torch.isnan(attn).any().item()
                    or torch.isinf(attn).any().item()
                ),
            }
        return feat, hn, cn

class Decoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2,output_dim=310):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    def forward(self, input, hidden, cell, time_steps):
        out =[]
        out1 = self.fc_out(input)
        out.append(out1)
        out1= out1.unsqueeze(0)  # input = [batch size] to [1, batch size]
        for i in range(time_steps-1):
            output, (hidden, cell) = self.rnn(out1,
                                              (hidden, cell))  # output =[seq len, batch size, hid dim* ndirection]
            out_cur = self.fc_out(output.squeeze(0))  # prediction = [batch size, output dim]
            out.append(out_cur)
            out1 = out_cur.unsqueeze(0)
        out.reverse()
        out = torch.stack(out)
        out = out.transpose(1,0) #batch first
        return out, hidden, cell


#namely The Subject Classifier SD
class DomainClassifier(nn.Module):
    def __init__(self, input_dim =64, output_dim=14):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.classifier(x)
        return x

# The MSE loss
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


def timeStepsShuffle(source_data):
    source_data_1 = source_data.clone()
    #retain the last time step
    curTimeStep_1 = source_data_1[:, -1, :]
    # get data of other time steps
    dim_size = source_data[:, :-1, :].size(1)
    # generate a random sequence
    idxs_1 = list(range(dim_size))
    # generate a shuffled sequence
    random.shuffle(idxs_1)
    # get data corresponding to the shuffled sequence
    else_1 = source_data_1[:, idxs_1, :]
    # add the origin last time step
    result_1 = torch.cat([else_1, curTimeStep_1.unsqueeze(1)], dim=1)
    return result_1


def timeStepsShuffleTransformerFriendly(source_data, shuffle_ratio=0.8):
    # Shuffle a subset of time steps without privileging any fixed step.
    bsz, t, _ = source_data.shape
    num_shuffle = max(2, int(t * shuffle_ratio))
    idx = torch.randperm(t, device=source_data.device)[:num_shuffle]
    perm = idx[torch.randperm(num_shuffle, device=source_data.device)]
    out = source_data.clone()
    out[:, idx, :] = source_data[:, perm, :]
    return out

def maskTimeSteps(source_data, rate):
    source_data_1 = source_data.clone()
    num_zeros = int(source_data.size(1) * rate)
    #mask certain rate of time steps ignoring the last
    zero_indices_1 = torch.randperm(source_data_1.size(1)-1)[:num_zeros]
    source_data_1[:, zero_indices_1,:] = 0
    return source_data_1

def maskChannels(source_data, args, rate):
    # reshape for operating the channel dimension
    source_data_reshaped = source_data.reshape(args.batch_size, args.time_steps, 5, 62)
    source_data_reshaped_1 = source_data_reshaped.clone()
    num_zeros = int(source_data_reshaped.size(-1) * rate)
    # mask certain rate of channels
    zero_indices_1 = torch.randperm(source_data_reshaped_1.size(-1))[:num_zeros]
    source_data_reshaped_1[..., zero_indices_1] = 0
    source_data_reshaped_1 = source_data_reshaped_1.reshape(args.batch_size, args.time_steps, 310)
    return source_data_reshaped_1

def shuffleChannels(source_data, args):
    # reshape for operating the channel dimension
    source_data_reshaped = source_data.reshape(args.batch_size, args.time_steps, 5, 62)
    source_data_reshaped_1 = source_data_reshaped.clone()
    dim_size = source_data_reshaped[..., :].size(-1)
    # # generate a random sequence
    idxs_1 = list(range(dim_size))
    random.shuffle(idxs_1)
    # shuffle channels
    source_data_reshaped_1 = source_data_reshaped_1[..., idxs_1]
    result_1 = source_data_reshaped_1.reshape(args.batch_size, args.time_steps, 310)
    return result_1


def maskChannelBandBlocks(source_data, mask_rate):
    if mask_rate <= 0:
        return source_data
    x = source_data.view(source_data.shape[0], source_data.shape[1], 62, 5)
    keep = (torch.rand(x.shape[0], 1, 62, 5, device=x.device) > mask_rate).float()
    return (x * keep).view(source_data.shape[0], source_data.shape[1], 310)


# Standard 62-channel order used for hemispheric masks (SEED-style 10-20 layout).
CHANNEL_NAMES_62 = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]


def build_hemisphere_masks(channel_names):
    left_keywords = ("1", "3", "5", "7", "9")
    right_keywords = ("2", "4", "6", "8", "10")
    left_idx = []
    right_idx = []
    mid_idx = []
    for i, ch in enumerate(channel_names):
        name = ch.upper()
        if name.endswith("Z"):
            mid_idx.append(i)
        elif any(name.endswith(k) for k in left_keywords) or name in {"T7", "FT7", "TP7", "CB1"}:
            left_idx.append(i)
        elif any(name.endswith(k) for k in right_keywords) or name in {"T8", "FT8", "TP8", "CB2"}:
            right_idx.append(i)
        else:
            mid_idx.append(i)
    mask_left = torch.zeros(len(channel_names), dtype=torch.float32)
    mask_right = torch.zeros(len(channel_names), dtype=torch.float32)
    mask_left[left_idx] = 1.0
    mask_right[right_idx] = 1.0
    return mask_left, mask_right, left_idx, right_idx, mid_idx

# proposed DMMR model
class DMMRPreTrainingModel(nn.Module):
    def __init__(
        self,
        cuda,
        number_of_source=14,
        number_of_category=3,
        batch_size=10,
        time_steps=15,
        use_tgb=False,
        use_mst=False,
        use_csgformer=False,
        use_emt_lite_v2=False,
        use_dmmr_hemi=False,
        lambda_static=0.7,
        gamma_init=0.1,
        channel_embed_dim=8,
        node_dim=16,
        d_model=128,
        transformer_layers=1,
        transformer_heads=4,
        use_temporal_attn_pool=True,
        transformer_friendly_shuffle=True,
        hemi_gate_init=-2.2,
        hemi_dropout=0.1,
        hemi_hidden_dim=128,
        use_band_mask_recon=False,
        band_mask_rate=0.1,
        use_msr=False,
        msr_tau=1.0,
        msr_alpha_init=-1.4,
        msr_hidden_dim=128,
        msr_dropout=0.1,
        msr_memory_init_std=0.02,
        msr_delta_init_std=1e-3,
        use_class_proto_calib=False,
        proto_alpha=0.1,
        proto_temperature=0.2,
        proto_learnable_alpha=True,
        use_feature_calib=False,
        feature_calib_alpha=0.5,
        feature_calib_learnable_alpha=False,
        feature_calib_eps=1e-5,
        feature_calib_use_std=False,
        use_rspa=False,
        rspa_temperature=0.2,
        rspa_alpha_init=0.1,
        rspa_alpha_max=0.5,
        rspa_reliability_tau=1.0,
        rspa_reliability_min=0.8,
        rspa_reliability_max=1.2,
        rspa_hidden_dim=128,
        rspa_dropout=0.1,
        rspa_use_warmup=False,
        rspa_warmup_epochs=2,
        rspa_ramp_epochs=4,
        rspa_use_class_hint=False,
        rspa_class_hint_weight=1.0,
        rspa_class_hint_detach=True,
        rspa_filter_low_conf=False,
        rspa_min_reliability=0.0,
        rspa_source_balance=False,
        rspa_source_cap=0.12,
        rspa_adaptive_gate=False,
        rspa_adaptive_gate_min=0.0,
        rspa_adaptive_gate_max=1.0,
        rspa_centered_adaptive_gate=False,
        rspa_centered_gate_delta=0.2,
        rspa_gate_output_init_std=0.0,
        rspa_logit_blend_weight=0.0,
        use_parallel_tcn=False,
        ptcn_hidden_dim=64,
        ptcn_layers=2,
        ptcn_kernel_size=3,
        ptcn_dropout=0.1,
        ptcn_alpha_init=0.1,
        ptcn_alpha_max=0.3,
        ptcn_delta_init_std=1e-2,
        use_attn_lstm_readout=False,
        attn_lstm_alpha_init=0.3,
        attn_lstm_alpha_max=1.0,
        attn_lstm_dropout=0.1,
        use_mamba_lite=False,
        mamba_d_model=128,
        mamba_layers=1,
        mamba_kernel_size=3,
        mamba_dropout=0.1,
        mamba_alpha_init=0.2,
        mamba_alpha_max=0.8,
        mamba_delta_init_std=0.01,
        use_eeg_conformer=False,
        eeg_conf_node_dim=32,
        eeg_conf_d_model=128,
        eeg_conf_heads=4,
        eeg_conf_layers=1,
        eeg_conf_dropout=0.1,
        eeg_conf_alpha_init=0.25,
        eeg_conf_alpha_max=0.8,
        eeg_conf_delta_init_std=0.02,
        eeg_conf_use_cls_pool=False,
        eeg_conf_warmup_finetune_only=False,
        eeg_conf_warmup_epochs=2,
        eeg_conf_ramp_epochs=2,
        use_patch_transformer=False,
        patch_len=6,
        patch_stride=3,
        patch_d_model=128,
        patch_heads=4,
        patch_layers=1,
        patch_dropout=0.1,
        patch_alpha_init=0.25,
        patch_alpha_max=0.8,
        patch_delta_init_std=0.02,
        mst_alpha_init=0.1,
        tgb_num_channels=62,
        tgb_num_bands=5,
        tgb_dropout=0.1,
        tgb_kernel_size=3,
        tgb_alpha_init=0.1,
        use_gcn_residual=False,
        gcn_alpha_init=0.1,
        gcn_learnable_alpha=True,
        use_self_loop_prior=False,
        self_loop_weight=0.1,
        use_pre_lstm_dropout=False,
        pre_lstm_dropout_p=0.1,
        stable_adj_alpha=1.0,
        use_sspb_v2=False,
        num_subjects_total=15,
        target_subject=0,
        prompt_tau=2.0,
        prompt_alpha_max=0.2,
        prompt_beta_max=0.3,
        prompt_alpha_init=0.1,
        prompt_beta_init=0.1,
        prompt_dropout=0.0,
        use_zero_init_prompt_residual=True,
        prompt_fusion_dropout=0.1,
        prompt_gate_init=0.01,
        use_prompt_gate_warmup=False,
        prompt_warmup_epochs=2,
        prompt_ramp_epochs=2,
        use_prompt_fusion_detach=False,
        use_hyp_contrast=False,
        hyp_proj_dim=32,
        hyp_temperature=0.1,
        hyp_curvature=1.0,
    ):
        super(DMMRPreTrainingModel, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.use_tgb = use_tgb
        self.use_mst = use_mst
        self.use_csgformer = bool(use_csgformer)
        self.use_emt_lite_v2 = bool(use_emt_lite_v2)
        self.use_dmmr_hemi = bool(use_dmmr_hemi)
        self.use_band_mask_recon = bool(use_band_mask_recon)
        self.band_mask_rate = float(band_mask_rate)
        self.last_band_mask_loss = None
        self.use_msr = bool(use_msr)
        self.use_class_proto_calib = bool(use_class_proto_calib)
        self.use_feature_calib = bool(use_feature_calib)
        self.use_rspa = bool(use_rspa)
        self.rspa_logit_blend_weight = float(rspa_logit_blend_weight)
        self.use_parallel_tcn = bool(use_parallel_tcn)
        self.use_attn_lstm_readout = bool(use_attn_lstm_readout)
        self.use_mamba_lite = bool(use_mamba_lite)
        self.use_eeg_conformer = bool(use_eeg_conformer)
        self.use_patch_transformer = bool(use_patch_transformer)
        self.transformer_friendly_shuffle = bool(transformer_friendly_shuffle)
        self.use_sspb_v2 = use_sspb_v2
        self.use_sspb = use_sspb_v2
        self.use_hyp_contrast = bool(use_hyp_contrast)
        self.hyp_proj_dim = int(hyp_proj_dim)
        self.hyp_temperature = float(hyp_temperature)
        self.hyp_curvature = float(hyp_curvature)
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.multiScaleTemporalBlock = MultiScaleSpatiotemporalBlock(
            input_dim=310,
            alpha_init=mst_alpha_init,
        )
        self.csgformerBlock = CSGFormerBlock(
            input_dim=310,
            num_channels=62,
            num_bands=5,
            channel_embed_dim=channel_embed_dim,
            node_dim=node_dim,
            d_model=d_model,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            lambda_static=lambda_static,
            gamma_init=gamma_init,
        )
        self.emtLiteV2Encoder = EmTLiteV2Encoder(
            input_dim=310,
            num_channels=62,
            num_bands=5,
            node_dim=node_dim,
            channel_embed_dim=channel_embed_dim,
            lambda_static=lambda_static,
            gamma_graph_init=gamma_init,
            d_model=d_model,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            use_temporal_attn_pool=use_temporal_attn_pool,
        )
        self.hemiFusion = HemiAsymmetryFusion(
            feature_dim=64,
            hidden_dim=hemi_hidden_dim,
            dropout=hemi_dropout,
            gate_init=hemi_gate_init,
        )
        self.temporalGraphBlock = TemporalGraphBlock(
            num_channels=tgb_num_channels,
            num_bands=tgb_num_bands,
            dropout=tgb_dropout,
            kernel_size=tgb_kernel_size,
            alpha_init=tgb_alpha_init,
            use_gcn_residual=use_gcn_residual,
            gcn_alpha_init=gcn_alpha_init,
            gcn_learnable_alpha=gcn_learnable_alpha,
            use_self_loop_prior=use_self_loop_prior,
            self_loop_weight=self_loop_weight,
            use_pre_lstm_dropout=use_pre_lstm_dropout,
            pre_lstm_dropout_p=pre_lstm_dropout_p,
            stable_adj_alpha=stable_adj_alpha,
        )
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.attentiveSharedEncoder = AttentiveLSTMEncoder(
            input_dim=310,
            hid_dim=64,
            n_layers=1,
            alpha_init=attn_lstm_alpha_init,
            alpha_max=attn_lstm_alpha_max,
            dropout=attn_lstm_dropout,
        )
        self.multiSourceSubjectRouter = MultiSourceSubjectRouter(
            feature_dim=64,
            num_sources=number_of_source,
            hidden_dim=msr_hidden_dim,
            tau=msr_tau,
            alpha_init=msr_alpha_init,
            dropout=msr_dropout,
            memory_init_std=msr_memory_init_std,
            delta_init_std=msr_delta_init_std,
        )
        self.classPrototypeCalibrator = ClassPrototypeCalibrator(
            feature_dim=64,
            num_classes=number_of_category,
            alpha=proto_alpha,
            temperature=proto_temperature,
            learnable_alpha=proto_learnable_alpha,
        )
        self.reliabilitySourcePrototypeAttention = ReliabilitySourcePrototypeAttention(
            feature_dim=64,
            num_subjects=number_of_source,
            num_classes=number_of_category,
            temperature=rspa_temperature,
            alpha_init=rspa_alpha_init,
            alpha_max=rspa_alpha_max,
            reliability_tau=rspa_reliability_tau,
            reliability_min=rspa_reliability_min,
            reliability_max=rspa_reliability_max,
            hidden_dim=rspa_hidden_dim,
            dropout=rspa_dropout,
            use_warmup=rspa_use_warmup,
            warmup_epochs=rspa_warmup_epochs,
            ramp_epochs=rspa_ramp_epochs,
            use_class_hint=rspa_use_class_hint,
            class_hint_weight=rspa_class_hint_weight,
            class_hint_detach=rspa_class_hint_detach,
            filter_low_conf=rspa_filter_low_conf,
            min_reliability=rspa_min_reliability,
            source_balance=rspa_source_balance,
            source_cap=rspa_source_cap,
            adaptive_gate=rspa_adaptive_gate,
            adaptive_gate_min=rspa_adaptive_gate_min,
            adaptive_gate_max=rspa_adaptive_gate_max,
            centered_adaptive_gate=rspa_centered_adaptive_gate,
            centered_gate_delta=rspa_centered_gate_delta,
            gate_output_init_std=rspa_gate_output_init_std,
        )
        self.featureDistributionCalibrator = FeatureDistributionCalibrator(
            feature_dim=64,
            alpha=feature_calib_alpha,
            learnable_alpha=feature_calib_learnable_alpha,
            eps=feature_calib_eps,
            use_std=feature_calib_use_std,
        )
        self.parallelTCNBranch = ParallelTCNBranch(
            input_dim=310,
            feature_dim=64,
            hidden_dim=ptcn_hidden_dim,
            num_layers=ptcn_layers,
            kernel_size=ptcn_kernel_size,
            dropout=ptcn_dropout,
            alpha_init=ptcn_alpha_init,
            alpha_max=ptcn_alpha_max,
            delta_init_std=ptcn_delta_init_std,
        )
        self.mambaLiteBranch = MambaLiteFusionBranch(
            input_dim=310,
            feature_dim=64,
            d_model=mamba_d_model,
            num_layers=mamba_layers,
            kernel_size=mamba_kernel_size,
            dropout=mamba_dropout,
            alpha_init=mamba_alpha_init,
            alpha_max=mamba_alpha_max,
            delta_init_std=mamba_delta_init_std,
        )
        self.eegConformerBranch = EEGConformerFusionBranch(
            input_dim=310,
            feature_dim=64,
            num_channels=62,
            num_bands=5,
            node_dim=eeg_conf_node_dim,
            d_model=eeg_conf_d_model,
            num_heads=eeg_conf_heads,
            num_layers=eeg_conf_layers,
            dropout=eeg_conf_dropout,
            alpha_init=eeg_conf_alpha_init,
            alpha_max=eeg_conf_alpha_max,
            delta_init_std=eeg_conf_delta_init_std,
            use_cls_pool=eeg_conf_use_cls_pool,
            max_time_steps=time_steps,
            use_gate_warmup=eeg_conf_warmup_finetune_only,
            warmup_epochs=eeg_conf_warmup_epochs,
            ramp_epochs=eeg_conf_ramp_epochs,
        )
        self.patchTransformerBranch = PatchTransformerFusionBranch(
            input_dim=310,
            feature_dim=64,
            time_steps=time_steps,
            patch_len=patch_len,
            patch_stride=patch_stride,
            d_model=patch_d_model,
            num_heads=patch_heads,
            num_layers=patch_layers,
            dropout=patch_dropout,
            alpha_init=patch_alpha_init,
            alpha_max=patch_alpha_max,
            delta_init_std=patch_delta_init_std,
        )
        self.sourcePromptBank = SourceSubjectPromptBank(
            feature_dim=64,
            num_subjects_total=num_subjects_total,
            prompt_tau=prompt_tau,
            prompt_alpha_max=prompt_alpha_max,
            prompt_beta_max=prompt_beta_max,
            prompt_alpha_init=prompt_alpha_init,
            prompt_beta_init=prompt_beta_init,
            prompt_dropout=prompt_dropout,
            use_zero_init_prompt_residual=use_zero_init_prompt_residual,
            prompt_fusion_dropout=prompt_fusion_dropout,
            prompt_gate_init=prompt_gate_init,
            use_prompt_gate_warmup=use_prompt_gate_warmup,
            prompt_warmup_epochs=prompt_warmup_epochs,
            prompt_ramp_epochs=prompt_ramp_epochs,
            use_prompt_fusion_detach=use_prompt_fusion_detach,
        )
        assert 0 <= int(target_subject) < int(num_subjects_total), "target_subject {} out of range [0, {})".format(target_subject, num_subjects_total)
        source_prompt_mask = torch.ones(num_subjects_total, dtype=torch.bool)
        source_prompt_mask[int(target_subject)] = False
        self.register_buffer("source_prompt_mask", source_prompt_mask)
        self.source_prompt_count = int(source_prompt_mask.sum().item())
        self.target_subject = int(target_subject)
        hemi_left, hemi_right, left_idx, right_idx, mid_idx = build_hemisphere_masks(CHANNEL_NAMES_62)
        self.register_buffer("hemi_left_mask", hemi_left.view(1, 1, 62, 1), persistent=True)
        self.register_buffer("hemi_right_mask", hemi_right.view(1, 1, 62, 1), persistent=True)
        self.hemi_left_count = int(len(left_idx))
        self.hemi_right_count = int(len(right_idx))
        self.hemi_midline_count = int(len(mid_idx))
        self.hemi_channel_order_name = "standard_62_seed_order_fallback"
        self.mse = MSE()
        self.rspa_pre_logits = None
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')

    def _encode_sequence(self, x, current_epoch=None, enable_eeg_conformer=True):
        # CSGFormer path replaces ABP->LSTM main temporal backbone.
        if self.use_emt_lite_v2:
            return self.emtLiteV2Encoder(x)
        if self.use_csgformer:
            seq_out = self.csgformerBlock(x)  # [B,T,64]
            feat = seq_out[:, -1, :]
            hn = feat.unsqueeze(0)
            cn = torch.zeros_like(hn)
            return feat, hn, cn
        if self.use_mst:
            x = self.multiScaleTemporalBlock(x)
        if self.use_tgb:
            x = self.temporalGraphBlock(x)
        if self.use_attn_lstm_readout:
            shared_last_out, shared_hn, shared_cn = self.attentiveSharedEncoder(x)
            if self.use_mamba_lite:
                shared_last_out = self.mambaLiteBranch(x, shared_last_out)
            if self.use_eeg_conformer and enable_eeg_conformer:
                shared_last_out = self.eegConformerBranch(x, shared_last_out, current_epoch=current_epoch)
            if self.use_patch_transformer:
                shared_last_out = self.patchTransformerBranch(x, shared_last_out)
            return shared_last_out, shared_hn, shared_cn
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)
        if self.use_parallel_tcn:
            shared_last_out = self.parallelTCNBranch(x, shared_last_out)
        if self.use_mamba_lite:
            shared_last_out = self.mambaLiteBranch(x, shared_last_out)
        if self.use_eeg_conformer and enable_eeg_conformer:
            shared_last_out = self.eegConformerBranch(x, shared_last_out, current_epoch=current_epoch)
        if self.use_patch_transformer:
            shared_last_out = self.patchTransformerBranch(x, shared_last_out)
        return shared_last_out, shared_hn, shared_cn

    def _encode_with_hemi(self, x):
        full_feat, full_hn, full_cn = self.sharedEncoder(x)
        x_reshaped = x.view(x.shape[0], x.shape[1], 62, 5)
        x_left = (x_reshaped * self.hemi_left_mask).view(x.shape[0], x.shape[1], 310)
        x_right = (x_reshaped * self.hemi_right_mask).view(x.shape[0], x.shape[1], 310)
        left_feat, _, _ = self.sharedEncoder(x_left)
        right_feat, _, _ = self.sharedEncoder(x_right)
        fused_feat = self.hemiFusion(full_feat, left_feat, right_feat)
        return fused_feat, full_hn, full_cn

    def forward(self, x, corres, subject_id, m=0, mark=0, return_feature=False):
        # Noise Injection, with the proposed method Time Steps Shuffling
        if self.use_emt_lite_v2 and self.transformer_friendly_shuffle:
            x = timeStepsShuffleTransformerFriendly(x)
        else:
            x = timeStepsShuffle(x)
        # The ABP module
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        self.last_band_mask_loss = x.new_zeros(())
        # Encoder the weighted features with one-layer LSTM
        if self.use_dmmr_hemi:
            shared_last_out, shared_hn, shared_cn = self._encode_with_hemi(x)
        else:
            shared_last_out, shared_hn, shared_cn = self._encode_sequence(x, enable_eeg_conformer=False)
        if self.use_band_mask_recon:
            x_masked = maskChannelBandBlocks(x, self.band_mask_rate)
            if self.use_dmmr_hemi:
                mask_feat, mask_hn, mask_cn = self._encode_with_hemi(x_masked)
            else:
                mask_feat, mask_hn, mask_cn = self._encode_sequence(x_masked, enable_eeg_conformer=False)
            x_mask_recon, *_ = eval('self.decoder' + str(mark))(mask_feat, mask_hn, mask_cn, self.time_steps)
            self.last_band_mask_loss = self.mse(x_mask_recon, x.detach())
        # The DG_DANN module
        # The GRL layer in the first stage
        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        # The Subject Discriminator
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        # The domain adversarial loss
        sim_loss = F.nll_loss(subject_predict, subject_id)

        # Build Supervision for Decoders, the inputs are also weighted
        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        if self.use_csgformer:
            # Decoder supervision keeps reconstructing ABP-space features [B,T,310].
            pass
        else:
            if self.use_mst:
                corres = self.multiScaleTemporalBlock(corres)
            if self.use_tgb:
                corres = self.temporalGraphBlock(corres)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            # Reconstruct features in the first stage
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            # The proposed mix method for data augmentation
            mixSubjectFeature += x_out
        if self.use_dmmr_hemi:
            shared_last_out_2, shared_hn_2, shared_cn_2 = self._encode_with_hemi(mixSubjectFeature)
        else:
            shared_last_out_2, shared_hn_2, shared_cn_2 = self._encode_sequence(mixSubjectFeature, enable_eeg_conformer=False)
        for i in range(self.number_of_source):
            # Reconstruct features in the second stage
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            # Compute the reconstructive loss in the second stage only
            rec_loss += self.mse(x_out, splitted_tensors[i])
        if return_feature:
            return rec_loss, sim_loss, shared_last_out
        return rec_loss, sim_loss
class DMMRFineTuningModel(nn.Module):
    def __init__(self, cuda, baseModel, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(DMMRFineTuningModel, self).__init__()
        self.baseModel = copy.deepcopy(baseModel)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        # The ABP module and sharedEncoder are from the pretrained model
        self.attentionLayer = self.baseModel.attentionLayer
        self.multiScaleTemporalBlock = self.baseModel.multiScaleTemporalBlock
        self.use_mst = self.baseModel.use_mst
        self.csgformerBlock = self.baseModel.csgformerBlock
        self.use_csgformer = self.baseModel.use_csgformer
        self.emtLiteV2Encoder = self.baseModel.emtLiteV2Encoder
        self.use_emt_lite_v2 = self.baseModel.use_emt_lite_v2
        self.hemiFusion = self.baseModel.hemiFusion
        self.use_dmmr_hemi = self.baseModel.use_dmmr_hemi
        self.hemi_left_mask = self.baseModel.hemi_left_mask
        self.hemi_right_mask = self.baseModel.hemi_right_mask
        self.hemi_left_count = self.baseModel.hemi_left_count
        self.hemi_right_count = self.baseModel.hemi_right_count
        self.hemi_midline_count = self.baseModel.hemi_midline_count
        self.hemi_channel_order_name = self.baseModel.hemi_channel_order_name
        self.temporalGraphBlock = self.baseModel.temporalGraphBlock
        self.use_tgb = self.baseModel.use_tgb
        self.multiSourceSubjectRouter = self.baseModel.multiSourceSubjectRouter
        self.use_msr = self.baseModel.use_msr
        self.classPrototypeCalibrator = self.baseModel.classPrototypeCalibrator
        self.use_class_proto_calib = self.baseModel.use_class_proto_calib
        self.featureDistributionCalibrator = self.baseModel.featureDistributionCalibrator
        self.use_feature_calib = self.baseModel.use_feature_calib
        self.reliabilitySourcePrototypeAttention = self.baseModel.reliabilitySourcePrototypeAttention
        self.use_rspa = self.baseModel.use_rspa
        self.rspa_logit_blend_weight = getattr(self.baseModel, "rspa_logit_blend_weight", 0.0)
        self.parallelTCNBranch = self.baseModel.parallelTCNBranch
        self.use_parallel_tcn = self.baseModel.use_parallel_tcn
        self.attentiveSharedEncoder = self.baseModel.attentiveSharedEncoder
        self.use_attn_lstm_readout = self.baseModel.use_attn_lstm_readout
        self.mambaLiteBranch = self.baseModel.mambaLiteBranch
        self.use_mamba_lite = self.baseModel.use_mamba_lite
        self.eegConformerBranch = self.baseModel.eegConformerBranch
        self.use_eeg_conformer = self.baseModel.use_eeg_conformer
        self.patchTransformerBranch = self.baseModel.patchTransformerBranch
        self.use_patch_transformer = self.baseModel.use_patch_transformer
        self.sourcePromptBank = self.baseModel.sourcePromptBank
        self.source_prompt_mask = self.baseModel.source_prompt_mask
        self.use_sspb_v2 = self.baseModel.use_sspb_v2
        self.use_sspb = self.use_sspb_v2
        self.use_hyp_contrast = bool(getattr(self.baseModel, "use_hyp_contrast", False))
        self.sharedEncoder = self.baseModel.sharedEncoder
        # Add a new emotion classifier for emotion recognition
        self.cls_fc = nn.Sequential(nn.Linear(64, 64, bias=False), nn.BatchNorm1d(64),
                               nn.ReLU(inplace=True), nn.Linear(64, number_of_category, bias=True))
        self.hyperbolicContrastHead = None
        if self.use_hyp_contrast:
            self.hyperbolicContrastHead = HyperbolicContrastiveHead(
                input_dim=64,
                proj_dim=int(getattr(self.baseModel, "hyp_proj_dim", 32)),
                temperature=float(getattr(self.baseModel, "hyp_temperature", 0.1)),
                curvature=float(getattr(self.baseModel, "hyp_curvature", 1.0)),
            )
        self.mse = MSE()
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')

    def _encode_sequence(self, x, current_epoch=None):
        if self.use_emt_lite_v2:
            feat, _, _ = self.emtLiteV2Encoder(x)
            return feat
        if self.use_csgformer:
            seq_out = self.csgformerBlock(x)
            return seq_out[:, -1, :]
        if self.use_mst:
            x = self.multiScaleTemporalBlock(x)
        if self.use_tgb:
            x = self.temporalGraphBlock(x)
        if self.use_attn_lstm_readout:
            shared_last_out, _, _ = self.attentiveSharedEncoder(x)
            if self.use_mamba_lite:
                shared_last_out = self.mambaLiteBranch(x, shared_last_out)
            if self.use_eeg_conformer:
                shared_last_out = self.eegConformerBranch(x, shared_last_out, current_epoch=current_epoch)
            if self.use_patch_transformer:
                shared_last_out = self.patchTransformerBranch(x, shared_last_out)
            return shared_last_out
        shared_last_out, _, _ = self.sharedEncoder(x)
        if self.use_parallel_tcn:
            shared_last_out = self.parallelTCNBranch(x, shared_last_out)
        if self.use_mamba_lite:
            shared_last_out = self.mambaLiteBranch(x, shared_last_out)
        if self.use_eeg_conformer:
            shared_last_out = self.eegConformerBranch(x, shared_last_out, current_epoch=current_epoch)
        if self.use_patch_transformer:
            shared_last_out = self.patchTransformerBranch(x, shared_last_out)
        return shared_last_out

    def _encode_with_hemi(self, x):
        full_feat, _, _ = self.sharedEncoder(x)
        x_reshaped = x.view(x.shape[0], x.shape[1], 62, 5)
        x_left = (x_reshaped * self.hemi_left_mask).view(x.shape[0], x.shape[1], 310)
        x_right = (x_reshaped * self.hemi_right_mask).view(x.shape[0], x.shape[1], 310)
        left_feat, _, _ = self.sharedEncoder(x_left)
        right_feat, _, _ = self.sharedEncoder(x_right)
        fused_feat = self.hemiFusion(full_feat, left_feat, right_feat)
        return fused_feat

    def forward(self, x, label_src=0, current_epoch=None, return_attn=False, return_hyp=False, return_feature=False):
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        if self.use_dmmr_hemi:
            shared_last_out = self._encode_with_hemi(x)
        else:
            shared_last_out = self._encode_sequence(x, current_epoch=current_epoch)
        if self.use_msr:
            shared_last_out = self.multiSourceSubjectRouter(shared_last_out)
        attn = None
        if self.use_sspb:
            shared_last_out, attn = self.sourcePromptBank(
                shared_last_out,
                self.source_prompt_mask,
                current_epoch=current_epoch,
            )
        if self.use_feature_calib:
            shared_last_out = self.featureDistributionCalibrator(shared_last_out)
        self.rspa_pre_logits = None
        if self.use_rspa:
            rspa_pre_logits = self.cls_fc(shared_last_out)
            self.rspa_pre_logits = rspa_pre_logits.detach()
            rspa_class_probs = None
            if getattr(self.reliabilitySourcePrototypeAttention, "use_class_hint", False):
                rspa_class_probs = F.softmax(rspa_pre_logits, dim=1)
            shared_last_out = self.reliabilitySourcePrototypeAttention(
                shared_last_out,
                current_epoch=current_epoch,
                class_probs=rspa_class_probs,
            )
        x_logits = self.cls_fc(shared_last_out)
        if (
            self.use_rspa
            and self.rspa_logit_blend_weight > 0.0
            and self.rspa_pre_logits is not None
        ):
            blend_w = min(max(self.rspa_logit_blend_weight, 0.0), 1.0)
            x_logits = (1.0 - blend_w) * x_logits + blend_w * self.rspa_pre_logits
        if self.use_class_proto_calib:
            x_logits, _ = self.classPrototypeCalibrator(shared_last_out, x_logits)
        x_pred = F.log_softmax(x_logits, dim=1)
        cls_loss = F.nll_loss(x_pred, label_src.squeeze())
        hyp_loss = shared_last_out.new_zeros(())
        hyp_stats = None
        if self.use_hyp_contrast and (self.hyperbolicContrastHead is not None):
            _, hyp_loss, hyp_stats = self.hyperbolicContrastHead(shared_last_out, label_src.squeeze())

        if return_feature and return_hyp:
            return x_pred, x_logits, cls_loss, attn, hyp_loss, hyp_stats, shared_last_out
        if return_feature and return_attn:
            return x_pred, x_logits, cls_loss, attn, shared_last_out
        if return_feature:
            return x_pred, x_logits, cls_loss, shared_last_out
        if return_hyp:
            return x_pred, x_logits, cls_loss, attn, hyp_loss, hyp_stats
        if return_attn:
            return x_pred, x_logits, cls_loss, attn
        return x_pred, x_logits, cls_loss

class DMMRTestModel(nn.Module):
    def __init__(self, baseModel):
        super(DMMRTestModel, self).__init__()
        self.baseModel = copy.deepcopy(baseModel)
    def forward(self, x, return_attn=False):
        x = self.baseModel.attentionLayer(x, self.baseModel.batch_size, self.baseModel.time_steps)
        if getattr(self.baseModel, "use_dmmr_hemi", False):
            full_feat, _, _ = self.baseModel.sharedEncoder(x)
            x_reshaped = x.view(x.shape[0], x.shape[1], 62, 5)
            x_left = (x_reshaped * self.baseModel.hemi_left_mask).view(x.shape[0], x.shape[1], 310)
            x_right = (x_reshaped * self.baseModel.hemi_right_mask).view(x.shape[0], x.shape[1], 310)
            left_feat, _, _ = self.baseModel.sharedEncoder(x_left)
            right_feat, _, _ = self.baseModel.sharedEncoder(x_right)
            shared_last_out = self.baseModel.hemiFusion(full_feat, left_feat, right_feat)
        elif getattr(self.baseModel, "use_emt_lite_v2", False):
            shared_last_out, _, _ = self.baseModel.emtLiteV2Encoder(x)
        elif getattr(self.baseModel, "use_csgformer", False):
            seq_out = self.baseModel.csgformerBlock(x)
            shared_last_out = seq_out[:, -1, :]
        else:
            if self.baseModel.use_mst:
                x = self.baseModel.multiScaleTemporalBlock(x)
            if self.baseModel.use_tgb:
                x = self.baseModel.temporalGraphBlock(x)
            if getattr(self.baseModel, "use_attn_lstm_readout", False):
                shared_last_out, _, _ = self.baseModel.attentiveSharedEncoder(x)
            else:
                shared_last_out, _, _ = self.baseModel.sharedEncoder(x)
            if getattr(self.baseModel, "use_parallel_tcn", False):
                shared_last_out = self.baseModel.parallelTCNBranch(x, shared_last_out)
            if getattr(self.baseModel, "use_mamba_lite", False):
                shared_last_out = self.baseModel.mambaLiteBranch(x, shared_last_out)
            if getattr(self.baseModel, "use_eeg_conformer", False):
                shared_last_out = self.baseModel.eegConformerBranch(x, shared_last_out, current_epoch=None)
            if getattr(self.baseModel, "use_patch_transformer", False):
                shared_last_out = self.baseModel.patchTransformerBranch(x, shared_last_out)
        if getattr(self.baseModel, "use_msr", False):
            shared_last_out = self.baseModel.multiSourceSubjectRouter(shared_last_out)
        attn = None
        if self.baseModel.use_sspb_v2:
            shared_last_out, attn = self.baseModel.sourcePromptBank(
                shared_last_out,
                self.baseModel.source_prompt_mask,
                current_epoch=None,
            )
        if getattr(self.baseModel, "use_feature_calib", False):
            shared_last_out = self.baseModel.featureDistributionCalibrator(shared_last_out)
        rspa_pre_logits = None
        if getattr(self.baseModel, "use_rspa", False):
            rspa_pre_logits = self.baseModel.cls_fc(shared_last_out)
            rspa_class_probs = None
            if getattr(self.baseModel.reliabilitySourcePrototypeAttention, "use_class_hint", False):
                rspa_class_probs = F.softmax(rspa_pre_logits, dim=1)
            shared_last_out = self.baseModel.reliabilitySourcePrototypeAttention(
                shared_last_out,
                current_epoch=None,
                class_probs=rspa_class_probs,
            )
        x_shared_logits = self.baseModel.cls_fc(shared_last_out)
        if (
            getattr(self.baseModel, "use_rspa", False)
            and getattr(self.baseModel, "rspa_logit_blend_weight", 0.0) > 0.0
            and rspa_pre_logits is not None
        ):
            blend_w = min(max(float(getattr(self.baseModel, "rspa_logit_blend_weight", 0.0)), 0.0), 1.0)
            x_shared_logits = (1.0 - blend_w) * x_shared_logits + blend_w * rspa_pre_logits
        if getattr(self.baseModel, "use_class_proto_calib", False):
            x_shared_logits, _ = self.baseModel.classPrototypeCalibrator(shared_last_out, x_shared_logits)
        if return_attn:
            return x_shared_logits, attn
        return x_shared_logits

############## other noisy injection methods ##############
class PreTrainingWithMaskTimeSteps(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithMaskTimeSteps, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, args, m=0, mark=0):
        x = maskTimeSteps(x, 0.2)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss

class PreTrainingWithMaskChannels(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithMaskChannels, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, args, m=0, mark=0):
        x = maskChannels(x, args, 0.2)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss

class PreTrainingWithChannelsShuffling(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithChannelsShuffling, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, args, m=0, mark=0):
        x = shuffleChannels(x, args)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss

class PreTrainingWithDropout(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15, dropout_rate=0.2):
        super(PreTrainingWithDropout, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        self.dropout = nn.Dropout(dropout_rate)  # noise
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, args, m=0, mark=0):
        x = self.dropout(x)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss


############## noiseInjectionMethods stydy ##############
#w/o mix
class PreTrainingWithoutMix(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithoutMix, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, m=0, mark=0):
        x = timeStepsShuffle(x)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss
#w/o noise
class PreTrainingWithoutNoise(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithoutNoise, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, m=0, mark=0):
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss
#w/o both
class PreTrainingWithoutBothMixAndNoise(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithoutBothMixAndNoise, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, m=0, mark=0):
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss

#return feature of shared feature for T_SNE plots
class ModelReturnFeatures(nn.Module):
    def __init__(self, baseModel, time_steps=15):
        super(ModelReturnFeatures, self).__init__()
        self.baseModel = baseModel
        self.time_steps = time_steps
    def forward(self, x):
        x = self.baseModel.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.baseModel.sharedEncoder(x)
        return x, shared_last_out


