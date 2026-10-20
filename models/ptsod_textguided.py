import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from easydict import EasyDict
from pointnet2_utils import index_points, square_distance, PointNetFeaturePropagation
from openpoints.models.backbone.pointnext import PointNextEncoder, PointNextDecoder


class PretrainedTextEncoder(nn.Module):
    def __init__(self, d_model=64, clip_model_name="ViT-B/32"):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model_name, device="cpu", jit=False)
        clip_dim = self.clip_model.text_projection.shape[1]
        self.proj = nn.Linear(clip_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self._clip_feature_cache = {}

        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def _encode_clip(self, text_list):
        clip_device = next(self.clip_model.parameters()).device
        text_keys = [str(text) for text in text_list]
        missing_keys = list(
            dict.fromkeys(
                text
                for text in text_keys
                if text not in self._clip_feature_cache
            )
        )

        if missing_keys:
            tokens = clip.tokenize(missing_keys, truncate=True).to(clip_device)
            missing_features = self.clip_model.encode_text(tokens).detach().cpu()
            for text, feature in zip(missing_keys, missing_features):
                self._clip_feature_cache[text] = feature.clone()

        return torch.stack(
            [self._clip_feature_cache[text] for text in text_keys], dim=0
        ).to(clip_device, non_blocking=True)

    def forward(self, text_list, device: torch.device):
        features = self._encode_clip(text_list)
        projection_device = self.proj.weight.device
        features = features.to(projection_device)
        features = self.proj(features)
        features = self.norm(features)
        return features.to(device)


class TGPA(nn.Module):

    def __init__(self, dim_point, dim_text=64):
        super().__init__()
        self.dim_point = dim_point

        self.text_mlp = nn.Sequential(nn.Linear(dim_text, dim_text),nn.ReLU(inplace=True),nn.Linear(dim_text, dim_point),nn.Dropout(0.0),)
        self.region_mlp = nn.Sequential(nn.Linear(1, dim_point),nn.LayerNorm(dim_point),nn.ReLU(inplace=True),)

    def forward(self, point_feat, text_feat):
        if text_feat.dim() == 3:
            text_feat = text_feat.squeeze(1)

        _, channels, _ = point_feat.shape
        text_query = self.text_mlp(text_feat)

        point_tokens = point_feat.transpose(1, 2).contiguous()

        similarity = torch.einsum("bnc,bc->bn", point_tokens, text_query)
        similarity = similarity / (channels ** 0.5)

        spatial_weight = torch.sigmoid(similarity)

        region_prior = self.region_mlp(spatial_weight.unsqueeze(-1))
        region_prior = region_prior.transpose(1, 2).contiguous()

        return point_feat + region_prior


class get_model(nn.Module):
    def __init__(self, num_classes=2, nsample=None):
        super().__init__()
        if nsample is None:
            nsample = [32, 32, 16, 16]
        self.nsample = nsample

        encoder_dims = [64, 128, 256, 512, 1024]
        self.pointnext_encoder = PointNextEncoder(
            in_channels=9,
            width=64,
            blocks=[1, 3, 5, 3, 3],
            strides=[1, 4, 4, 4, 4],
            block="InvResMLP",
            nsample=32,
            radius=0.1,
            group_args=EasyDict({"NAME": "ballquery"}),
            aggr_args={"feature_type": "dp_fj", "reduction": "max"},
            conv_args={},
            expansion=4,
            use_res=True,
        )
        self.pointnext_decoder = PointNextDecoder(
            encoder_channel_list=encoder_dims,
            decoder_layers=2,
            decoder_stages=4,
        )

        self.text_encoder = PretrainedTextEncoder(d_model=64)

        self.encoder_text_layers = [0]
        self.encoder_tgpa = nn.ModuleDict()
        for index in self.encoder_text_layers:
            fusion_dim = encoder_dims[index + 1]
            self.encoder_tgpa[str(index)] = TGPA(
                dim_point=fusion_dim,
                dim_text=64,
            )

        self.shortcut_proj = nn.Sequential(
            nn.Conv1d(1024 + 64 + 64, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.use_encoder_last_tgpa = True
        if self.use_encoder_last_tgpa:
            self.encoder_last_tgpa = TGPA(dim_point=1024, dim_text=64)

        self.decoder_text_layers = [3]
        decoder_dims = [512, 256, 128, 64]
        self.decoder_tgpa = nn.ModuleDict()
        for index in self.decoder_text_layers:
            self.decoder_tgpa[str(index)] = TGPA(
                dim_point=decoder_dims[index],
                dim_text=64,
            )

        self.pred_head = nn.Sequential(
            nn.Conv1d(64 + 64, 64, 1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.7),
            nn.Conv1d(64, num_classes, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.Conv1d(64 + 64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, 1),
        )

        self.aux_fp_first = PointNetFeaturePropagation(576, [64, 64])
        self.aux_head_first = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, num_classes, 1),
        )

        self.cross_modal_proj = nn.Identity()

    def _encode_text(self, text_list, device, batch_size):
        if text_list is None:
            text_list = ["salient object"] * batch_size
        elif not isinstance(text_list, (list, tuple)):
            text_list = [str(text_list)] * batch_size
        elif len(text_list) == 1 and batch_size > 1:
            text_list = list(text_list) * batch_size

        return self.text_encoder(list(text_list), device=device)

    def forward(self, x, full_pts=None, text_list=None, text_feats=None):
        xyz = x[:, :3, :]
        batch_size = x.shape[0]
        text_features = (
            self._encode_text(text_list, x.device, batch_size)
            if text_feats is None
            else text_feats
        )

        point_coords = xyz.transpose(1, 2).contiguous()
        point_coords = point_coords - point_coords.mean(dim=1, keepdim=True)
        point_scale = torch.sqrt(
            torch.sum(point_coords * point_coords, dim=-1)
        ).amax(dim=1, keepdim=True).unsqueeze(-1)
        point_coords = point_coords / point_scale.clamp_min(1e-6)

        point_list = [point_coords.contiguous()]
        feature_list = [x]

        for stage_index, stage in enumerate(self.pointnext_encoder.encoder):
            new_point, new_feature = stage(
                [point_list[-1], feature_list[-1]]
            )
            encoder_index = stage_index - 1

            if encoder_index in self.encoder_text_layers:
                new_feature = self.encoder_tgpa[str(encoder_index)](
                    new_feature,
                    text_features,
                )

            point_list.append(new_point)
            feature_list.append(new_feature)

        shallow_points = point_list[1]
        shallow_features = feature_list[1]
        bottleneck_points = point_list[-1]
        bottleneck_features = feature_list[-1]

        distance = square_distance(bottleneck_points, shallow_points)
        nearest_index = torch.argmin(distance, dim=-1)
        shortcut_features = index_points(
            shallow_features.transpose(1, 2).contiguous(),
            nearest_index,
        ).transpose(1, 2).contiguous()
        shortcut_features = shortcut_features.detach()

        global_features = torch.max(
            shallow_features, dim=2, keepdim=True
        )[0]
        global_features = global_features.expand(
            -1, -1, bottleneck_points.shape[1]
        ).detach()

        enhanced_features = torch.cat(
            [bottleneck_features, shortcut_features, global_features],
            dim=1,
        )
        x = self.shortcut_proj(enhanced_features)

        if self.use_encoder_last_tgpa:
            x = self.encoder_last_tgpa(x, text_features)
        feature_list[-1] = x

        auxiliary_predictions = []
        decoder_indices = range(
            -1,
            -len(self.pointnext_decoder.decoder) - 1,
            -1,
        )

        for decoder_stage, sparse_index in enumerate(decoder_indices):
            dense_index = sparse_index - 1
            x = self.pointnext_decoder.decoder[sparse_index][0](
                [point_list[dense_index], feature_list[dense_index]],
                [point_list[sparse_index], feature_list[sparse_index]],
            )
            feature_list[dense_index] = x

            if decoder_stage in self.decoder_text_layers:
                x = self.decoder_tgpa[str(decoder_stage)](
                    x,
                    text_features,
                )

            feature_list[dense_index] = x

            if self.training and decoder_stage == 0:
                auxiliary_features = self.aux_fp_first(
                    shallow_points.transpose(1, 2).contiguous(),
                    point_list[dense_index].transpose(1, 2).contiguous(),
                    shallow_features,
                    x,
                )
                auxiliary_logits = self.aux_head_first(auxiliary_features)
                auxiliary_predictions.append(
                    F.log_softmax(
                        auxiliary_logits.transpose(2, 1).contiguous(),
                        dim=-1,
                    )
                )

        decoder_features = self.cross_modal_proj(x)
        concatenated_features = torch.cat(
            [x, shallow_features], dim=1
        )
        logits = self.pred_head(
            concatenated_features
        ).transpose(2, 1).contiguous()

        return (
            F.log_softmax(logits, dim=-1),
            logits,
            auxiliary_predictions,
            concatenated_features,
            self.boundary_head(concatenated_features),
            xyz,
            decoder_features,
            text_features,
        )


def compute_cross_modal_loss(
    dense_features,
    text_features,
    target_mask,
    sample_size=256,
):
    batch_size, num_points = target_mask.shape

    if num_points > sample_size:
        sampled_features = []
        sampled_masks = []

        for batch_index in range(batch_size):
            mask = target_mask[batch_index]
            foreground_indices = torch.where(mask >= 0.5)[0]
            background_indices = torch.where(mask < 0.5)[0]

            num_foreground = min(
                len(foreground_indices), sample_size // 2
            )
            num_background = min(
                len(background_indices), sample_size - num_foreground
            )
            num_foreground = min(
                len(foreground_indices), sample_size - num_background
            )

            if len(foreground_indices) > 0:
                foreground_sample = foreground_indices[
                    torch.randperm(
                        len(foreground_indices),
                        device=foreground_indices.device,
                    )[:num_foreground]
                ]
            else:
                foreground_sample = foreground_indices

            if len(background_indices) > 0:
                background_sample = background_indices[
                    torch.randperm(
                        len(background_indices),
                        device=background_indices.device,
                    )[:num_background]
                ]
            else:
                background_sample = background_indices

            sampled_index = torch.cat(
                [foreground_sample, background_sample]
            )
            sampled_features.append(
                dense_features[batch_index, :, sampled_index]
            )
            sampled_masks.append(mask[sampled_index])

        dense_features = torch.stack(sampled_features, dim=0)
        target_mask = torch.stack(sampled_masks, dim=0)

    dense_normalized = F.normalize(
        dense_features.transpose(1, 2).contiguous(),
        p=2,
        dim=-1,
    )
    text_normalized = F.normalize(
        text_features, p=2, dim=-1
    ).unsqueeze(2)
    similarity = torch.bmm(
        dense_normalized, text_normalized
    ).squeeze(2)

    foreground_mask = (target_mask >= 0.5).float()
    background_mask = (target_mask < 0.5).float()

    pull_loss = (
        ((1.0 - similarity) * foreground_mask).sum()
        / (foreground_mask.sum() + 1e-6)
    )
    push_loss = (
        (F.relu(similarity) * background_mask).sum()
        / (background_mask.sum() + 1e-6)
    )

    return pull_loss + 0.5 * push_loss


def get_boundary_gt(xyz, target, k=16):
    batch_size, _, num_points = xyz.shape
    xyz_transposed = xyz.transpose(1, 2).contiguous()
    distance = square_distance(xyz_transposed, xyz_transposed)
    _, indices = torch.topk(
        distance,
        k,
        dim=-1,
        largest=False,
        sorted=False,
    )
    flat_indices = indices.view(batch_size, num_points * k)
    gathered_targets = torch.gather(
        target, 1, flat_indices
    ).view(batch_size, num_points, k)

    return (
        gathered_targets.max(dim=-1)[0]
        != gathered_targets.min(dim=-1)[0]
    ).float()


class PC_SharpnessLoss(nn.Module):
    def __init__(self, cross_modal_weight=0.2, boundary_weight=0.5):
        super().__init__()
        self.cross_modal_weight = cross_modal_weight
        self.boundary_weight = boundary_weight

    def compute_saliency_loss(
        self,
        pred_log_softmax,
        target,
        weights=None,
    ):
        pred_log_softmax = pred_log_softmax.reshape(-1, 2)
        target = target.reshape(-1).long()
        return F.nll_loss(
            pred_log_softmax,
            target,
            weight=weights,
        )

    def forward(
        self,
        pred_log_softmax,
        target,
        weights=None,
        bound_pred=None,
        xyz=None,
        dense_features_64=None,
        text_features=None,
    ):
        total_loss = self.compute_saliency_loss(
            pred_log_softmax,
            target,
            weights,
        )

        if dense_features_64 is not None and text_features is not None:
            cross_modal_loss = compute_cross_modal_loss(
                dense_features_64,
                text_features,
                target,
            )
            total_loss = (
                total_loss
                + self.cross_modal_weight * cross_modal_loss
            )

        if bound_pred is not None and xyz is not None:
            with torch.no_grad():
                boundary_target = get_boundary_gt(
                    xyz,
                    target.long(),
                    k=16,
                )

            boundary_loss = F.binary_cross_entropy_with_logits(
                bound_pred.squeeze(1),
                boundary_target,
            )
            total_loss = (
                total_loss
                + self.boundary_weight * boundary_loss
            )

        return total_loss


def get_loss():
    return PC_SharpnessLoss(
        cross_modal_weight=0.2,
        boundary_weight=0.5,
    )
