import importlib.util
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from modeling_finetune import (
	labram_base_patch200_200,
	labram_huge_patch200_200,
	labram_large_patch200_200,
)

_root_utils_spec = importlib.util.spec_from_file_location(
	"labram_root_utils", REPO_ROOT / "utils.py"
)
if _root_utils_spec is None or _root_utils_spec.loader is None:
	raise ImportError("Unable to load root utils.py for LaBraM checkpoint utilities")
labram_utils = importlib.util.module_from_spec(_root_utils_spec)
_root_utils_spec.loader.exec_module(labram_utils)


class PositionalEncoding(nn.Module):
	"""Sinusoidal positional encoding for the EAGLE temporal branch."""

	def __init__(self, d_model, max_len=2000):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
		div = torch.exp(
			torch.arange(0, d_model, 2).float()
			* (-torch.log(torch.tensor(10000.0)) / d_model)
		)
		pe[:, 0::2] = torch.sin(pos * div)
		pe[:, 1::2] = torch.cos(pos * div)
		self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

	def forward(self, x):
		time_steps = x.size(1)
		return x + self.pe[:, :time_steps, :]


class BandTokenAdapter(nn.Module):
	"""Inject band-aware EAGLE summaries into LaBraM patch tokens."""

	def __init__(self, token_dim, num_heads=4, dropout=0.1):
		super().__init__()
		self.attn = nn.MultiheadAttention(
			embed_dim=token_dim,
			num_heads=num_heads,
			dropout=dropout,
			batch_first=True,
		)
		self.norm1 = nn.LayerNorm(token_dim)
		self.norm2 = nn.LayerNorm(token_dim)
		self.ffn = nn.Sequential(
			nn.Linear(token_dim, token_dim * 2),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(token_dim * 2, token_dim),
		)

	def forward(self, patch_tokens, band_tokens):
		attn_out, attn_weights = self.attn(
			query=patch_tokens,
			key=band_tokens,
			value=band_tokens,
			need_weights=True,
		)
		fused = self.norm1(patch_tokens + attn_out)
		fused = self.norm2(fused + self.ffn(fused))
		return fused, attn_weights


class EagleLaBraMSemanticDecoder(nn.Module):
	"""
	EAGLE + LaBraM hybrid for bilingual sentence-level semantic decoding.

	Design goals:
	- Keep EAGLE's band-specific inductive bias.
	- Reuse LaBraM as a pretrained patch-token backbone.
	- Produce a language-agnostic semantic embedding suitable for English/Korean
	  same-meaning sentence pair alignment.
	- Preserve the output contract expected by baseline_mix.py and
	  cross_linguistic.py.
	"""

	def __init__(
		self,
		nb_classes,
		Chans=64,
		Samples=1500,
		sfreq=500,
		dropoutRate=0.2,
		F1=16,
		D=2,
		gamma_F1=32,
		reduced_dim=48,
		band_defs=((8, 13), (13, 30), (30, 60), (60, 80)),
		labram_variant="base",
		labram_patch_size=200,
		labram_sfreq=200,
		labram_input_scale=100.0,
		labram_checkpoint=None,
		freeze_labram=False,
		unfreeze_last_n_blocks=0,
		semantic_dim=256,
		adapter_heads=4,
		pair_temperature=0.1,
		default_input_chans=None,
	):
		super().__init__()
		self.sfreq = sfreq
		self.band_defs = band_defs
		self.labram_patch_size = labram_patch_size
		self.labram_sfreq = labram_sfreq
		self.labram_input_scale = labram_input_scale
		self.pair_temperature = pair_temperature
		self.target_time_dim = Samples // 4

		raw_labram_samples = max(
			labram_patch_size,
			int(round(Samples * float(labram_sfreq) / float(sfreq))),
		)
		self.labram_num_windows = max(1, int(round(raw_labram_samples / labram_patch_size)))
		self.labram_eeg_size = self.labram_num_windows * labram_patch_size

		if default_input_chans is None:
			default_input_chans = list(range(Chans + 1))
		self.register_buffer(
			"default_input_chans",
			torch.tensor(default_input_chans, dtype=torch.long),
			persistent=False,
		)

		def freq_kernel(low_f):
			cycles = 1.5
			klen = int((cycles * sfreq) / low_f)
			return max(8, min(klen, Samples))

		self.band_branches = nn.ModuleList()
		self.kernel_lengths = []
		for lo, hi in band_defs:
			klen = freq_kernel(lo)
			self.kernel_lengths.append(klen)

			temp = nn.Conv2d(
				1,
				F1,
				kernel_size=(1, klen),
				padding=(0, klen // 2),
				bias=False,
			)
			bn1 = nn.BatchNorm2d(F1)

			spatial = nn.Conv2d(
				F1,
				F1 * D,
				kernel_size=(Chans, 1),
				groups=F1,
				bias=False,
			)
			bn2 = nn.BatchNorm2d(F1 * D)

			if hi >= 60:
				gamma_extra = nn.Sequential(
					nn.Conv2d(
						F1 * D,
						gamma_F1,
						kernel_size=(1, 9),
						padding=(0, 4),
						bias=False,
					),
					nn.BatchNorm2d(gamma_F1),
					nn.ELU(),
					nn.Conv2d(
						gamma_F1,
						gamma_F1,
						kernel_size=(1, 5),
						padding=(0, 2),
						dilation=(1, 2),
						groups=gamma_F1,
						bias=False,
					),
					nn.BatchNorm2d(gamma_F1),
				)
			else:
				gamma_extra = nn.Identity()

			branch = nn.Sequential(
				temp,
				bn1,
				nn.ELU(),
				spatial,
				bn2,
				nn.ELU(),
				nn.AvgPool2d((1, 2)),
				nn.Dropout(dropoutRate),
				gamma_extra,
				nn.AvgPool2d((1, 2)),
				nn.Dropout(dropoutRate),
				nn.AdaptiveAvgPool2d((1, self.target_time_dim)),
			)
			self.band_branches.append(branch)

		self.band_channels = [gamma_F1 if hi >= 60 else F1 * D for _, hi in band_defs]
		feature_channels = sum(self.band_channels)
		self.num_bands = len(band_defs)

		self.dim_reduction = nn.Sequential(
			nn.Conv1d(feature_channels, reduced_dim, kernel_size=1),
			nn.BatchNorm1d(reduced_dim),
			nn.ELU(),
		)
		self.band_score = nn.ModuleList(
			[
				nn.Sequential(
					nn.Linear(channels, max(4, channels // 4)),
					nn.ReLU(),
					nn.Linear(max(4, channels // 4), 1),
				)
				for channels in self.band_channels
			]
		)
		self.pos_enc = PositionalEncoding(reduced_dim, max_len=self.target_time_dim)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=reduced_dim,
			nhead=4,
			batch_first=True,
		)
		self.seq_model = nn.TransformerEncoder(encoder_layer, num_layers=2)

		self.labram = self._build_labram_backbone(labram_variant)
		if labram_checkpoint:
			self._load_labram_checkpoint(labram_checkpoint)
		self._set_labram_trainability(freeze_labram, unfreeze_last_n_blocks)

		self.labram_dim = self.labram.embed_dim
		self.band_token_projs = nn.ModuleList(
			[nn.Linear(channels, self.labram_dim) for channels in self.band_channels]
		)
		self.band_token_norm = nn.LayerNorm(self.labram_dim)
		self.eagle_global_proj = nn.Sequential(
			nn.LayerNorm(reduced_dim),
			nn.Linear(reduced_dim, self.labram_dim),
			nn.GELU(),
		)
		self.adapter = BandTokenAdapter(
			token_dim=self.labram_dim,
			num_heads=adapter_heads,
			dropout=dropoutRate,
		)

		fusion_dim = self.labram_dim * 2 + reduced_dim
		self.semantic_projector = nn.Sequential(
			nn.LayerNorm(fusion_dim),
			nn.Linear(fusion_dim, semantic_dim),
			nn.GELU(),
			nn.Dropout(dropoutRate),
			nn.Linear(semantic_dim, semantic_dim),
		)
		self.classifier = nn.Sequential(
			nn.LayerNorm(semantic_dim),
			nn.Linear(semantic_dim, 128),
			nn.GELU(),
			nn.Dropout(dropoutRate),
			nn.Linear(128, nb_classes, bias=False),
		)

	def _build_labram_backbone(self, variant):
		builders = {
			"base": labram_base_patch200_200,
			"large": labram_large_patch200_200,
			"huge": labram_huge_patch200_200,
		}
		if variant not in builders:
			raise ValueError(f"Unsupported LaBraM variant: {variant}")
		return builders[variant](num_classes=0, use_mean_pooling=True)

	def _load_labram_checkpoint(self, checkpoint_path):
		try:
			checkpoint = torch.load(checkpoint_path, map_location="cpu")
		except pickle.UnpicklingError as exc:
			if "Weights only load failed" not in str(exc):
				raise
			checkpoint = torch.load(
				checkpoint_path,
				map_location="cpu",
				weights_only=False,
			)
		checkpoint_model = None
		for model_key in ("model", "module"):
			if model_key in checkpoint:
				checkpoint_model = checkpoint[model_key]
				break
		if checkpoint_model is None:
			checkpoint_model = checkpoint

		stripped_model = {}
		has_student_prefix = any(key.startswith("student.") for key in checkpoint_model)
		if has_student_prefix:
			for key, value in checkpoint_model.items():
				if key.startswith("student."):
					stripped_model[key[8:]] = value
			checkpoint_model = stripped_model

		state_dict = self.labram.state_dict()
		for key in ["head.weight", "head.bias"]:
			if key in checkpoint_model and key in state_dict:
				if checkpoint_model[key].shape != state_dict[key].shape:
					del checkpoint_model[key]

		for key in list(checkpoint_model.keys()):
			if "relative_position_index" in key:
				checkpoint_model.pop(key)

		labram_utils.load_state_dict(self.labram, checkpoint_model, prefix="")

	def _set_labram_trainability(self, freeze_labram, unfreeze_last_n_blocks):
		for param in self.labram.parameters():
			param.requires_grad = not freeze_labram

		if freeze_labram and unfreeze_last_n_blocks > 0:
			for block in self.labram.blocks[-unfreeze_last_n_blocks:]:
				for param in block.parameters():
					param.requires_grad = True
			for module in [self.labram.norm, self.labram.fc_norm, self.labram.head]:
				if module is None:
					continue
				for param in module.parameters():
					param.requires_grad = True

	def _resolve_input_chans(self, input_chans, batch_size, device):
		if input_chans is None:
			return self.default_input_chans.to(device=device)
		if isinstance(input_chans, torch.Tensor):
			return input_chans.to(device=device)
		return torch.tensor(input_chans, dtype=torch.long, device=device)

	def _prepare_labram_input(self, x):
		raw = x.squeeze(1) / self.labram_input_scale
		if raw.size(-1) != self.labram_eeg_size:
			raw = F.interpolate(
				raw,
				size=self.labram_eeg_size,
				mode="linear",
				align_corners=False,
			)
		batch_size, num_chans, _ = raw.shape
		return raw.view(batch_size, num_chans, self.labram_num_windows, self.labram_patch_size)

	def _extract_eagle_features(self, x):
		branch_feats = []
		band_temporal = []
		band_globals = []
		scores = []

		for idx, branch in enumerate(self.band_branches):
			feat = branch(x)
			branch_feats.append(feat)

			temporal_map = feat.mean(dim=1).squeeze(1)
			band_temporal.append(temporal_map)

			global_feat = feat.mean(dim=[2, 3])
			band_globals.append(global_feat)
			scores.append(self.band_score[idx](global_feat))

		scores = torch.cat(scores, dim=1)
		band_weights = F.softmax(scores, dim=1)

		weighted_branch_feats = []
		for idx, feat in enumerate(branch_feats):
			weight = band_weights[:, idx].view(-1, 1, 1, 1)
			weighted_branch_feats.append(feat * weight)

		feat = torch.cat(weighted_branch_feats, dim=1)
		feat = feat.squeeze(2)
		feat = self.dim_reduction(feat)
		feat = feat.transpose(1, 2)
		feat = self.pos_enc(feat)
		seq_out = self.seq_model(feat)
		eagle_global = seq_out.mean(dim=1)

		band_temporal = torch.stack(band_temporal, dim=1)
		temporal_attention = band_temporal * band_weights.unsqueeze(-1)

		return seq_out, eagle_global, band_globals, band_weights, temporal_attention

	def _build_band_tokens(self, band_globals):
		band_tokens = []
		for proj, global_feat in zip(self.band_token_projs, band_globals):
			band_tokens.append(proj(global_feat))
		band_tokens = torch.stack(band_tokens, dim=1)
		return self.band_token_norm(band_tokens)

	def forward(self, x, input_chans=None, return_attention=False, return_patch_tokens=False):
		if x.dim() == 3:
			x = x.unsqueeze(1)
		if x.dim() != 4:
			raise ValueError("Expected input with shape (B, 1, C, T) or (B, C, T)")

		batch_size = x.size(0)
		device = x.device
		input_chans = self._resolve_input_chans(input_chans, batch_size, device)

		seq_out, eagle_global, band_globals, band_weights, temporal_attention = self._extract_eagle_features(x)

		labram_input = self._prepare_labram_input(x)
		patch_tokens = self.labram.forward_features(
			labram_input,
			input_chans=input_chans,
			return_patch_tokens=True,
		)
		band_tokens = self._build_band_tokens(band_globals)
		adapted_tokens, token_band_attention = self.adapter(patch_tokens, band_tokens)

		labram_global = patch_tokens.mean(dim=1)
		adapted_global = adapted_tokens.mean(dim=1)
		eagle_context = self.eagle_global_proj(eagle_global)

		fused = torch.cat([labram_global, adapted_global + eagle_context, eagle_global], dim=1)
		semantic_embedding = self.semantic_projector(fused)
		semantic_embedding = F.normalize(semantic_embedding, dim=-1)
		logits = self.classifier(semantic_embedding)

		if return_patch_tokens:
			return logits, semantic_embedding, band_weights, temporal_attention, adapted_tokens

		if return_attention:
			self.last_token_band_attention = token_band_attention.detach()
			return logits, semantic_embedding, band_weights, temporal_attention

		return logits, semantic_embedding

	def forward_pair(self, x_a, x_b, input_chans=None, return_attention=False):
		out_a = self.forward(x_a, input_chans=input_chans, return_attention=return_attention)
		out_b = self.forward(x_b, input_chans=input_chans, return_attention=return_attention)

		emb_a = out_a[1]
		emb_b = out_b[1]
		alignment_loss = self.compute_semantic_alignment_loss(emb_a, emb_b)
		consistency_loss = self.compute_band_consistency_loss(out_a[2], out_b[2])

		return {
			"outputs_a": out_a,
			"outputs_b": out_b,
			"alignment_loss": alignment_loss,
			"band_consistency_loss": consistency_loss,
		}

	def compute_semantic_alignment_loss(self, emb_a, emb_b):
		emb_a = F.normalize(emb_a, dim=-1)
		emb_b = F.normalize(emb_b, dim=-1)
		logits = emb_a @ emb_b.t() / self.pair_temperature
		targets = torch.arange(emb_a.size(0), device=emb_a.device)
		loss_ab = F.cross_entropy(logits, targets)
		loss_ba = F.cross_entropy(logits.t(), targets)
		return 0.5 * (loss_ab + loss_ba)

	def compute_band_consistency_loss(self, band_weights_a, band_weights_b):
		band_weights_a = torch.clamp(band_weights_a, min=1e-6)
		band_weights_b = torch.clamp(band_weights_b, min=1e-6)
		mean_dist = 0.5 * (band_weights_a + band_weights_b)
		kl_a = F.kl_div(mean_dist.log(), band_weights_a, reduction="batchmean")
		kl_b = F.kl_div(mean_dist.log(), band_weights_b, reduction="batchmean")
		return 0.5 * (kl_a + kl_b)


class eagle_w_labram(EagleLaBraMSemanticDecoder):
	pass


ExtendedSemanticEEGNetWithLaBraM = EagleLaBraMSemanticDecoder

