# -*- coding: utf-8 -*-
#
# Hand-written in the style of the pyside6-uic generated UI files in this directory
# (see AnimaUI.py / FluxUI.py), since there is no .ui source for this panel.

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget)

from modules.DragDropLineEdit import DragDropLineEdit
from modules.ScrollOnSelect import ComboBox, DoubleSpinBox, SpinBox


class Ui_moss_ui(object):
    def setupUi(self, moss_ui):
        if not moss_ui.objectName():
            moss_ui.setObjectName(u"moss_ui")
        moss_ui.resize(523, 620)
        self.gridLayout = QGridLayout(moss_ui)
        self.gridLayout.setObjectName(u"gridLayout")
        self.moss_training_box = QGroupBox(moss_ui)
        self.moss_training_box.setObjectName(u"moss_training_box")
        self.moss_training_box.setCheckable(True)
        self.moss_training_box.setChecked(False)
        self.gridLayout_2 = QGridLayout(self.moss_training_box)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")

        # DiT model (maps to pretrained_model_name_or_path)
        self.label = QLabel(self.moss_training_box)
        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.label)
        self.horizontalLayout = QHBoxLayout()
        self.dit_model_input = DragDropLineEdit(self.moss_training_box)
        self.dit_model_input.setObjectName(u"dit_model_input")
        self.horizontalLayout.addWidget(self.dit_model_input)
        self.dit_model_selector = QPushButton(self.moss_training_box)
        self.horizontalLayout.addWidget(self.dit_model_selector)
        self.formLayout_3.setLayout(0, QFormLayout.FieldRole, self.horizontalLayout)

        # MOSS text encoder (Qwen3.5-9B) (maps to --moss_text_encoder)
        self.label_2 = QLabel(self.moss_training_box)
        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.label_2)
        self.horizontalLayout_2 = QHBoxLayout()
        self.moss_text_encoder_input = DragDropLineEdit(self.moss_training_box)
        self.moss_text_encoder_input.setObjectName(u"moss_text_encoder_input")
        self.horizontalLayout_2.addWidget(self.moss_text_encoder_input)
        self.moss_text_encoder_selector = QPushButton(self.moss_training_box)
        self.horizontalLayout_2.addWidget(self.moss_text_encoder_selector)
        self.formLayout_3.setLayout(1, QFormLayout.FieldRole, self.horizontalLayout_2)

        # VAE (maps to --vae)
        self.label_3 = QLabel(self.moss_training_box)
        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.label_3)
        self.horizontalLayout_3 = QHBoxLayout()
        self.vae_model_input = DragDropLineEdit(self.moss_training_box)
        self.vae_model_input.setObjectName(u"vae_model_input")
        self.horizontalLayout_3.addWidget(self.vae_model_input)
        self.vae_model_selector = QPushButton(self.moss_training_box)
        self.horizontalLayout_3.addWidget(self.vae_model_selector)
        self.formLayout_3.setLayout(2, QFormLayout.FieldRole, self.horizontalLayout_3)

        # Tokenizer dir (maps to --moss_tokenizer_dir)
        self.label_tok = QLabel(self.moss_training_box)
        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.label_tok)
        self.horizontalLayout_tok = QHBoxLayout()
        self.tokenizer_dir_input = DragDropLineEdit(self.moss_training_box)
        self.tokenizer_dir_input.setObjectName(u"tokenizer_dir_input")
        self.horizontalLayout_tok.addWidget(self.tokenizer_dir_input)
        self.tokenizer_dir_selector = QPushButton(self.moss_training_box)
        self.horizontalLayout_tok.addWidget(self.tokenizer_dir_selector)
        self.formLayout_3.setLayout(3, QFormLayout.FieldRole, self.horizontalLayout_tok)

        # Max token length + Blocks to swap
        self.horizontalLayout_tokens = QHBoxLayout()
        self.max_token_label = QLabel(self.moss_training_box)
        self.horizontalLayout_tokens.addWidget(self.max_token_label)
        self.max_token_input = SpinBox(self.moss_training_box)
        self.max_token_input.setMaximum(16777215)
        self.max_token_input.setValue(4096)
        self.horizontalLayout_tokens.addWidget(self.max_token_input)

        self.label_blocks_to_swap = QLabel(self.moss_training_box)
        self.horizontalLayout_tokens.addWidget(self.label_blocks_to_swap)
        self.blocks_to_swap_input = SpinBox(self.moss_training_box)
        self.blocks_to_swap_input.setMinimum(0)
        self.blocks_to_swap_input.setMaximum(100)
        self.blocks_to_swap_input.setValue(0)
        self.horizontalLayout_tokens.addWidget(self.blocks_to_swap_input)
        self.formLayout_3.setLayout(4, QFormLayout.SpanningRole, self.horizontalLayout_tokens)

        # Timestep sampling + shift + sigmoid scale
        self.horizontalLayout_sampling = QHBoxLayout()
        self.label_sampling = QLabel(self.moss_training_box)
        self.horizontalLayout_sampling.addWidget(self.label_sampling)
        self.timestep_sampling_selector = ComboBox(self.moss_training_box)
        self.timestep_sampling_selector.addItem("moss")
        self.timestep_sampling_selector.addItem("sigma")
        self.timestep_sampling_selector.addItem("uniform")
        self.timestep_sampling_selector.addItem("sigmoid")
        self.timestep_sampling_selector.addItem("shift")
        self.horizontalLayout_sampling.addWidget(self.timestep_sampling_selector)

        self.label_shift = QLabel(self.moss_training_box)
        self.horizontalLayout_sampling.addWidget(self.label_shift)
        self.discrete_flow_shift_input = DoubleSpinBox(self.moss_training_box)
        self.discrete_flow_shift_input.setDecimals(2)
        self.discrete_flow_shift_input.setSingleStep(0.1)
        self.discrete_flow_shift_input.setValue(1.0)
        self.horizontalLayout_sampling.addWidget(self.discrete_flow_shift_input)

        self.label_sigmoid = QLabel(self.moss_training_box)
        self.horizontalLayout_sampling.addWidget(self.label_sigmoid)
        self.sigmoid_scale_input = DoubleSpinBox(self.moss_training_box)
        self.sigmoid_scale_input.setDecimals(2)
        self.sigmoid_scale_input.setSingleStep(0.1)
        self.sigmoid_scale_input.setValue(1.0)
        self.horizontalLayout_sampling.addWidget(self.sigmoid_scale_input)
        self.formLayout_3.setLayout(5, QFormLayout.SpanningRole, self.horizontalLayout_sampling)

        # MOSS-only sampling distribution params (timestep_sampling == "moss")
        self.horizontalLayout_moss_dist = QHBoxLayout()
        self.label_logit_mean = QLabel(self.moss_training_box)
        self.horizontalLayout_moss_dist.addWidget(self.label_logit_mean)
        self.logit_mean_input = DoubleSpinBox(self.moss_training_box)
        self.logit_mean_input.setDecimals(3)
        self.logit_mean_input.setMinimum(-10.0)
        self.logit_mean_input.setMaximum(10.0)
        self.logit_mean_input.setSingleStep(0.01)
        self.logit_mean_input.setValue(1.09)
        self.horizontalLayout_moss_dist.addWidget(self.logit_mean_input)

        self.label_logit_std = QLabel(self.moss_training_box)
        self.horizontalLayout_moss_dist.addWidget(self.label_logit_std)
        self.logit_std_input = DoubleSpinBox(self.moss_training_box)
        self.logit_std_input.setDecimals(3)
        self.logit_std_input.setSingleStep(0.01)
        self.logit_std_input.setValue(0.95)
        self.horizontalLayout_moss_dist.addWidget(self.logit_std_input)
        self.formLayout_3.setLayout(6, QFormLayout.SpanningRole, self.horizontalLayout_moss_dist)

        self.horizontalLayout_moss_dist2 = QHBoxLayout()
        self.label_sigma_cutoff = QLabel(self.moss_training_box)
        self.horizontalLayout_moss_dist2.addWidget(self.label_sigma_cutoff)
        self.sigma_cutoff_input = DoubleSpinBox(self.moss_training_box)
        self.sigma_cutoff_input.setDecimals(3)
        self.sigma_cutoff_input.setSingleStep(0.01)
        self.sigma_cutoff_input.setValue(0.95)
        self.horizontalLayout_moss_dist2.addWidget(self.sigma_cutoff_input)

        self.label_uniform_mix = QLabel(self.moss_training_box)
        self.horizontalLayout_moss_dist2.addWidget(self.label_uniform_mix)
        self.uniform_mix_input = DoubleSpinBox(self.moss_training_box)
        self.uniform_mix_input.setDecimals(3)
        self.uniform_mix_input.setSingleStep(0.01)
        self.uniform_mix_input.setValue(0.01)
        self.horizontalLayout_moss_dist2.addWidget(self.uniform_mix_input)
        self.formLayout_3.setLayout(7, QFormLayout.SpanningRole, self.horizontalLayout_moss_dist2)

        # Inference / sampling-time options (used for the periodic sample images, matches official sampler)
        self.horizontalLayout_sample = QHBoxLayout()
        self.label_sample_shift = QLabel(self.moss_training_box)
        self.horizontalLayout_sample.addWidget(self.label_sample_shift)
        self.sample_shift_input = DoubleSpinBox(self.moss_training_box)
        self.sample_shift_input.setDecimals(2)
        self.sample_shift_input.setSingleStep(0.1)
        self.sample_shift_input.setValue(5.0)
        self.horizontalLayout_sample.addWidget(self.sample_shift_input)

        self.label_uncond_skip = QLabel(self.moss_training_box)
        self.horizontalLayout_sample.addWidget(self.label_uncond_skip)
        self.sample_uncond_skip_layers_input = QLineEdit(self.moss_training_box)
        self.sample_uncond_skip_layers_input.setObjectName(u"sample_uncond_skip_layers_input")
        self.horizontalLayout_sample.addWidget(self.sample_uncond_skip_layers_input)
        self.formLayout_3.setLayout(8, QFormLayout.SpanningRole, self.horizontalLayout_sample)

        # LoRA exclude blocks (network-affecting, do not resume a differently-excluded state)
        self.label_exclude_blocks = QLabel(self.moss_training_box)
        self.formLayout_3.setWidget(9, QFormLayout.LabelRole, self.label_exclude_blocks)
        self.lora_exclude_blocks_input = QLineEdit(self.moss_training_box)
        self.lora_exclude_blocks_input.setObjectName(u"lora_exclude_blocks_input")
        self.formLayout_3.setWidget(9, QFormLayout.FieldRole, self.lora_exclude_blocks_input)

        # Misc checkboxes, row 1
        self.horizontalLayout_misc = QHBoxLayout()
        self.flash_attn_enable = QCheckBox(self.moss_training_box)
        self.flash_attn_enable.setObjectName(u"flash_attn_enable")
        self.horizontalLayout_misc.addWidget(self.flash_attn_enable)
        self.split_attn_enable = QCheckBox(self.moss_training_box)
        self.split_attn_enable.setObjectName(u"split_attn_enable")
        self.horizontalLayout_misc.addWidget(self.split_attn_enable)
        self.use_ramtorch_enable = QCheckBox(self.moss_training_box)
        self.use_ramtorch_enable.setObjectName(u"use_ramtorch_enable")
        self.horizontalLayout_misc.addWidget(self.use_ramtorch_enable)
        self.unsloth_offload_checkpointing = QCheckBox(self.moss_training_box)
        self.unsloth_offload_checkpointing.setObjectName(u"unsloth_offload_checkpointing")
        self.horizontalLayout_misc.addWidget(self.unsloth_offload_checkpointing)
        self.formLayout_3.setLayout(10, QFormLayout.SpanningRole, self.horizontalLayout_misc)

        # Misc checkboxes, row 2 (ablation / debug flags)
        self.horizontalLayout_misc2 = QHBoxLayout()
        self.no_compile_enable = QCheckBox(self.moss_training_box)
        self.no_compile_enable.setObjectName(u"no_compile_enable")
        self.horizontalLayout_misc2.addWidget(self.no_compile_enable)
        self.stock_lycoris_enable = QCheckBox(self.moss_training_box)
        self.stock_lycoris_enable.setObjectName(u"stock_lycoris_enable")
        self.horizontalLayout_misc2.addWidget(self.stock_lycoris_enable)
        self.no_sample_cache_enable = QCheckBox(self.moss_training_box)
        self.no_sample_cache_enable.setObjectName(u"no_sample_cache_enable")
        self.horizontalLayout_misc2.addWidget(self.no_sample_cache_enable)
        self.formLayout_3.setLayout(11, QFormLayout.SpanningRole, self.horizontalLayout_misc2)

        self.gridLayout_2.addLayout(self.formLayout_3, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.moss_training_box, 0, 0, 1, 1)

        self.retranslateUi(moss_ui)

    def retranslateUi(self, moss_ui):
        moss_ui.setWindowTitle(QCoreApplication.translate("moss_ui", u"Form", None))
        self.moss_training_box.setTitle(QCoreApplication.translate("moss_ui", u"Train MOSS-Image", None))
        self.label.setText(QCoreApplication.translate("moss_ui", u"DiT Model", None))
        self.label_2.setText(QCoreApplication.translate("moss_ui", u"Text Encoder (Qwen3.5-9B)", None))
        self.label_3.setText(QCoreApplication.translate("moss_ui", u"VAE Model", None))
        self.label_tok.setText(QCoreApplication.translate("moss_ui", u"Tokenizer Dir", None))
        self.max_token_label.setText(QCoreApplication.translate("moss_ui", u"Max Tokens", None))
        self.label_blocks_to_swap.setText(QCoreApplication.translate("moss_ui", u"Blocks to Swap", None))
        self.label_sampling.setText(QCoreApplication.translate("moss_ui", u"Timestep Sampling", None))
        self.label_shift.setText(QCoreApplication.translate("moss_ui", u"Discrete Flow Shift", None))
        self.label_sigmoid.setText(QCoreApplication.translate("moss_ui", u"Sigmoid Scale", None))
        self.label_logit_mean.setText(QCoreApplication.translate("moss_ui", u"Logit Mean", None))
        self.label_logit_std.setText(QCoreApplication.translate("moss_ui", u"Logit Std", None))
        self.label_sigma_cutoff.setText(QCoreApplication.translate("moss_ui", u"Sigma Cutoff", None))
        self.label_uniform_mix.setText(QCoreApplication.translate("moss_ui", u"Uniform Mix", None))
        self.label_sample_shift.setText(QCoreApplication.translate("moss_ui", u"Sample Shift", None))
        self.label_uncond_skip.setText(QCoreApplication.translate("moss_ui", u"Uncond Skip Layers", None))
        self.label_exclude_blocks.setText(QCoreApplication.translate("moss_ui", u"LoRA Exclude Blocks", None))
        self.flash_attn_enable.setText(QCoreApplication.translate("moss_ui", u"Flash Attention", None))
        self.split_attn_enable.setText(QCoreApplication.translate("moss_ui", u"Split Attention", None))
        self.use_ramtorch_enable.setText(QCoreApplication.translate("moss_ui", u"Use RamTorch", None))
        self.unsloth_offload_checkpointing.setText(QCoreApplication.translate("moss_ui", u"Unsloth Offload", None))
        self.no_compile_enable.setText(QCoreApplication.translate("moss_ui", u"No Compile", None))
        self.stock_lycoris_enable.setText(QCoreApplication.translate("moss_ui", u"Stock LyCORIS", None))
        self.no_sample_cache_enable.setText(QCoreApplication.translate("moss_ui", u"No Sample Cache", None))

        # Tooltips / placeholders
        self.dit_model_input.setPlaceholderText(QCoreApplication.translate("moss_ui",
            u"Path to MOSS-Image-Base transformer (.safetensors) [Required]", None))
        self.dit_model_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>MOSS-Image-Base transformer, one file. Train on Base (the official guide leaves "
            u"Base without post-training for LoRAs); the LoRA also loads on SFT. "
            u"Maps to <code>--pretrained_model_name_or_path</code>.</p></body></html>", None))

        self.moss_text_encoder_input.setPlaceholderText(QCoreApplication.translate("moss_ui",
            u"Path to Qwen3.5-9B text encoder (.safetensors) [Required]", None))
        self.moss_text_encoder_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Qwen3.5-9B text encoder, shared by Base and SFT. Only loaded while caching text "
            u"encoder outputs, then released. Maps to <code>--moss_text_encoder</code>.</p></body></html>", None))

        self.vae_model_input.setPlaceholderText(QCoreApplication.translate("moss_ui",
            u"Path to FLUX.2 small-decoder VAE (.safetensors) [Required]", None))
        self.vae_model_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>FLUX.2 small-decoder VAE, kept fp32. Maps to <code>--vae</code>.</p></body></html>", None))

        self.tokenizer_dir_input.setPlaceholderText(QCoreApplication.translate("moss_ui",
            u"Folder with the Qwen3.5 tokenizer.json [Required]", None))
        self.tokenizer_dir_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Folder containing the Qwen3.5 tokenizer's <code>tokenizer.json</code> "
            u"(e.g. MOSS-Image/tokenizer). No machine-specific default is bundled, training will fail without "
            u"this. Maps to <code>--moss_tokenizer_dir</code>.</p></body></html>", None))

        self.max_token_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Chat template + caption tokens, official limit. Default: 4096. "
            u"Maps to <code>--moss_max_token_length</code>.</p></body></html>", None))
        self.blocks_to_swap_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Number of transformer blocks to swap CPU/GPU to reduce VRAM, at the cost of speed. "
            u"0 disables. Maps to <code>--blocks_to_swap</code>.</p></body></html>", None))

        self.timestep_sampling_selector.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p><b>moss</b>: Base's last pretraining distribution (logit(sigma) ~ N(mean, std), "
            u"truncated by Sigma Cutoff, mixed with Uniform Mix). Recommended default."
            u"<br/><b>sigma / shift</b>: use Discrete Flow Shift."
            u"<br/><b>sigmoid / shift</b>: use Sigmoid Scale."
            u"<br/><b>uniform</b>: flat random, no extra parameters.</p></body></html>", None))
        self.discrete_flow_shift_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Used by <code>sigma</code> and <code>shift</code> timestep sampling. Default: 1.0."
            u"</p></body></html>", None))
        self.sigmoid_scale_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Used by <code>sigmoid</code> and <code>shift</code> timestep sampling. "
            u"Default: 1.0.</p></body></html>", None))
        self.logit_mean_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Only used by <code>timestep_sampling = moss</code>: mean of logit(sigma). "
            u"Default: 1.09 (median sigma 0.78).</p></body></html>", None))
        self.logit_std_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Only used by <code>timestep_sampling = moss</code>: std of logit(sigma). "
            u"Default: 0.95.</p></body></html>", None))
        self.sigma_cutoff_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Only used by <code>timestep_sampling = moss</code>: truncation before rescaling. "
            u"Default: 0.95.</p></body></html>", None))
        self.uniform_mix_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Only used by <code>timestep_sampling = moss</code>: weight of the uniform mixture. "
            u"Default: 0.01.</p></body></html>", None))

        self.sample_shift_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Shift used for sample images during training (official sampler default: 5.0, "
            u"Euler, CFG). Maps to <code>--moss_sample_shift</code>.</p></body></html>", None))
        self.sample_uncond_skip_layers_input.setPlaceholderText(QCoreApplication.translate("moss_ui",
            u"e.g. 22", None))
        self.sample_uncond_skip_layers_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Comma-separated block indices the negative/uncond branch skips at sample-image "
            u"inference time (official default: block 22). Maps to "
            u"<code>--moss_sample_uncond_skip_layers</code>.</p></body></html>", None))

        self.lora_exclude_blocks_input.setPlaceholderText(QCoreApplication.translate("moss_ui",
            u"e.g. 22 (leave empty to train all blocks)", None))
        self.lora_exclude_blocks_input.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Leave these blocks out of the LoRA network entirely (e.g. block 22, which the "
            u"negative branch skips at inference, so a LoRA there acts on cond only and is amplified by CFG). "
            u"<b>This changes the network</b> — do not resume a state trained with a different value. "
            u"Maps to <code>--moss_lora_exclude_blocks</code>.</p></body></html>", None))

        self.flash_attn_enable.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Flash-attn varlen attention. Needs the flash-attn package and an Ampere+ GPU. "
            u"Maps to <code>--attn_mode flash</code>.</p></body></html>", None))
        self.split_attn_enable.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Per-sample attention: slower, lowest memory. Maps to "
            u"<code>--split_attn</code>.</p></body></html>", None))
        self.use_ramtorch_enable.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Keeps the joint-block Linears (the LoRA targets) pinned in CPU RAM and streams "
            u"them in, trading ~9 GB of system RAM for a large VRAM reduction at the same speed and losses. "
            u"Maps to <code>--use_ramtorch</code>.</p></body></html>", None))
        self.unsloth_offload_checkpointing.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Offload checkpointed activations to CPU RAM (async). Off is ~1% faster but uses "
            u"more VRAM. Maps to <code>--unsloth_offload_checkpointing</code>.</p></body></html>", None))
        self.no_compile_enable.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Disable elementwise op compilation. Debug/compatibility flag; leave off for normal "
            u"training. Maps to <code>--moss_no_compile</code>.</p></body></html>", None))
        self.stock_lycoris_enable.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Use stock LyCORIS LoKr forward instead of this fork's faster patched version. "
            u"Maps to <code>--moss_stock_lycoris</code>.</p></body></html>", None))
        self.no_sample_cache_enable.setToolTip(QCoreApplication.translate("moss_ui",
            u"<html><body><p>Disable on-disk caching of sample-prompt text-encoder outputs. Maps to "
            u"<code>--moss_no_sample_cache</code>.</p></body></html>", None))
