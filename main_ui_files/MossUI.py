from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QPushButton
from modules.DragDropLineEdit import DragDropLineEdit
from ui_files.MossUI import Ui_moss_ui
from modules.BaseWidget import BaseWidget
from pathlib import Path


class MossWidget(BaseWidget):
    Toggled = Signal(bool)  # send to general args

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("MOSS-Image Args")
        self.widget = Ui_moss_ui()

        self.name = "moss_args"

        self.setup_widget()
        self.setup_connections()

    def setup_widget(self) -> None:
        super().setup_widget()
        self.widget.setupUi(self.content)

        def setup_file(elem: DragDropLineEdit, selector: QPushButton, folder=False):
            selector_icon = QIcon(str(Path("icons/more-horizontal.svg")))
            if folder:
                elem.setMode("folder")
            else:
                elem.setMode("file", [".ckpt", ".pt", ".safetensors", ".sft", ".pth"])
            elem.highlight = True
            selector.setIcon(selector_icon)

        setup_file(self.widget.dit_model_input, self.widget.dit_model_selector)
        setup_file(self.widget.moss_text_encoder_input, self.widget.moss_text_encoder_selector)
        setup_file(self.widget.vae_model_input, self.widget.vae_model_selector)
        setup_file(self.widget.tokenizer_dir_input, self.widget.tokenizer_dir_selector, folder=True)

    def setup_connections(self) -> None:
        self.widget.moss_training_box.clicked.connect(self.enable_disable)

        # File inputs — keys match sd_scripts argparse names
        self.widget.dit_model_input.textChanged.connect(lambda x: self.edit_args("pretrained_model_name_or_path", x))
        self.widget.dit_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.dit_model_input, "MOSS DiT Model", "Model File")
        )

        self.widget.moss_text_encoder_input.textChanged.connect(lambda x: self.edit_args("moss_text_encoder", x))
        self.widget.moss_text_encoder_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.moss_text_encoder_input, "MOSS Text Encoder", "Model File")
        )

        self.widget.vae_model_input.textChanged.connect(lambda x: self.edit_args("vae", x))
        self.widget.vae_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.vae_model_input, "VAE Model", "Model File")
        )

        self.widget.tokenizer_dir_input.textChanged.connect(lambda x: self.edit_args("moss_tokenizer_dir", x))
        self.widget.tokenizer_dir_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(self.widget.tokenizer_dir_input, "MOSS Tokenizer Folder")
        )

        # Token length / block swap
        self.widget.max_token_input.valueChanged.connect(
            lambda x: self.edit_args("moss_max_token_length", x)
        )
        self.widget.blocks_to_swap_input.valueChanged.connect(self.change_blocks_to_swap)

        # Sampling params
        self.widget.timestep_sampling_selector.currentIndexChanged.connect(self.change_timestep_sampling_type)
        self.widget.discrete_flow_shift_input.valueChanged.connect(
            lambda x: self.edit_args("discrete_flow_shift", x)
        )
        self.widget.sigmoid_scale_input.valueChanged.connect(
            lambda x: self.edit_args("sigmoid_scale", x)
        )
        self.widget.logit_mean_input.valueChanged.connect(
            lambda x: self.edit_args("moss_logit_mean", x)
        )
        self.widget.logit_std_input.valueChanged.connect(
            lambda x: self.edit_args("moss_logit_std", x)
        )
        self.widget.sigma_cutoff_input.valueChanged.connect(
            lambda x: self.edit_args("moss_sigma_cutoff", x)
        )
        self.widget.uniform_mix_input.valueChanged.connect(
            lambda x: self.edit_args("moss_uniform_mix", x)
        )

        # Sample-image inference params
        self.widget.sample_shift_input.valueChanged.connect(
            lambda x: self.edit_args("moss_sample_shift", x)
        )
        self.widget.sample_uncond_skip_layers_input.textChanged.connect(
            lambda x: self.edit_args("moss_sample_uncond_skip_layers", x, optional=True)
        )

        # Network-affecting
        self.widget.lora_exclude_blocks_input.textChanged.connect(
            lambda x: self.edit_args("moss_lora_exclude_blocks", x, optional=True)
        )

        # Misc
        self.widget.flash_attn_enable.clicked.connect(self.change_flash_attn)
        self.widget.split_attn_enable.clicked.connect(
            lambda x: self.edit_args("split_attn", x, True)
        )
        self.widget.use_ramtorch_enable.clicked.connect(
            lambda x: self.edit_args("use_ramtorch", x, True)
        )
        self.widget.unsloth_offload_checkpointing.clicked.connect(
            lambda x: self.edit_args("unsloth_offload_checkpointing", x, True)
        )
        self.widget.no_compile_enable.clicked.connect(
            lambda x: self.edit_args("moss_no_compile", x, True)
        )
        self.widget.stock_lycoris_enable.clicked.connect(
            lambda x: self.edit_args("moss_stock_lycoris", x, True)
        )
        self.widget.no_sample_cache_enable.clicked.connect(
            lambda x: self.edit_args("moss_no_sample_cache", x, True)
        )

    def enable_disable(self, checked: bool) -> None:
        self.args = {}
        self.Toggled.emit(checked)
        if not checked:
            return

        self.edit_args("pretrained_model_name_or_path", self.widget.dit_model_input.text())
        self.edit_args("moss_text_encoder", self.widget.moss_text_encoder_input.text())
        self.edit_args("vae", self.widget.vae_model_input.text())
        self.edit_args("moss_tokenizer_dir", self.widget.tokenizer_dir_input.text())

        self.edit_args("moss_max_token_length", self.widget.max_token_input.value())
        self.change_blocks_to_swap(self.widget.blocks_to_swap_input.value())

        self.change_timestep_sampling_type(self.widget.timestep_sampling_selector.currentIndex())

        self.edit_args("moss_sample_shift", self.widget.sample_shift_input.value())
        self.edit_args(
            "moss_sample_uncond_skip_layers", self.widget.sample_uncond_skip_layers_input.text(), optional=True
        )
        self.edit_args(
            "moss_lora_exclude_blocks", self.widget.lora_exclude_blocks_input.text(), optional=True
        )

        self.change_flash_attn(self.widget.flash_attn_enable.isChecked())
        self.edit_args("split_attn", self.widget.split_attn_enable.isChecked(), True)
        self.edit_args("use_ramtorch", self.widget.use_ramtorch_enable.isChecked(), True)
        self.edit_args("unsloth_offload_checkpointing", self.widget.unsloth_offload_checkpointing.isChecked(), True)
        self.edit_args("moss_no_compile", self.widget.no_compile_enable.isChecked(), True)
        self.edit_args("moss_stock_lycoris", self.widget.stock_lycoris_enable.isChecked(), True)
        self.edit_args("moss_no_sample_cache", self.widget.no_sample_cache_enable.isChecked(), True)

    def external_enable_disable(self, checked: bool) -> None:
        self.args = {}
        self.widget.moss_training_box.setEnabled(not checked)
        if self.widget.moss_training_box.isEnabled() and self.widget.moss_training_box.isChecked():
            self.enable_disable(True)

    def change_timestep_sampling_type(self, index: int) -> None:
        sampling_type = self.widget.timestep_sampling_selector.currentText()
        self.edit_args("timestep_sampling", sampling_type)

        # sigmoid and shift use sigmoid_scale to control concentration
        uses_sigmoid_scale = sampling_type in ("sigmoid", "shift")
        self.widget.sigmoid_scale_input.setEnabled(uses_sigmoid_scale)
        if uses_sigmoid_scale:
            self.edit_args("sigmoid_scale", self.widget.sigmoid_scale_input.value())
        elif "sigmoid_scale" in self.args:
            del self.args["sigmoid_scale"]

        # sigma and shift use discrete_flow_shift
        uses_flow_shift = sampling_type in ("sigma", "shift")
        self.widget.discrete_flow_shift_input.setEnabled(uses_flow_shift)
        if uses_flow_shift:
            self.edit_args("discrete_flow_shift", self.widget.discrete_flow_shift_input.value())
        elif "discrete_flow_shift" in self.args:
            del self.args["discrete_flow_shift"]

        # moss's own logit-normal distribution params only apply to timestep_sampling == "moss"
        uses_moss_dist = sampling_type == "moss"
        for widget, key in (
            (self.widget.logit_mean_input, "moss_logit_mean"),
            (self.widget.logit_std_input, "moss_logit_std"),
            (self.widget.sigma_cutoff_input, "moss_sigma_cutoff"),
            (self.widget.uniform_mix_input, "moss_uniform_mix"),
        ):
            widget.setEnabled(uses_moss_dist)
            if uses_moss_dist:
                self.edit_args(key, widget.value())
            elif key in self.args:
                del self.args[key]

    def change_blocks_to_swap(self, value: int) -> None:
        """Blocks to swap: 0 means disabled (None), any positive value is used as-is."""
        if value > 0:
            self.edit_args("blocks_to_swap", value)
        elif "blocks_to_swap" in self.args:
            del self.args["blocks_to_swap"]

    def change_flash_attn(self, checked: bool) -> None:
        """Flash attention: sets attn_mode to 'flash' when enabled, removes it when disabled."""
        if checked:
            self.edit_args("attn_mode", "flash")
        elif "attn_mode" in self.args:
            del self.args["attn_mode"]

    def load_args(self, args: dict) -> bool:
        args: dict = args.get(self.name, {})

        self.widget.moss_training_box.setChecked(bool(args))
        self.widget.dit_model_input.setText(args.get("pretrained_model_name_or_path", ""))
        self.widget.moss_text_encoder_input.setText(args.get("moss_text_encoder", ""))
        self.widget.vae_model_input.setText(args.get("vae", ""))
        self.widget.tokenizer_dir_input.setText(args.get("moss_tokenizer_dir", ""))

        self.widget.max_token_input.setValue(args.get("moss_max_token_length", 4096))
        self.widget.blocks_to_swap_input.setValue(args.get("blocks_to_swap", 0))

        self.widget.timestep_sampling_selector.setCurrentText(args.get("timestep_sampling", "moss"))
        self.widget.discrete_flow_shift_input.setValue(args.get("discrete_flow_shift", 1.0))
        self.widget.sigmoid_scale_input.setValue(args.get("sigmoid_scale", 1.0))
        self.widget.logit_mean_input.setValue(args.get("moss_logit_mean", 1.09))
        self.widget.logit_std_input.setValue(args.get("moss_logit_std", 0.95))
        self.widget.sigma_cutoff_input.setValue(args.get("moss_sigma_cutoff", 0.95))
        self.widget.uniform_mix_input.setValue(args.get("moss_uniform_mix", 0.01))

        self.widget.sample_shift_input.setValue(args.get("moss_sample_shift", 5.0))
        self.widget.sample_uncond_skip_layers_input.setText(args.get("moss_sample_uncond_skip_layers", "22"))
        self.widget.lora_exclude_blocks_input.setText(args.get("moss_lora_exclude_blocks", ""))

        # flash_attn is stored as attn_mode="flash"
        self.widget.flash_attn_enable.setChecked(args.get("attn_mode", "") == "flash")
        self.widget.split_attn_enable.setChecked(args.get("split_attn", False))
        self.widget.use_ramtorch_enable.setChecked(args.get("use_ramtorch", False))
        self.widget.unsloth_offload_checkpointing.setChecked(args.get("unsloth_offload_checkpointing", False))
        self.widget.no_compile_enable.setChecked(args.get("moss_no_compile", False))
        self.widget.stock_lycoris_enable.setChecked(args.get("moss_stock_lycoris", False))
        self.widget.no_sample_cache_enable.setChecked(args.get("moss_no_sample_cache", False))

        self.enable_disable(self.widget.moss_training_box.isChecked())
