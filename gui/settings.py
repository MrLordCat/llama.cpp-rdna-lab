"""
Settings management for llama.cpp GUI

Provides utilities for:
- Loading saved settings from QSettings
- Saving current settings
- Handling application close events
"""

import os
from PyQt6.QtCore import QSettings


class SettingsManager:
    """Manages GUI settings persistence"""
    
    def __init__(self, gui_instance):
        """Initialize SettingsManager
        
        Args:
            gui_instance: Reference to LlamaCppGUI instance
        """
        self.gui = gui_instance
        self.settings = gui_instance.settings if hasattr(gui_instance, 'settings') else QSettings("LlamaCpp", "GUI")
    
    def load_settings(self):
        """Load saved settings"""
        # Load server settings first - they are always available
        if hasattr(self.gui, 'server_port_spin'):
            self.gui.server_port_spin.setValue(self.settings.value("server_port", 8080, type=int))
        if hasattr(self.gui, 'server_gpu_checkbox'):
            self.gui.server_gpu_checkbox.setChecked(self.settings.value("server_gpu", True, type=bool))
        if hasattr(self.gui, 'server_cors_checkbox'):
            self.gui.server_cors_checkbox.setChecked(self.settings.value("server_cors", True, type=bool))
        if hasattr(self.gui, 'server_api_key_edit'):
            self.gui.server_api_key_edit.setText(self.settings.value("server_api_key", ""))
        if hasattr(self.gui, 'server_backend_combo'):
            self.gui.server_backend_combo.setCurrentIndex(self.settings.value("server_backend", 0, type=int))
        if hasattr(self.gui, 'server_model_path_edit'):
            self.gui.server_model_path_edit.setText(self.settings.value("server_model_path", ""))
        
        # Load server parameters (sliders)
        if hasattr(self.gui, 'server_ctx_slider'):
            ctx_val = self.settings.value("server_ctx", 8192, type=int)
            self.gui.server_ctx_slider.setValue(max(1, ctx_val // 8192))  # Convert to step
        if hasattr(self.gui, 'server_batch_slider'):
            batch_val = self.settings.value("server_batch", 512, type=int)
            self.gui.server_batch_slider.setValue(max(1, batch_val // 32))  # Convert to step
        if hasattr(self.gui, 'server_ubatch_slider'):
            ubatch_val = self.settings.value("server_ubatch", 512, type=int)
            self.gui.server_ubatch_slider.setValue(max(1, ubatch_val // 32))  # Convert to step
        if hasattr(self.gui, 'server_threads_spin'):
            self.gui.server_threads_spin.setValue(self.settings.value("server_threads", os.cpu_count() or 4, type=int))
        if hasattr(self.gui, 'server_http_threads_spin'):
            self.gui.server_http_threads_spin.setValue(self.settings.value("server_http_threads", max(1, (os.cpu_count() or 4) // 2), type=int))
        if hasattr(self.gui, 'server_gpu_layers_slider'):
            self.gui.server_gpu_layers_slider.setValue(self.settings.value("server_gpu_layers", 33, type=int))
        if hasattr(self.gui, 'server_parallel_spin'):
            self.gui.server_parallel_spin.setValue(self.settings.value("server_parallel", 1, type=int))
        if hasattr(self.gui, 'server_flash_attn_checkbox'):
            self.gui.server_flash_attn_checkbox.setChecked(self.settings.value("server_flash_attn", False, type=bool))
        if hasattr(self.gui, 'server_no_mmap_checkbox'):
            self.gui.server_no_mmap_checkbox.setChecked(self.settings.value("server_no_mmap", False, type=bool))
        if hasattr(self.gui, 'server_kv_cache_combo'):
            self.gui.server_kv_cache_combo.setCurrentIndex(self.settings.value("server_kv_cache", 0, type=int))
        if hasattr(self.gui, 'server_no_warmup_checkbox'):
            self.gui.server_no_warmup_checkbox.setChecked(self.settings.value("server_no_warmup", os.name != 'nt', type=bool))
        if hasattr(self.gui, 'server_spec_type_combo'):
            self.gui.server_spec_type_combo.setCurrentIndex(self.settings.value("server_spec_type", 0, type=int))
        if hasattr(self.gui, 'server_spec_draft_n_max_spin'):
            self.gui.server_spec_draft_n_max_spin.setValue(self.settings.value("server_spec_draft_n_max", 8, type=int))
        if hasattr(self.gui, 'server_spec_ngram_match_spin'):
            self.gui.server_spec_ngram_match_spin.setValue(self.settings.value("server_spec_ngram_match", 16, type=int))
        if hasattr(self.gui, 'server_spec_ngram_n_min_spin'):
            self.gui.server_spec_ngram_n_min_spin.setValue(self.settings.value("server_spec_ngram_n_min", 12, type=int))
        if hasattr(self.gui, 'server_spec_ngram_n_max_spin'):
            self.gui.server_spec_ngram_n_max_spin.setValue(self.settings.value("server_spec_ngram_n_max", 32, type=int))
        if hasattr(self.gui, 'server_vision_checkbox'):
            self.gui.server_vision_checkbox.setChecked(self.settings.value("server_vision", False, type=bool))
        if hasattr(self.gui, 'server_mmproj_path_edit'):
            self.gui.server_mmproj_path_edit.setText(self.settings.value("server_mmproj_path", ""))
        if hasattr(self.gui, 'server_image_max_tokens_spin'):
            self.gui.server_image_max_tokens_spin.setValue(self.settings.value("server_image_max_tokens", 1024, type=int))
        if hasattr(self.gui, 'server_mmproj_offload_checkbox'):
            self.gui.server_mmproj_offload_checkbox.setChecked(self.settings.value("server_mmproj_offload", True, type=bool))
        if hasattr(self.gui, 'server_spec_type_combo'):
            self.gui._set_server_speculative_controls_enabled(self.gui.server_spec_type_combo.currentIndex())
        if hasattr(self.gui, 'server_model_path_edit') and hasattr(self.gui, 'server_vision_checkbox'):
            self.gui.refresh_server_vision_controls(clear_existing=False)
        
        # Sampling defaults
        if hasattr(self.gui, 'server_temp_spin'):
            self.gui.server_temp_spin.setValue(self.settings.value("server_temp", 0.8, type=float))
        if hasattr(self.gui, 'server_top_p_spin'):
            self.gui.server_top_p_spin.setValue(self.settings.value("server_top_p", 0.95, type=float))
        if hasattr(self.gui, 'server_top_k_spin'):
            self.gui.server_top_k_spin.setValue(self.settings.value("server_top_k", 40, type=int))
        if hasattr(self.gui, 'server_min_p_spin'):
            self.gui.server_min_p_spin.setValue(self.settings.value("server_min_p", 0.05, type=float))
        if hasattr(self.gui, 'server_presence_penalty_spin'):
            self.gui.server_presence_penalty_spin.setValue(self.settings.value("server_presence_penalty", 0.0, type=float))
        if hasattr(self.gui, 'server_repeat_penalty_spin'):
            self.gui.server_repeat_penalty_spin.setValue(self.settings.value("server_repeat_penalty", 1.0, type=float))
        
        # Load Quick Select model selection
        saved_model_name = self.settings.value("server_quick_select_model", "")
        if saved_model_name and hasattr(self.gui, 'server_models_list'):
            index = self.gui.server_models_list.findText(saved_model_name)
            if index >= 0:
                self.gui.server_models_list.setCurrentIndex(index)
        
        # Load inference settings
        if hasattr(self.gui, 'n_predict_spin'):
            self.gui.n_predict_spin.setValue(self.settings.value("n_predict", 128, type=int))
        if hasattr(self.gui, 'temp_spin'):
            self.gui.temp_spin.setValue(self.settings.value("temperature", 0.8, type=float))
        if hasattr(self.gui, 'top_p_spin'):
            self.gui.top_p_spin.setValue(self.settings.value("top_p", 0.9, type=float))
        if hasattr(self.gui, 'top_k_spin'):
            self.gui.top_k_spin.setValue(self.settings.value("top_k", 40, type=int))
        if hasattr(self.gui, 'ctx_size_spin'):
            self.gui.ctx_size_spin.setValue(self.settings.value("ctx_size", 2048, type=int))
        if hasattr(self.gui, 'threads_spin'):
            self.gui.threads_spin.setValue(self.settings.value("threads", os.cpu_count() or 4, type=int))
        if hasattr(self.gui, 'gpu_layers_spin'):
            self.gui.gpu_layers_spin.setValue(self.settings.value("gpu_layers", 33, type=int))
        
        last_model = self.settings.value("last_model", "")
        if last_model and hasattr(self.gui, 'model_path_edit'):
            self.gui.model_path_edit.setText(last_model)
    
    def save_settings(self):
        """Save settings"""
        # Save server settings first
        if hasattr(self.gui, 'server_port_spin'):
            self.settings.setValue("server_port", self.gui.server_port_spin.value())
        if hasattr(self.gui, 'server_gpu_checkbox'):
            self.settings.setValue("server_gpu", self.gui.server_gpu_checkbox.isChecked())
        if hasattr(self.gui, 'server_cors_checkbox'):
            self.settings.setValue("server_cors", self.gui.server_cors_checkbox.isChecked())
        if hasattr(self.gui, 'server_api_key_edit'):
            self.settings.setValue("server_api_key", self.gui.server_api_key_edit.text())
        if hasattr(self.gui, 'server_backend_combo'):
            self.settings.setValue("server_backend", self.gui.server_backend_combo.currentIndex())
        if hasattr(self.gui, 'server_model_path_edit'):
            self.settings.setValue("server_model_path", self.gui.server_model_path_edit.text())
        if hasattr(self.gui, 'server_threads_spin'):
            self.settings.setValue("server_threads", self.gui.server_threads_spin.value())
        if hasattr(self.gui, 'server_http_threads_spin'):
            self.settings.setValue("server_http_threads", self.gui.server_http_threads_spin.value())
        if hasattr(self.gui, 'server_ctx_slider'):
            self.settings.setValue("server_ctx", self.gui.server_ctx_slider.value() * 8192)  # Save actual value
        if hasattr(self.gui, 'server_batch_slider'):
            self.settings.setValue("server_batch", self.gui.server_batch_slider.value() * 32)  # Save actual value
        if hasattr(self.gui, 'server_ubatch_slider'):
            self.settings.setValue("server_ubatch", self.gui.server_ubatch_slider.value() * 32)  # Save actual value
        if hasattr(self.gui, 'server_gpu_layers_slider'):
            self.settings.setValue("server_gpu_layers", self.gui.server_gpu_layers_slider.value())
        if hasattr(self.gui, 'server_parallel_spin'):
            self.settings.setValue("server_parallel", self.gui.server_parallel_spin.value())
        if hasattr(self.gui, 'server_flash_attn_checkbox'):
            self.settings.setValue("server_flash_attn", self.gui.server_flash_attn_checkbox.isChecked())
        if hasattr(self.gui, 'server_no_mmap_checkbox'):
            self.settings.setValue("server_no_mmap", self.gui.server_no_mmap_checkbox.isChecked())
        if hasattr(self.gui, 'server_kv_cache_combo'):
            self.settings.setValue("server_kv_cache", self.gui.server_kv_cache_combo.currentIndex())
        if hasattr(self.gui, 'server_no_warmup_checkbox'):
            self.settings.setValue("server_no_warmup", self.gui.server_no_warmup_checkbox.isChecked())
        if hasattr(self.gui, 'server_spec_type_combo'):
            self.settings.setValue("server_spec_type", self.gui.server_spec_type_combo.currentIndex())
        if hasattr(self.gui, 'server_spec_draft_n_max_spin'):
            self.settings.setValue("server_spec_draft_n_max", self.gui.server_spec_draft_n_max_spin.value())
        if hasattr(self.gui, 'server_spec_ngram_match_spin'):
            self.settings.setValue("server_spec_ngram_match", self.gui.server_spec_ngram_match_spin.value())
        if hasattr(self.gui, 'server_spec_ngram_n_min_spin'):
            self.settings.setValue("server_spec_ngram_n_min", self.gui.server_spec_ngram_n_min_spin.value())
        if hasattr(self.gui, 'server_spec_ngram_n_max_spin'):
            self.settings.setValue("server_spec_ngram_n_max", self.gui.server_spec_ngram_n_max_spin.value())
        if hasattr(self.gui, 'server_vision_checkbox'):
            self.settings.setValue("server_vision", self.gui.server_vision_checkbox.isChecked())
        if hasattr(self.gui, 'server_mmproj_path_edit'):
            self.settings.setValue("server_mmproj_path", self.gui.server_mmproj_path_edit.text())
        if hasattr(self.gui, 'server_image_max_tokens_spin'):
            self.settings.setValue("server_image_max_tokens", self.gui.server_image_max_tokens_spin.value())
        if hasattr(self.gui, 'server_mmproj_offload_checkbox'):
            self.settings.setValue("server_mmproj_offload", self.gui.server_mmproj_offload_checkbox.isChecked())
        
        # Save sampling defaults
        if hasattr(self.gui, 'server_temp_spin'):
            self.settings.setValue("server_temp", self.gui.server_temp_spin.value())
        if hasattr(self.gui, 'server_top_p_spin'):
            self.settings.setValue("server_top_p", self.gui.server_top_p_spin.value())
        if hasattr(self.gui, 'server_top_k_spin'):
            self.settings.setValue("server_top_k", self.gui.server_top_k_spin.value())
        if hasattr(self.gui, 'server_min_p_spin'):
            self.settings.setValue("server_min_p", self.gui.server_min_p_spin.value())
        if hasattr(self.gui, 'server_presence_penalty_spin'):
            self.settings.setValue("server_presence_penalty", self.gui.server_presence_penalty_spin.value())
        if hasattr(self.gui, 'server_repeat_penalty_spin'):
            self.settings.setValue("server_repeat_penalty", self.gui.server_repeat_penalty_spin.value())
        
        # Save Quick Select model selection
        if hasattr(self.gui, 'server_models_list'):
            current_model = self.gui.server_models_list.currentText()
            if current_model and current_model != "-- Select Model --":
                self.settings.setValue("server_quick_select_model", current_model)
            
        # Save inference settings
        if hasattr(self.gui, 'n_predict_spin'):
            self.settings.setValue("n_predict", self.gui.n_predict_spin.value())
        if hasattr(self.gui, 'temp_spin'):
            self.settings.setValue("temperature", self.gui.temp_spin.value())
        if hasattr(self.gui, 'top_p_spin'):
            self.settings.setValue("top_p", self.gui.top_p_spin.value())
        if hasattr(self.gui, 'top_k_spin'):
            self.settings.setValue("top_k", self.gui.top_k_spin.value())
        if hasattr(self.gui, 'ctx_size_spin'):
            self.settings.setValue("ctx_size", self.gui.ctx_size_spin.value())
        if hasattr(self.gui, 'threads_spin'):
            self.settings.setValue("threads", self.gui.threads_spin.value())
        if hasattr(self.gui, 'gpu_layers_spin'):
            self.settings.setValue("gpu_layers", self.gui.gpu_layers_spin.value())
        if hasattr(self.gui, 'model_path_edit'):
            self.settings.setValue("last_model", self.gui.model_path_edit.text())
