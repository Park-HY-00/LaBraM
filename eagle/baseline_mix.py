"""
Baseline Mixed Language EAGLE+LaBraM: Within-Subject, Session-Agnostic Mixed Language Training
Hybrid Model Only

For each subject: All English and Korean sessions combined (task1, task2, task4)
Data split: 7:1.5:1.5 = train:valid:test with balanced language sampling
Training includes both English and Korean data with equal proportions with 15-class semantic labels
Purpose: Establish baseline performance for mixed-language within-subject learning with the EAGLE+LaBraM hybrid
Model: EAGLE+LaBraM
"""

import sys
import os

# Get the absolute path to the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Parent directory is CrossLinguistic (where utils.py is located)
parent_dir = os.path.dirname(current_dir)
# Keep the local eagle directory first so `from utils import ...` resolves to eagle/utils.py.
# Append the repo root so root-level modules remain importable.
sys.path.append(parent_dir)

# Set CUDA_LAUNCH_BLOCKING for better error debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import copy
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import csv
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    EEGDataLoader, 
    SELECTED_TASKS, 
    STIMULUS_MAPPING, 
    EPOCH_TMIN, 
    EPOCH_TMAX,
    create_epochs_from_raw,
    load_subject_sessions_data
)

# Model import - EAGLE + LaBraM hybrid semantic decoder
from eagle_w_labram import EagleLaBraMSemanticDecoder
print("✅ Using EagleLaBraMSemanticDecoder from eagle_w_labram")
MODEL_AVAILABLE = True
import warnings
warnings.filterwarnings('ignore')

# Center Loss for representation space regularization
class CenterLoss(nn.Module):
    """Center Loss: intra-class variance reduction by clustering same-class features"""
    def __init__(self, num_classes, feat_dim, lambda_c=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        # Learnable class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feats, labels):
        # feats: (B, feat_dim), labels: (B,)
        centers_batch = self.centers[labels]      # (B, feat_dim) - select centers for each sample
        loss = ((feats - centers_batch) ** 2).sum(dim=1).mean()  # L2 distance to center
        return self.lambda_c * 0.5 * loss

class BaselineMixedEEGNet:
    def __init__(self, data_path="D:\\SemanticDecoding\\Dataset\\", loss_type='ce', label_smoothing=0.1):
        """
        Args:
            data_path: Path to dataset
            loss_type: Loss function type. Options:
                - 'ce': Standard Cross Entropy Loss
                - 'label_smoothing': Cross Entropy with Label Smoothing
                - 'scl_ce_hybrid': Supervised Contrastive Loss + Cross Entropy Hybrid (future)
            label_smoothing: Label smoothing value (0.0 to 1.0). Only used when loss_type='label_smoothing'
        """
        self.data_path = data_path
        self.loader = EEGDataLoader(data_path)
        # Use common parameters from utils
        self.selected_tasks = SELECTED_TASKS  # task1, task2, task4 only
        self.stimulus_mapping = STIMULUS_MAPPING
        self.epoch_tmin = EPOCH_TMIN
        self.epoch_tmax = EPOCH_TMAX
        
        # Loss function configuration
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        
        # EEG data configuration for sentence-level semantic decoding
        self.n_channels = 64
        self.n_classes = 15  # Semantic classes for tasks 1, 2, 4
        
        # Calculate expected samples from utils constants
        # ✅ Updated to match proposed.py: sfreq=500Hz, samples=1500 (3.0s)
        expected_samples = int((EPOCH_TMAX - EPOCH_TMIN) * 500)  # 1500 samples (3.0s at 500Hz)
        
        # Use the expected samples from utils for model creation
        # Data will be cropped/padded to match this size in _prepare_data_for_model
        self.input_window_samples = expected_samples  # Use utils epoch window for model
        self.sfreq = 500.0  # ✅ Updated from 250.0 to 500.0
        
        print(f"🔧 Expected samples from utils: {expected_samples}")
        print(f"🔧 Model will be created with: {self.input_window_samples} samples")
        print(f"🔧 Input window seconds: {self.input_window_samples / self.sfreq:.2f}s")
        print(f"🔧 Data will be automatically cropped/padded to match model input")
        
        # EAGLE + LaBraM configuration
        self.model_name = 'EagleLaBraMSemanticDecoder'
        self.labram_variant = 'base'
        self.labram_checkpoint = Path(parent_dir) / 'checkpoints' / 'labram-base.pth'
        self.freeze_labram = True
        self.unfreeze_last_n_blocks = 2
        self.semantic_dim = 256
        self.adapter_heads = 4
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Hybrid model needs a smaller batch size than the plain EAGLE baseline.
        self.batch_size = 16
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🖥️  GPU: {gpu_name}")
        else:
            print(f"💻 Using CPU mode")
        
        # Hybrid training configuration
        self.model_config = {
            'max_epochs': 100,
            'learning_rate': 3e-4,
            'weight_decay': 5e-4,
            'patience': 20,  # Reduced from 25
            'warmup_epochs': 8,
            'scheduler_factor': 0.5,
            'scheduler_patience': 10  # Reduced from 15
        }
        
        # Multi-band configuration for the hybrid model
        self.band_defs = ((8, 13), (13, 30), (30, 60), (60, 80))
        
        # Print loss configuration
        print(f"🔧 Loss Type: {self.loss_type}")
        if self.loss_type == 'label_smoothing':
            print(f"🔧 Label Smoothing: {self.label_smoothing}")
        
        # DataLoader workers - set to 0 to avoid Windows paging file issues
        # Windows multiprocessing spawns new processes that reload PyTorch/CUDA libs
        # This can cause "페이징 파일이 너무 작습니다" errors
        self.num_workers = 0
        
        # Results directory
        self.results_dir = Path(current_dir) / "temp_result"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        self.models_dir = self.results_dir / "saved_models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.training_history_dir = self.results_dir / "training_history"
        self.training_history_dir.mkdir(exist_ok=True)
        
        self.test_results_dir = self.results_dir / "test_results"
        self.test_results_dir.mkdir(exist_ok=True)
    
    def create_model(self):
        """Create the EAGLE + LaBraM hybrid model for bilingual semantic decoding"""
        if EagleLaBraMSemanticDecoder is None:
            raise ValueError("EagleLaBraMSemanticDecoder is not available")
        
        print(f"    ⚙️  Creating {self.model_name} with {self.input_window_samples} samples")
        print(f"    🔍 Configuration:")
        print(f"       - Channels: {self.n_channels}")
        print(f"       - Classes: {self.n_classes}")
        print(f"       - Samples: {self.input_window_samples}")
        print(f"       - Frequency bands: {self.band_defs}")
        print(f"       - LaBraM variant: {self.labram_variant}")
        print(f"       - Freeze LaBraM: {self.freeze_labram}")
        print(f"       - Unfreeze last N blocks: {self.unfreeze_last_n_blocks}")

        checkpoint_path = str(self.labram_checkpoint) if self.labram_checkpoint.exists() else None
        if checkpoint_path is None:
            print(f"    ⚠️  LaBraM checkpoint not found at {self.labram_checkpoint}; training will start without pretrained LaBraM weights")
        
        model = EagleLaBraMSemanticDecoder(
            nb_classes=self.n_classes,
            Chans=self.n_channels,
            Samples=self.input_window_samples,
            sfreq=self.sfreq,
            dropoutRate=0.3,
            F1=16,
            D=2,
            gamma_F1=32,
            reduced_dim=48,
            band_defs=self.band_defs,
            labram_variant=self.labram_variant,
            labram_checkpoint=checkpoint_path,
            freeze_labram=self.freeze_labram,
            unfreeze_last_n_blocks=self.unfreeze_last_n_blocks,
            semantic_dim=self.semantic_dim,
            adapter_heads=self.adapter_heads,
        )
        
        print(f"    ✅ {self.model_name} created successfully")
        print(f"    🎯 Multi-band architecture: {len(self.band_defs)} frequency branches")
        return model
    
    def load_subject_mixed_data(self, subject_id):
        """Load all English and Korean session data for a specific subject with balanced sampling"""
        print(f"  📂 Loading mixed language data for subject {subject_id}...")
        
        # Use the same approach as trackA.py
        all_epochs, session_info = load_subject_sessions_data(self.loader, subject_id, self.selected_tasks)
        
        # Extract English and Korean data
        eng_epochs = all_epochs.get('eng', [])
        kor_epochs = all_epochs.get('kor', [])
        eng_session_info = session_info.get('eng', [])
        kor_session_info = session_info.get('kor', [])
        
        # Combine epochs for each language
        eng_combined = mne.concatenate_epochs(eng_epochs) if eng_epochs else None
        kor_combined = mne.concatenate_epochs(kor_epochs) if kor_epochs else None
        
        if eng_combined is None and kor_combined is None:
            print(f"    ❌ No epochs found for subject {subject_id}")
            return None, None
        
        # Print epoch counts
        eng_count = len(eng_combined) if eng_combined else 0
        kor_count = len(kor_combined) if kor_combined else 0
        print(f"    📊 English epochs: {eng_count}")
        print(f"    📊 Korean epochs: {kor_count}")
        
        # Balance the data - use minimum count from both languages
        if eng_combined and kor_combined:
            min_epochs = min(len(eng_combined), len(kor_combined))
            print(f"    ⚖️  Balancing to {min_epochs} epochs per language")
            
            # Randomly sample to balance
            np.random.seed(42)
            eng_indices = np.random.choice(len(eng_combined), min_epochs, replace=False)
            kor_indices = np.random.choice(len(kor_combined), min_epochs, replace=False)
            
            eng_balanced = eng_combined[eng_indices]
            kor_balanced = kor_combined[kor_indices]
            
            # Add language metadata
            eng_balanced.metadata['language'] = 'eng'
            kor_balanced.metadata['language'] = 'kor'
            
            # Combine both languages
            combined_epochs = mne.concatenate_epochs([eng_balanced, kor_balanced])
            total_epochs = len(combined_epochs)
            
            print(f"    ✅ Total balanced epochs: {total_epochs} (English: {min_epochs}, Korean: {min_epochs})")
            
        elif eng_combined:
            print(f"    ⚠️  Only English data available")
            eng_combined.metadata['language'] = 'eng'
            combined_epochs = eng_combined
            total_epochs = len(combined_epochs)
            
        else:  # kor_combined only
            print(f"    ⚠️  Only Korean data available")
            kor_combined.metadata['language'] = 'kor'
            combined_epochs = kor_combined
            total_epochs = len(combined_epochs)
        
        # Combine session info
        combined_session_info = eng_session_info + kor_session_info
        print(f"    📊 Total sessions loaded: {len(combined_session_info)} (English: {len(eng_session_info)}, Korean: {len(kor_session_info)})")
        
        return combined_epochs, combined_session_info
    
    def _prepare_data_for_model(self, epochs):
        """Convert MNE epochs to PyTorch tensors with preprocessing for the hybrid model."""
        # Get EEG data and labels
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        current_samples = X.shape[-1]
        
        print(f"    🔍 Original data shape: {X.shape}")
        print(f"    🔍 Current samples: {current_samples}, Expected: {self.input_window_samples}")
        
        # Crop or pad data to match expected input size
        if current_samples != self.input_window_samples:
            if current_samples > self.input_window_samples:
                # Crop from center
                start_idx = (current_samples - self.input_window_samples) // 2
                X = X[:, :, start_idx:start_idx + self.input_window_samples]
                print(f"    ✂️  Cropped data from {current_samples} to {self.input_window_samples} samples")
            else:
                # Pad with zeros
                pad_size = self.input_window_samples - current_samples
                X = np.pad(X, ((0, 0), (0, 0), (0, pad_size)), mode='constant', constant_values=0)
                print(f"    📏 Padded data from {current_samples} to {self.input_window_samples} samples")
        
        print(f"    ✅ Final data shape: {X.shape}")
        
        y_semantic_raw = epochs.metadata['class_label'].values
        
        # Remap semantic labels to 0-14 range for tasks [1, 2, 4]
        # Original mapping: task1->0-4, task2->5-9, task4->15-19
        # New mapping: task1->0-4, task2->5-9, task4->10-14
        y_semantic = np.copy(y_semantic_raw)
        task_nums = epochs.metadata['task'].values
        
        for i, (label, task) in enumerate(zip(y_semantic_raw, task_nums)):
            if task == 4:  # task4 labels need remapping from 15-19 to 10-14
                y_semantic[i] = label - 5  # 15->10, 16->11, ..., 19->14
        
        # Validate semantic labels are in correct range [0, 14]
        if np.min(y_semantic) < 0 or np.max(y_semantic) >= 15:
            print(f"❌ Invalid semantic labels after remapping - min: {np.min(y_semantic)}, max: {np.max(y_semantic)}")
            print(f"   Tasks present: {np.unique(task_nums)}")
            print(f"   Original labels range: {np.min(y_semantic_raw)}-{np.max(y_semantic_raw)}")
        else:
            print(f"✅ Semantic labels correctly mapped to range [0, 14]")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_semantic_tensor = torch.LongTensor(y_semantic)
        
        return X_tensor, y_semantic_tensor
    
    def calculate_additional_metrics(self, y_true, y_pred, y_probs):
        """Calculate additional performance metrics (Macro F1, Macro Recall, Top-k accuracy, Confusion Matrix)"""
        # Calculate macro F1 score
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Calculate macro recall (balanced accuracy)
        macro_recall = recall_score(y_true, y_pred, average='macro')
        
        # Calculate top-k accuracy
        def top_k_accuracy(y_true, y_probs, k):
            top_k_pred = np.argsort(y_probs, axis=1)[:, -k:]
            correct = 0
            for i, true_label in enumerate(y_true):
                if true_label in top_k_pred[i]:
                    correct += 1
            return correct / len(y_true)
        
        top2_accuracy = top_k_accuracy(y_true, y_probs, 2)
        top3_accuracy = top_k_accuracy(y_true, y_probs, 3)
        
        # Calculate normalized confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return {
            'macro_f1': macro_f1,
            'macro_recall': macro_recall,
            'top2_accuracy': top2_accuracy,
            'top3_accuracy': top3_accuracy,
            'confusion_matrix': cm_normalized
        }
    
    def _save_confusion_matrix_plot(self, cm_normalized, subject_id, save_dir):
        """Save confusion matrix heatmap plot"""
        try:
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(cm_normalized, 
                       annot=True, 
                       fmt='.4f', 
                       cmap='Blues',
                       xticklabels=list(STIMULUS_MAPPING.keys()),
                       yticklabels=list(STIMULUS_MAPPING.keys()),
                       cbar_kws={'label': 'Normalized Count'})
            
            plt.title(f'Confusion Matrix - Baseline Mixed {self.model_name} (Subject: {subject_id})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'confusion_matrix_baseline_eng_subject_{subject_id}.png'
            plot_path = save_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    💾 Confusion matrix plot saved: {plot_filename}")
            return str(plot_path)
            
        except Exception as e:
            print(f"    ⚠️  Error saving confusion matrix plot: {e}")
            plt.close()
            return None
    
    def run_subject_analysis(self, subject_id):
        """Individual subject mixed-language baseline analysis with the hybrid model"""
        print(f"\n{'='*70}")
        print(f"BASELINE MIXED {self.model_name}: Subject-Dependent Mixed-Language Analysis")
        print(f"Subject: {subject_id}")
        print(f"Model: {self.model_name}")
        print(f"Multi-band: {self.band_defs}")
        print(f"{'='*70}")
        
        # Set current subject ID for confusion matrix plot naming
        self._current_subject_id = subject_id
        
        # Load mixed language data
        combined_epochs, session_info = self.load_subject_mixed_data(subject_id)
        
        if combined_epochs is None:
            print(f"❌ Insufficient data for subject {subject_id}")
            return None
        
        # Prepare data
        X, y_semantic = self._prepare_data_for_model(combined_epochs)
        
        print(f"\n📊 Data Summary:")
        print(f"  📈 Total epochs: {len(X)}")
        print(f"  📊 Data shape: {X.shape}")
        print(f"  🎯 Semantic classes: {len(torch.unique(y_semantic))}")
        print(f"  📋 Class distribution: {torch.bincount(y_semantic)}")
        
        # Create stratification labels combining semantic class and language
        # This ensures balanced language distribution across train/val/test splits
        language_labels = combined_epochs.metadata['language'].values
        stratify_labels = np.array([f"{sem}_{lang}" for sem, lang in zip(y_semantic.numpy(), language_labels)])
        
        print(f"\n📊 Language Distribution:")
        unique_langs, lang_counts = np.unique(language_labels, return_counts=True)
        for lang, count in zip(unique_langs, lang_counts):
            print(f"  🌐 {lang}: {count} epochs ({count/len(language_labels)*100:.1f}%)")
        
        # Split data: 7:1.5:1.5 = train:valid:test with language balance
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp, lang_train, lang_temp = train_test_split(
            X, y_semantic, language_labels, test_size=0.3, random_state=42, stratify=stratify_labels
        )
        
        # Create stratify labels for second split
        stratify_temp = np.array([f"{sem}_{lang}" for sem, lang in zip(y_temp.numpy(), lang_temp)])
        
        # Second split: 15% valid, 15% test from the 30% temp
        X_val, X_test, y_val, y_test, lang_val, lang_test = train_test_split(
            X_temp, y_temp, lang_temp, test_size=0.5, random_state=42, stratify=stratify_temp
        )
        
        print(f"\n📊 Data Split:")
        print(f"  🏋️ Train: {len(X_train)} epochs ({len(X_train)/len(X)*100:.1f}%)")
        for lang in np.unique(lang_train):
            lang_count = np.sum(lang_train == lang)
            print(f"      - {lang}: {lang_count} ({lang_count/len(lang_train)*100:.1f}%)")
        
        print(f"  🔍 Valid: {len(X_val)} epochs ({len(X_val)/len(X)*100:.1f}%)")
        for lang in np.unique(lang_val):
            lang_count = np.sum(lang_val == lang)
            print(f"      - {lang}: {lang_count} ({lang_count/len(lang_val)*100:.1f}%)")
        
        print(f"  🧪 Test: {len(X_test)} epochs ({len(X_test)/len(X)*100:.1f}%)")
        for lang in np.unique(lang_test):
            lang_count = np.sum(lang_test == lang)
            print(f"      - {lang}: {lang_count} ({lang_count/len(lang_test)*100:.1f}%)")
        
        # Train and evaluate the hybrid model
        print(f"\n{'='*50}")
        print(f"🤖 Training {self.model_name}")
        print(f"{'='*50}")
        
        try:
            result = self._train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, lang_test)
            if result:
                print(f"\n🏆 {self.model_name} Results:")
                print(f"  📈 Test Accuracy: {result['semantic_accuracy']:.4f}")
                print(f"  🏃 Epochs Trained: {result['epochs_trained']}/{self.model_config['max_epochs']}")
                print(f"  📊 Best Val Loss: {result['best_val_loss']:.4f}")
            else:
                print(f"\n❌ {self.model_name} training failed")
                
        except Exception as e:
            print(f"❌ Error training {self.model_name}: {e}")
            traceback.print_exc()
            result = None
        
        return result
    
    def _train_and_evaluate_model(self, X_train, y_train, X_val, y_val, X_test, y_test, lang_test=None):
        """Train and evaluate the EAGLE + LaBraM hybrid with multi-band semantic learning"""
        print(f"  🤖 Training {self.model_name}...")
        
        # Store language test distribution for later use
        if lang_test is not None:
            self._last_lang_test = lang_test
        
        # Get model configuration
        config = self.model_config
        print(f"    ⚙️  Config: max_epochs={config['max_epochs']}, lr={config['learning_rate']}, patience={config['patience']}")
        print(f"    🎵 Frequency bands: {self.band_defs}")
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        print(f"    📊 Train shape: {X_train.shape}, Semantic classes: {len(torch.unique(y_train))}")
        print(f"    📊 Val shape: {X_val.shape}, Test shape: {X_test.shape}")
        
        # Create hybrid model
        print(f"    🏗️  Creating {self.model_name}...")
        try:
            model = self.create_model().to(self.device)
            print(f"    ✅ {self.model_name} created and moved to {self.device}")
            model_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    📊 Model parameters: {model_params:,} (trainable: {trainable_params:,})")
        except Exception as e:
            print(f"    ❌ Error creating {self.model_name}: {e}")
            traceback.print_exc()
            return None
        
        # Loss function configuration based on loss_type
        if self.loss_type == 'ce':
            criterion = nn.CrossEntropyLoss()
            print(f"    📊 Using Standard Cross Entropy Loss")
        elif self.loss_type == 'label_smoothing':
            criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            print(f"    📊 Using Cross Entropy Loss with Label Smoothing (α={self.label_smoothing})")
        elif self.loss_type == 'scl_ce_hybrid':
            # TODO: Implement Supervised Contrastive Loss + Cross Entropy Hybrid
            raise NotImplementedError("Supervised Contrastive Loss + CE Hybrid not yet implemented")
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Choose from ['ce', 'label_smoothing', 'scl_ce_hybrid']")
        
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler_kwargs = {
            'mode': 'min',
            'factor': config['scheduler_factor'],
            'patience': config['scheduler_patience'],
            'min_lr': 1e-7,
        }
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                **scheduler_kwargs,
            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_kwargs,
            )
        
        print(f"    ⚙️  Optimizer and scheduler ready")
        
        # Create simple data loader (no multiprocessing workers)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # No multiprocessing to avoid Windows paging file issues
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop with early stopping
        model.train()
        print(f"    🏋️ Training for max {config['max_epochs']} epochs (patience={config['patience']})...")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(config['max_epochs']):
            # Training phase
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Apply warmup learning rate
            if epoch < config['warmup_epochs']:
                warmup_lr = config['learning_rate'] * (epoch + 1) / config['warmup_epochs']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            for batch_idx, (batch_X, batch_y_sem) in enumerate(train_loader):
                # === Original loss computation (can rollback by uncommenting) ===
                # optimizer.zero_grad()
                # batch_X = batch_X.unsqueeze(1)  # (B, 1, C, T)
                # logits = model(batch_X)
                # loss = criterion(logits, batch_y_sem)
                # loss.backward()
                # optimizer.step()
                # total_loss += loss.item()
                # num_batches += 1
                # === End of original loss ===
                
                # === New loss with Center Loss ===
                if epoch == 0 and batch_idx == 0:
                    print(f"      🔄 Starting first batch - batch shape: {batch_X.shape}")
                    if torch.cuda.is_available():
                        print(f"      💾 GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    else:
                        print(f"      💻 Using CPU")
                
                # Print progress every 20 batches (first epoch only)
                if epoch == 0 and batch_idx > 0 and batch_idx % 20 == 0:
                    print(f"      ⏳ Processing batch {batch_idx}/{len(train_loader)}...")
                
                optimizer.zero_grad()
                
                # Forward pass
                try:
                    # Prepare input - hybrid model expects 4D: (batch, 1, channels, time)
                    if len(batch_X.shape) == 3:
                        model_input = batch_X.unsqueeze(1)
                    else:
                        model_input = batch_X
                    
                    if epoch == 0 and batch_idx == 0:
                        print(f"      🔍 Model input shape: {model_input.shape}")
                        print(f"      🎵 Multi-band processing: {len(self.band_defs)} frequency branches")
                    
                    # Ensure targets are proper shape
                    if len(batch_y_sem.shape) > 1:
                        batch_y_sem = batch_y_sem.squeeze()
                    
                    # Forward pass - training mode (no attention extraction)
                    outputs = model(model_input, return_attention=False)
                    
                    # Handle tuple return (logits, glob)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        feats = outputs[1] if len(outputs) > 1 else None
                    else:
                        logits = outputs
                        feats = None
                    
                    # Cross-entropy loss only
                    loss = criterion(logits, batch_y_sem)
                    
                    if epoch == 0 and batch_idx == 0:
                        print(f"      ✅ Forward pass completed")
                        print(f"      🔍 Logits shape: {logits.shape}")
                        if feats is not None:
                            print(f"      🔍 Features shape: {feats.shape}")
                        print(f"      📊 Using Cross Entropy Loss only")
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"      ❌ Error in training batch {batch_idx}: {e}")
                    if epoch == 0 and batch_idx == 0:
                        traceback.print_exc()
                        return None  # Return early if first batch fails
                    continue
            
            # First epoch completion message
            if epoch == 0:
                print(f"      ✅ First epoch completed - processed {num_batches} batches")
            
            if num_batches == 0:
                print(f"      ❌ No successful batches in epoch {epoch}")
                continue
                
            avg_train_loss = total_loss / num_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_batches = 0
            
            with torch.no_grad():
                # === Original validation (can rollback by uncommenting) ===
                # val_X_unsqueezed = X_val.unsqueeze(1)
                # val_logits = model(val_X_unsqueezed)
                # loss = criterion(val_logits, y_val)
                # val_loss += loss.item()
                # _, predicted = torch.max(val_logits, 1)
                # val_correct += (predicted == y_val).sum().item()
                # val_total += y_val.size(0)
                # === End of original validation ===
                
                # === New validation with Center Loss ===
                try:
                    # Process validation data in batches
                    for i in range(0, len(X_val), self.batch_size):
                        batch_X_val = X_val[i:i+self.batch_size]
                        batch_y_val = y_val[i:i+self.batch_size]
                        
                        # Prepare input
                        if len(batch_X_val.shape) == 3:
                            model_input_val = batch_X_val.unsqueeze(1)
                        else:
                            model_input_val = batch_X_val
                        
                        # Ensure targets are proper shape
                        if len(batch_y_val.shape) > 1:
                            batch_y_val = batch_y_val.squeeze()
                        
                        # Forward - validation mode (no attention extraction)
                        outputs = model(model_input_val, return_attention=False)
                        
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                            feats = outputs[1] if len(outputs) > 1 else None
                        else:
                            logits = outputs
                            feats = None
                        
                        # Calculate loss (Cross Entropy only)
                        loss = criterion(logits, batch_y_val)
                        
                        val_loss += loss.item()
                        val_batches += 1
                        _, predicted = torch.max(logits.data, 1)
                        val_total += batch_y_val.size(0)
                        val_correct += (predicted == batch_y_val).sum().item()
                    
                    avg_val_loss = val_loss / max(1, val_batches)
                    val_accuracy = val_correct / val_total
                    
                    # Update training history
                    training_history['train_loss'].append(avg_train_loss)
                    training_history['val_loss'].append(avg_val_loss)
                    training_history['val_acc'].append(val_accuracy)
                    
                    # Learning rate scheduling (after warmup)
                    if epoch >= config['warmup_epochs']:
                        scheduler.step(avg_val_loss)
                    
                    # Early stopping check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        best_model_state = copy.deepcopy(model.state_dict())
                    else:
                        patience_counter += 1
                    
                    # Print progress - more frequently
                    if (epoch + 1) % 5 == 0 or epoch < 10:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"      Epoch [{epoch+1}/{config['max_epochs']}] "
                              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                              f"Val Acc: {val_accuracy:.4f}, LR: {current_lr:.2e}, "
                              f"Patience: {patience_counter}/{config['patience']}")
                    elif (epoch + 1) % 1 == 0:
                        # Simple progress indicator every epoch
                        print(f"      ⏳ Epoch {epoch+1}/{config['max_epochs']} - Val Acc: {val_accuracy:.4f}")
                    
                    # Early stopping
                    if patience_counter >= config['patience']:
                        print(f"      🛑 Early stopping at epoch {epoch+1}")
                        break
                        
                except Exception as e:
                    print(f"      ❌ Error in validation: {e}")
                    continue
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"      ✅ Loaded best model (val_loss: {best_val_loss:.4f})")
            
            # Save best model to disk
            subject_id = getattr(self, '_current_subject_id', 'unknown')
            model_filename = self.models_dir / f"best_model_mix_{subject_id}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'best_val_loss': best_val_loss,
                'model_config': {
                    'n_classes': self.n_classes,
                    'n_channels': self.n_channels,
                    'samples': self.input_window_samples,
                    'sfreq': self.sfreq,
                    'band_defs': self.band_defs,
                    'kernel_lengths': model.kernel_lengths
                }
            }, model_filename)
            print(f"      💾 Model saved: {model_filename.name}")
            
            # Save training history
            history_filename = self.training_history_dir / f"training_history_mix_{subject_id}.csv"
            import pandas as pd
            history_df = pd.DataFrame({
                'epoch': list(range(1, len(training_history['train_loss']) + 1)),
                'train_loss': training_history['train_loss'],
                'val_loss': training_history['val_loss'],
                'val_acc': training_history['val_acc']
            })
            history_df.to_csv(history_filename, index=False)
            print(f"      💾 Training history saved: {history_filename.name}")
        
        # Final evaluation on test set
        model.eval()
        all_preds = []
        all_probs = []
        all_band_weights = []  # Collect band attention weights
        all_embeddings = []  # Collect latent embeddings (glob) for UMAP/t-SNE
        all_temporal_attention = []  # Collect temporal attention weights for Band×Time heatmap
        
        with torch.no_grad():
            try:
                # === Original test evaluation (can rollback by uncommenting) ===
                # test_X_unsqueezed = X_test.unsqueeze(1)
                # test_logits = model(test_X_unsqueezed)
                # probabilities = F.softmax(test_logits, dim=1)
                # _, predicted = torch.max(test_logits, 1)
                # all_preds.extend(predicted.cpu().numpy())
                # all_probs.extend(probabilities.cpu().numpy())
                # === End of original test ===
                
                # === New test evaluation (handle tuple return) ===
                # Process test data in batches
                for i in range(0, len(X_test), self.batch_size):
                    batch_X = X_test[i:i+self.batch_size]
                    
                    # Prepare input
                    if len(batch_X.shape) == 3:
                        model_input_test = batch_X.unsqueeze(1)
                    else:
                        model_input_test = batch_X
                    
                    # Forward - test mode with attention extraction
                    outputs = model(model_input_test, return_attention=True)
                    
                    # Extract all outputs including attention weights
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        glob = outputs[1] if len(outputs) > 1 else None
                        band_weights = outputs[2] if len(outputs) > 2 else None
                        temporal_attention = outputs[3] if len(outputs) > 3 else None
                    else:
                        logits = outputs
                        glob = None
                        band_weights = None
                        temporal_attention = None
                    
                    # Get probabilities using softmax
                    probs = F.softmax(logits, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    
                    # Get semantic predictions
                    sem_preds = torch.argmax(logits, dim=1)
                    all_preds.extend(sem_preds.cpu().numpy())
                    
                    # Collect band weights, embeddings, and temporal attention if available
                    if band_weights is not None:
                        all_band_weights.extend(band_weights.cpu().numpy())
                    if glob is not None:
                        all_embeddings.extend(glob.cpu().numpy())
                    if temporal_attention is not None:
                        all_temporal_attention.extend(temporal_attention.cpu().numpy())
                
            except Exception as e:
                print(f"    ❌ Error in evaluation: {e}")
                return None
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        y_true = y_test.cpu().numpy()
        
        # Save test results for confusion matrix and band weight visualization
        subject_id = getattr(self, '_current_subject_id', 'unknown')
        test_results_filename = self.test_results_dir / f"test_results_mix_{subject_id}.npz"
        lang_test_np = lang_test if lang_test is not None else None
        np.savez(test_results_filename,
                 y_true=y_true,
                 y_pred=all_preds,
                 y_probs=all_probs,
                 embeddings=np.array(all_embeddings) if all_embeddings else None,
                 band_weights=np.array(all_band_weights) if all_band_weights else None,
                 temporal_attention=np.array(all_temporal_attention) if all_temporal_attention else None,
                 lang_test=lang_test_np,
                 subject_id=subject_id)
        print(f"    💾 Test results saved: {test_results_filename.name}")
        
        # Calculate basic accuracy
        semantic_accuracy = accuracy_score(y_true, all_preds)
        
        # Calculate additional metrics
        additional_metrics = self.calculate_additional_metrics(y_true, all_preds, all_probs)
        
        # Save confusion matrix plot
        plot_dir = self.results_dir / "confusion_matrices"
        plot_dir.mkdir(exist_ok=True)
        confusion_plot_path = self._save_confusion_matrix_plot(
            additional_metrics['confusion_matrix'], 
            getattr(self, '_current_subject_id', 'unknown'), 
            plot_dir
        )
        
        # Add language distribution info
        lang_test_dist = {}
        if hasattr(self, '_last_lang_test'):
            for lang in np.unique(self._last_lang_test):
                lang_test_dist[lang] = int(np.sum(self._last_lang_test == lang))
        
        result = {
            'model': self.model_name,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'language_distribution': lang_test_dist if lang_test_dist else {},
            'semantic_accuracy': semantic_accuracy,
            'macro_f1': additional_metrics['macro_f1'],
            'macro_recall': additional_metrics['macro_recall'],
            'top2_accuracy': additional_metrics['top2_accuracy'],
            'top3_accuracy': additional_metrics['top3_accuracy'],
            'confusion_matrix': additional_metrics['confusion_matrix'].tolist(),
            'confusion_plot_path': confusion_plot_path,
            'n_classes': len(torch.unique(y_train)),
            'model_params': model_params,
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'final_val_acc': training_history['val_acc'][-1] if training_history['val_acc'] else 0.0
        }
        
        print(f"    ✅ {self.model_name} Results:")
        print(f"      📈 Test Accuracy: {semantic_accuracy:.4f}")
        print(f"      📊 Macro F1: {additional_metrics['macro_f1']:.4f}")
        print(f"      🎯 Macro Recall: {additional_metrics['macro_recall']:.4f}")
        print(f"      🥈 Top-2 Accuracy: {additional_metrics['top2_accuracy']:.4f}")
        print(f"      🥉 Top-3 Accuracy: {additional_metrics['top3_accuracy']:.4f}")
        print(f"      🏃 Epochs Trained: {epoch + 1}/{config['max_epochs']}")
        print(f"      📊 Best Val Loss: {best_val_loss:.4f}")
        print(f"      🎵 Frequency bands used: {self.band_defs}")
        
        return result
    
    def run_all_subjects(self):
        """Execute analysis for all subjects with both English and Korean data"""
        print(f"\n🚀 Starting Baseline Mixed Language {self.model_name} Analysis for All Subjects")
        
        # Get all subject IDs with English or Korean data
        subjects = set()
        
        # Find files for both English and Korean data
        for language in ['eng', 'kor']:
            files = self.loader.find_eeg_files(
                language=language,
                tasks=self.selected_tasks
            )
            
            for file_path in files:
                filename = Path(file_path).stem
                subject_id = filename.split('_')[1]
                subjects.add(subject_id)
        
        subjects = sorted(list(subjects))
        print(f"📋 Found {len(subjects)} subjects with English/Korean data: {subjects}")
        
        all_results = {}
        
        for subject_id in subjects:
            try:
                subject_result = self.run_subject_analysis(subject_id)
                all_results[subject_id] = subject_result
            except Exception as e:
                print(f"❌ Error processing subject {subject_id}: {e}")
                all_results[subject_id] = None
        
        # Summarize results
        self._summarize_results(all_results)
        
        return all_results
    
    def run_first_subject_test(self):
        """Execute analysis for the first subject only (for testing)"""
        print(f"\n🧪 Starting Baseline Mixed Language {self.model_name} Analysis for First Subject (Testing Mode)")
        
        # Get all subject IDs with English or Korean data
        subjects = set()
        
        for language in ['eng', 'kor']:
            files = self.loader.find_eeg_files(
                language=language,
                tasks=self.selected_tasks
            )
            
            for file_path in files:
                filename = Path(file_path).stem
                subject_id = filename.split('_')[1]
                subjects.add(subject_id)
        
        subjects = sorted(list(subjects))
        if not subjects:
            print("❌ No subjects with English/Korean data found!")
            return None
            
        # Test only the first subject  
        first_subject = subjects[0]  # Use index 0 for the first subject
        print(f"🎯 Testing with first subject: {first_subject}")
        print(f"📋 Total subjects available: {len(subjects)}")
        
        try:
            subject_result = self.run_subject_analysis(first_subject)
            all_results = {first_subject: subject_result}
            
            # Summarize results
            self._summarize_results(all_results)
            
            return all_results
            
        except Exception as e:
            print(f"❌ Error processing subject {first_subject}: {e}")
            traceback.print_exc()
            return None
    
    def _summarize_results(self, all_results):
        """Summarize and print results for the EAGLE + LaBraM hybrid"""
        print(f"\n{'='*80}")
        print(f"BASELINE MIXED LANGUAGE {self.model_name}: RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Process results for each subject
        valid_results = []
        for subject_id, result in all_results.items():
            print(f"\n📊 Subject {subject_id} Results:")
            print(f"{'='*50}")
            
            if result:
                print(f"  🤖 {self.model_name}: {result['semantic_accuracy']:.4f} "
                      f"({result['model_params']:,} params, {result['epochs_trained']} epochs)")
                valid_results.append(result)
            else:
                print(f"  ❌ {self.model_name}: Training failed")
        
        # Overall summary
        if valid_results:
            avg_acc = np.mean([r['semantic_accuracy'] for r in valid_results])
            std_acc = np.std([r['semantic_accuracy'] for r in valid_results])
            print(f"\n🏆 Overall {self.model_name} Performance:")
            print(f"  📈 Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
            print(f"  📊 Successful Subjects: {len(valid_results)}/{len(all_results)}")
            print(f"  🎵 Multi-band architecture: {self.band_defs}")
        
        print(f"\n{'='*80}")
        
        # Save results to file
        self._save_results_to_file(all_results)
    
    def _save_results_to_file(self, all_results):
        """Save experiment results to text and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed text report
        txt_filename = self.results_dir / f"BaselineMixed_{self.model_name}_{timestamp}.txt"
        csv_filename = self.results_dir / f"BaselineMixed_{self.model_name}_{timestamp}.csv"
        
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"BASELINE MIXED LANGUAGE {self.model_name} RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Frequency Bands: {self.band_defs}\n")
            f.write(f"LaBraM Variant: {self.labram_variant}\n")
            f.write(f"Freeze LaBraM: {self.freeze_labram}\n")
            f.write(f"Unfreeze Last N Blocks: {self.unfreeze_last_n_blocks}\n")
            f.write(f"Loss Type: {self.loss_type}\n")
            if self.loss_type == 'label_smoothing':
                f.write(f"Label Smoothing: {self.label_smoothing}\n")
            f.write(f"Languages: English + Korean (Balanced)\n")
            f.write(f"Max Epochs: {self.model_config['max_epochs']}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Learning Rate: {self.model_config['learning_rate']}\n")
            f.write(f"Data Split: 7:1.5:1.5 (train:valid:test) with language stratification\n")
            f.write("-"*80 + "\n\n")
            
            # Process each subject
            for subject_id, result in all_results.items():
                f.write(f"SUBJECT {subject_id} RESULTS:\n")
                f.write("-"*50 + "\n")
                
                if result:
                    f.write(f"{self.model_name} Accuracy: {result['semantic_accuracy']:.4f}\n")
                    f.write(f"Macro F1 Score: {result['macro_f1']:.4f}\n")
                    f.write(f"Macro Recall (Balanced Acc): {result['macro_recall']:.4f}\n")
                    f.write(f"Top-2 Accuracy: {result['top2_accuracy']:.4f}\n")
                    f.write(f"Top-3 Accuracy: {result['top3_accuracy']:.4f}\n")
                    f.write(f"Model Params: {result['model_params']:,}\n")
                    f.write(f"Epochs Trained: {result['epochs_trained']}\n")
                    f.write(f"Best Val Loss: {result['best_val_loss']:.4f}\n")
                    f.write(f"Final Val Acc: {result['final_val_acc']:.4f}\n")
                    f.write(f"Train Size: {result['train_size']}\n")
                    f.write(f"Val Size: {result['val_size']}\n")
                    f.write(f"Test Size: {result['test_size']}\n")
                    if result.get('confusion_plot_path'):
                        f.write(f"Confusion Matrix Plot: {result['confusion_plot_path']}\n")
                else:
                    f.write("Training failed\n")
                
                f.write("\n" + "="*60 + "\n\n")
        
        # Create CSV file for easy data analysis
        csv_data = []
        csv_headers = [
            'Subject', 'Model', 'Semantic_Accuracy', 'Macro_F1', 'Macro_Recall', 
            'Top2_Accuracy', 'Top3_Accuracy', 'Train_Size', 'Val_Size', 'Test_Size', 
            'Model_Params', 'Epochs_Trained', 'Best_Val_Loss', 'Final_Val_Acc', 'Confusion_Plot_Path'
        ]
        
        for subject_id, result in all_results.items():
            if result:
                csv_data.append([
                    subject_id,
                    self.model_name,
                    result['semantic_accuracy'],
                    result['macro_f1'],
                    result['macro_recall'],
                    result['top2_accuracy'],
                    result['top3_accuracy'],
                    result['train_size'],
                    result['val_size'],
                    result['test_size'],
                    result['model_params'],
                    result['epochs_trained'],
                    result['best_val_loss'],
                    result['final_val_acc'],
                    result.get('confusion_plot_path', '')
                ])
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            writer.writerows(csv_data)
            
            # Add summary statistics (average and std) if there are valid results
            if csv_data:
                writer.writerow([])  # Empty row for separation
                
                # Calculate statistics for numeric columns (indices 2-13)
                numeric_cols = list(range(2, 14))  # Semantic_Accuracy to Final_Val_Acc
                
                # Average row
                avg_row = ['AVERAGE', '']
                for col_idx in numeric_cols:
                    values = [row[col_idx] for row in csv_data if row[col_idx] != '']
                    if values:
                        avg_row.append(np.mean(values))
                    else:
                        avg_row.append('')
                avg_row.append('')  # Confusion_Plot_Path column
                writer.writerow(avg_row)
                
                # Standard deviation row
                std_row = ['STD', '']
                for col_idx in numeric_cols:
                    values = [row[col_idx] for row in csv_data if row[col_idx] != '']
                    if values:
                        std_row.append(np.std(values, ddof=1) if len(values) > 1 else 0)
                    else:
                        std_row.append('')
                std_row.append('')  # Confusion_Plot_Path column
                writer.writerow(std_row)
        
        print(f"\n💾 Results saved to:")
        print(f"  📄 Detailed report: {txt_filename}")
        print(f"  📊 CSV data (with summary stats): {csv_filename}")
        print(f"  📂 Models directory: {self.models_dir}")
        print(f"  📈 Training history directory: {self.training_history_dir}")
        print(f"  📋 Test results directory: {self.test_results_dir}")

if __name__ == "__main__":
    # ========================================
    # Configuration: Change loss_type here
    # ========================================
    # Options:
    #   'ce' - Standard Cross Entropy Loss (default)
    #   'label_smoothing' - Cross Entropy with Label Smoothing
    #   'scl_ce_hybrid' - Supervised Contrastive Loss + CE Hybrid (future)
    
    LOSS_TYPE = 'ce'  # Change this to 'label_smoothing' to enable label smoothing
    LABEL_SMOOTHING = 0.1  # Only used when LOSS_TYPE='label_smoothing'
    
    # Execute Baseline Mixed Language with EAGLE + LaBraM
    baseline_mix = BaselineMixedEEGNet(
        loss_type=LOSS_TYPE,
        label_smoothing=LABEL_SMOOTHING
    )
    
    print("\n" + "="*70)
    print("🧪 Running Baseline Mixed Language Analysis")
    print("="*70)
    print(f"🤖 Model: {baseline_mix.model_name}")
    print(f"🎵 Frequency bands: {baseline_mix.band_defs}")
    print(f"🧠 LaBraM variant: {baseline_mix.labram_variant}")
    print(f"🧊 Freeze LaBraM: {baseline_mix.freeze_labram}")
    print(f"📊 Loss Type: {baseline_mix.loss_type}")
    if baseline_mix.loss_type == 'label_smoothing':
        print(f"📊 Label Smoothing: {baseline_mix.label_smoothing}")
    print(f"🌐 Languages: English + Korean (Balanced)")
    print(f"📦 Batch size: {baseline_mix.batch_size}")
    print(f"⚙️  Max epochs: {baseline_mix.model_config['max_epochs']}")
    print("="*70)
    
    # Run analysis
    # results = baseline_mix.run_first_subject_test()  # Test mode
    results = baseline_mix.run_all_subjects()  # Full analysis
    # results = baseline_mix.run_first_subject_test()
    
    if results:
        print("\n✅ Analysis completed successfully!")
        print(f"� Results saved to: {baseline_mix.results_dir}")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")
