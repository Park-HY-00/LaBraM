"""
Baseline Korean: Within-Subject, Session-Agnostic Korean-Only Training

For each subject: All Korean sessions combined (task1, task2, task4)
Data split: 7:1.5:1.5 = train:valid:test
Training includes only Korean data with 15-class semantic labels
Purpose: Establish baseline performance for Korean-only within-subject learning
Models: EEGNet and TIDNet from braindecode
"""

import sys
import os
# sys.path.append('..')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set CUDA_LAUNCH_BLOCKING for better error debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import csv
from utils import (
    EEGDataLoader, 
    SELECTED_TASKS, 
    STIMULUS_MAPPING, 
    EPOCH_TMIN, 
    EPOCH_TMAX,
    create_epochs_from_raw,
    load_subject_sessions_data
)

# Import custom models like trackA.py
from model.eegnet import EEGNet as CustomEEGNet
from model.proposed import ExtendedSemanticEEGNet

# Braindecode models
try:
    # Try braindecode 0.8 style imports
    try:
        from braindecode.models import ShallowConvNet, Deep4Net, TIDNet
    except ImportError:
        ShallowConvNet = None
        Deep4Net = None
        TIDNet = None
        
except ImportError as e:
    print(f"⚠️  Could not import braindecode models: {e}")
    ShallowConvNet = None
    Deep4Net = None
    TIDNet = None

import warnings
warnings.filterwarnings('ignore')

class BaselineKorean:
    def __init__(self, data_path="D:\\SemanticDecoding\\Dataset\\"):
        self.data_path = data_path
        self.loader = EEGDataLoader(data_path)
        # Use common parameters from utils
        self.selected_tasks = SELECTED_TASKS  # task1, task2, task4 only
        self.stimulus_mapping = STIMULUS_MAPPING
        self.epoch_tmin = EPOCH_TMIN
        self.epoch_tmax = EPOCH_TMAX
        
        # EEG data configuration
        self.n_channels = 64
        self.n_classes = 15  # Semantic classes for tasks 1, 2, 4
        self.input_window_samples = int((EPOCH_TMAX - EPOCH_TMIN) * 250)  # 7.01s * 250Hz = 1751 samples
        self.sfreq = 250.0
        
        # Models to test (using same models as trackA.py)
        # Options: 'EEGNet', 'TIDNet', 'ExtendedSemanticEEGNet'
        self.model_names = ['ExtendedSemanticEEGNet']  # Currently testing only EAGLE model
        # self.model_names = ['EEGNet', 'TIDNet', 'ExtendedSemanticEEGNet']  # Uncomment to test all models
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        print(f"🔧 Input window samples: {self.input_window_samples}")
        print(f"🔧 Input window seconds: {self.input_window_samples / self.sfreq:.2f}s")
        
        # Model-specific training configurations
        self.model_configs = {
            'EEGNet': {
                'max_epochs': 200,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'patience': 30,
                'warmup_epochs': 15,
                'scheduler_factor': 0.5,
                'scheduler_patience': 20
            },
            'TIDNet': {
                'max_epochs': 250,
                'learning_rate': 1e-3,
                'weight_decay': 5e-4,
                'patience': 35,
                'warmup_epochs': 20,
                'scheduler_factor': 0.5,
                'scheduler_patience': 25
            },
            'ExtendedSemanticEEGNet': {
                'max_epochs': 300,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'patience': 40,
                'warmup_epochs': 20,
                'scheduler_factor': 0.5,
                'scheduler_patience': 25
            }
        }
        
        # Common training hyperparameters
        self.batch_size = 16
        
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
    
    def create_braindecode_model(self, model_name):
        """Create braindecode model based on model name"""
        # Use shorter window for model compatibility with some models
        shorter_window_samples = 250  # 1 second at 250Hz
        shorter_window_seconds = shorter_window_samples / self.sfreq
        
        print(f"    ⚙️  Using window for model creation: {shorter_window_samples} samples ({shorter_window_seconds:.2f}s)")
        
        if model_name == 'EEGNet':
            # Use custom EEGNet implementation like trackA.py
            return CustomEEGNet(
                nb_classes=self.n_classes,
                Chans=self.n_channels,
                Samples=self.input_window_samples,  # Use full 1751 samples like trackA.py
                dropoutRate=0.5,  # within-subject: 0.5
                kernLength=64,
                F1=8,
                D=2,
                F2=16
            )
                
        elif model_name == 'TIDNet':
            if TIDNet is None:
                raise ValueError(f"TIDNet is not available in this braindecode version")
            # Use full window samples like trackA.py
            return TIDNet(
                n_chans=self.n_channels,
                n_outputs=self.n_classes,
                input_window_samples=self.input_window_samples,  # Use full 1751 samples
                s_growth=24,
                t_filters=32,
                drop_prob=0.4,
                pooling=15,
                temp_layers=2,
                spat_layers=2,
                temp_span=0.05,
                bottleneck=3,
                summary=-1
            )
        elif model_name == 'ExtendedSemanticEEGNet':
            # Use 500Hz sampling rate and 1500 samples (3.0s)
            return ExtendedSemanticEEGNet(
                nb_classes=self.n_classes,
                Chans=self.n_channels,
                Samples=1500,  # 3.0s at 500Hz
                sfreq=500.0,
                dropoutRate=0.2,  # Model default
                F1=16,  # Model default
                D=2,
                gamma_F1=32,  # Model default
                reduced_dim=48,  # Model default
                band_defs=((8, 30), (30, 60), (60, 80)),
                use_transformer=True  # Model default
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def load_subject_korean_data(self, subject_id):
        """Load all Korean session data for a specific subject using trackA.py approach"""
        print(f"  📂 Loading Korean data for subject {subject_id}...")
        
        # Use the same approach as trackA.py
        all_epochs, session_info = load_subject_sessions_data(self.loader, subject_id, self.selected_tasks)
        
        # Extract only Korean data
        kor_epochs = all_epochs.get('kor', [])
        kor_session_info = session_info.get('kor', [])
        
        if not kor_epochs:
            print(f"    ❌ No Korean epochs found for subject {subject_id}")
            return None, None
        
        # Combine all Korean epochs
        combined_epochs = mne.concatenate_epochs(kor_epochs)
        total_epochs = len(combined_epochs)
        
        print(f"    ✅ Total combined Korean epochs: {total_epochs}")
        print(f"    📊 Korean sessions loaded: {len(kor_session_info)}")
        
        return combined_epochs, kor_session_info
    
    def calculate_additional_metrics(self, y_true, y_pred, y_probs):
        """Calculate additional performance metrics (Macro F1, Top-k accuracy)"""
        from sklearn.metrics import f1_score
        
        # Calculate macro F1 score
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Calculate top-k accuracy
        def top_k_accuracy(y_true, y_probs, k):
            top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
            correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
            return correct.mean()
        
        top2_accuracy = top_k_accuracy(y_true, y_probs, 2)
        top3_accuracy = top_k_accuracy(y_true, y_probs, 3)
        
        return {
            'macro_f1': macro_f1,
            'top2_accuracy': top2_accuracy,
            'top3_accuracy': top3_accuracy
        }
    
    def prepare_data_for_model(self, epochs):
        """Convert MNE epochs to PyTorch tensors with proper preprocessing"""
        # Get EEG data and labels
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
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
    
    def run_subject_analysis(self, subject_id):
        """Individual subject Korean-only baseline analysis"""
        print(f"\n{'='*70}")
        print(f"BASELINE KOREAN: Subject-Dependent Korean-Only Analysis")
        print(f"Subject: {subject_id}")
        print(f"Models to test: {', '.join(self.model_names)}")
        print(f"{'='*70}")
        
        # Set current subject ID for file naming
        self._current_subject_id = subject_id
        
        # Load Korean data
        combined_epochs, session_info = self.load_subject_korean_data(subject_id)
        
        if combined_epochs is None:
            print(f"❌ Insufficient Korean data for subject {subject_id}")
            return None
        
        # Prepare data
        X, y_semantic = self.prepare_data_for_model(combined_epochs)
        
        print(f"\n📊 Data Summary:")
        print(f"  📈 Total epochs: {len(X)}")
        print(f"  📊 Data shape: {X.shape}")
        print(f"  🎯 Semantic classes: {len(torch.unique(y_semantic))}")
        print(f"  📋 Class distribution: {torch.bincount(y_semantic)}")
        
        # Split data: 7:1.5:1.5 = train:valid:test
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_semantic, test_size=0.3, random_state=42, stratify=y_semantic
        )
        
        # Second split: 15% valid, 15% test from the 30% temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\n📊 Data Split:")
        print(f"  🏋️ Train: {len(X_train)} epochs ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  🔍 Valid: {len(X_val)} epochs ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  🧪 Test: {len(X_test)} epochs ({len(X_test)/len(X)*100:.1f}%)")
        
        # Store results for all models
        all_model_results = {}
        
        # Test each model
        for model_name in self.model_names:
            print(f"\n{'='*50}")
            print(f"🤖 Testing {model_name}")
            print(f"{'='*50}")
            
            try:
                result = self._train_and_evaluate_single_model(
                    X_train, y_train, X_val, y_val, X_test, y_test, model_name
                )
                
                if result is not None:
                    all_model_results[model_name] = result
                    print(f"✅ {model_name} completed successfully")
                else:
                    print(f"❌ {model_name} failed")
                    all_model_results[model_name] = None
                    
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                all_model_results[model_name] = None
        
        return all_model_results
    
    def _train_and_evaluate_single_model(self, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
        """Train and evaluate a single braindecode model"""
        print(f"  🤖 Training {model_name}...")
        
        # Get model-specific configuration
        config = self.model_configs[model_name]
        print(f"    ⚙️  Config: max_epochs={config['max_epochs']}, lr={config['learning_rate']}, patience={config['patience']}")
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        # Create model
        print(f"    🏗️  Creating {model_name}...")
        try:
            model = self.create_braindecode_model(model_name).to(self.device)
            print(f"    ✅ {model_name} created and moved to {self.device}")
            model_params = sum(p.numel() for p in model.parameters())
            print(f"    📊 Model parameters: {model_params:,}")
        except Exception as e:
            print(f"    ❌ Error creating {model_name}: {e}")
            return None
        
        # Model-specific optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=config['scheduler_factor'],
            patience=config['scheduler_patience'],
            verbose=True,
            min_lr=1e-7
        )
        
        print(f"    ⚙️  Optimizer and scheduler ready")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        
        # Training loop with early stopping
        model.train()
        print(f"    🏋️ Training for max {config['max_epochs']} epochs (patience={config['patience']})...")
        
        # Training history tracking for plots
        training_history = {
            'epoch': [],
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
                if epoch == 0 and batch_idx == 0:
                    print(f"      🔄 Starting first batch - batch shape: {batch_X.shape}")
                    print(f"      💾 GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "      💻 Using CPU")
                
                optimizer.zero_grad()
                
                # Forward pass
                try:
                    if epoch == 0 and batch_idx == 0:
                        print(f"      🧠 Forward pass starting...")
                    
                    # Prepare input based on model type
                    if model_name == 'EEGNet':
                        # CustomEEGNet expects 3D input: (batch, channels, time)
                        model_input = batch_X
                    elif model_name == 'ExtendedSemanticEEGNet':
                        # ExtendedSemanticEEGNet expects 4D input: (batch, 1, channels, time)
                        # Need to resample from 250Hz/1751 samples to 500Hz/1500 samples
                        target_samples = 1500
                        current_samples = batch_X.shape[-1]  # 1751
                        if current_samples != target_samples:
                            # Use bilinear interpolation
                            model_input = torch.nn.functional.interpolate(
                                batch_X.unsqueeze(1),  # (batch, 1, channels, time)
                                size=(batch_X.shape[1], target_samples),
                                mode='bilinear',
                                align_corners=False
                            )  # Keep 4D: (batch, 1, channels, time)
                        else:
                            model_input = batch_X.unsqueeze(1)  # (batch, 1, channels, time)
                    else:
                        # All other braindecode models expect 4D input: (batch, 1, channels, time)
                        if len(batch_X.shape) == 3:  # (batch, channels, time)
                            model_input = batch_X.unsqueeze(1)  # (batch, 1, channels, time)
                        else:
                            model_input = batch_X
                        
                        # Handle time dimension mismatch for other braindecode models
                        target_time = 250  # 1s at 250Hz
                        current_time = model_input.shape[-1]  # 1751
                        
                        if current_time != target_time and current_time > target_time:
                            # Crop from center
                            start_idx = (current_time - target_time) // 2
                            model_input = model_input[:, :, :, start_idx:start_idx + target_time]
                        elif current_time < target_time:
                            # Pad with zeros
                            pad_size = target_time - current_time
                            model_input = F.pad(model_input, (0, pad_size), mode='constant', value=0)
                    
                    if epoch == 0 and batch_idx == 0:
                        print(f"      🔍 {model_name} input shape: {model_input.shape}")
                        print(f"      🔍 Input dimensions: {model_input.dim()}D tensor")
                    
                    outputs = model(model_input)
                    if epoch == 0 and batch_idx == 0:
                        print(f"      ✅ Forward pass completed")
                    
                    # Handle model output format
                    if model_name == 'ExtendedSemanticEEGNet':
                        # ExtendedSemanticEEGNet returns (logits, features)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # Use only logits for loss
                    
                    # Compute loss
                    if epoch == 0 and batch_idx == 0:
                        print(f"      🔍 Debug: outputs shape = {outputs.shape}, targets shape = {batch_y_sem.shape}")
                        print(f"      🔍 Debug: outputs range = [{outputs.min():.3f}, {outputs.max():.3f}]")
                        print(f"      🔍 Debug: targets range = [{batch_y_sem.min()}, {batch_y_sem.max()}]")
                    
                    # Ensure targets are proper shape for CrossEntropyLoss
                    if len(batch_y_sem.shape) > 1:
                        batch_y_sem = batch_y_sem.squeeze()
                    
                    loss = criterion(outputs, batch_y_sem)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"      ❌ Error in training batch {batch_idx}: {e}")
                    if epoch == 0 and batch_idx == 0:
                        return None  # Return early if first batch fails
                    continue
            
            if num_batches == 0:
                print(f"      ❌ No successful batches in epoch {epoch}")
                continue
                
            avg_train_loss = total_loss / num_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                try:
                    # Process validation data in batches
                    for i in range(0, len(X_val), self.batch_size):
                        batch_X_val = X_val[i:i+self.batch_size]
                        batch_y_val = y_val[i:i+self.batch_size]
                        
                        # Prepare input based on model type
                        if model_name == 'EEGNet':
                            model_input_val = batch_X_val
                        elif model_name == 'ExtendedSemanticEEGNet':
                            # Resample to 500Hz/1500 samples and keep 4D
                            target_samples = 1500
                            current_samples = batch_X_val.shape[-1]
                            if current_samples != target_samples:
                                model_input_val = torch.nn.functional.interpolate(
                                    batch_X_val.unsqueeze(1),
                                    size=(batch_X_val.shape[1], target_samples),
                                    mode='bilinear',
                                    align_corners=False
                                )  # Keep 4D: (batch, 1, channels, time)
                            else:
                                model_input_val = batch_X_val.unsqueeze(1)  # (batch, 1, channels, time)
                        else:
                            # Other braindecode models
                            if len(batch_X_val.shape) == 3:  # (batch, channels, time)
                                model_input_val = batch_X_val.unsqueeze(1)  # (batch, 1, channels, time)
                            else:
                                model_input_val = batch_X_val
                        
                        # Handle time dimension mismatch
                        target_time = 250  # 1s at 250Hz
                        current_time = model_input_val.shape[-1]  # 1751
                        
                        if current_time != target_time:
                            if current_time > target_time:
                                # Crop from center
                                start_idx = (current_time - target_time) // 2
                                model_input_val = model_input_val[:, :, :, start_idx:start_idx + target_time]
                            elif current_time < target_time:
                                # Pad with zeros
                                pad_size = target_time - current_time
                                model_input_val = F.pad(model_input_val, (0, pad_size), mode='constant', value=0)
                        
                        outputs = model(model_input_val)
                        
                        # Handle model output format
                        if model_name == 'ExtendedSemanticEEGNet':
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                        
                        # Ensure targets are proper shape for CrossEntropyLoss
                        if len(batch_y_val.shape) > 1:
                            batch_y_val = batch_y_val.squeeze()
                        
                        loss = criterion(outputs, batch_y_val)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y_val.size(0)
                        val_correct += (predicted == batch_y_val).sum().item()
                    
                    avg_val_loss = val_loss / (len(X_val) // self.batch_size + 1)
                    val_accuracy = val_correct / val_total
                    
                    # Update training history
                    training_history['epoch'].append(epoch + 1)
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
                        best_model_state = model.state_dict().copy()
                        best_epoch = epoch + 1
                    else:
                        patience_counter += 1
                    
                    # Print progress
                    if (epoch + 1) % 20 == 0 or epoch < 10:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"      Epoch [{epoch+1}/{config['max_epochs']}] "
                              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                              f"Val Acc: {val_accuracy:.3f}, LR: {current_lr:.2e}, "
                              f"Patience: {patience_counter}/{config['patience']}")
                    
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
        
        # Save best model checkpoint
        model_save_path = self.models_dir / f"best_model_kor_{self._current_subject_id}_{model_name}.pth"
        torch.save({
            'model_state_dict': best_model_state if best_model_state is not None else model.state_dict(),
            'model_name': model_name,
            'subject_id': self._current_subject_id,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'config': config,
            'n_classes': self.n_classes,
            'n_channels': self.n_channels,
            'input_window_samples': self.input_window_samples,
            'sfreq': self.sfreq,
            'band_defs': model.band_defs if hasattr(model, 'band_defs') else None,
            'kernel_lengths': model.kernel_lengths if hasattr(model, 'kernel_lengths') else None
        }, model_save_path)
        print(f"      💾 Saved best model to: {model_save_path.name}")
        
        # Save training history to CSV
        history_save_path = self.training_history_dir / f"training_history_kor_{self._current_subject_id}_{model_name}.csv"
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(history_save_path, index=False)
        print(f"      📊 Saved training history to: {history_save_path.name}")
        
        # Final evaluation on test set
        model.eval()
        all_preds = []
        all_probs = []
        all_band_weights = []  # Collect band attention weights
        all_embeddings = []  # Collect latent embeddings (glob) for UMAP/t-SNE
        all_temporal_attention = []  # Collect temporal attention weights for Band×Time heatmap
        
        with torch.no_grad():
            try:
                # Process test data in batches
                for i in range(0, len(X_test), self.batch_size):
                    batch_X = X_test[i:i+self.batch_size]
                    
                    # Prepare input based on model type
                    if model_name == 'EEGNet':
                        model_input_test = batch_X
                    elif model_name == 'ExtendedSemanticEEGNet':
                        # Resample to 500Hz/1500 samples and keep 4D
                        target_samples = 1500
                        current_samples = batch_X.shape[-1]
                        if current_samples != target_samples:
                            model_input_test = torch.nn.functional.interpolate(
                                batch_X.unsqueeze(1),
                                size=(batch_X.shape[1], target_samples),
                                mode='bilinear',
                                align_corners=False
                            )  # Keep 4D: (batch, 1, channels, time)
                        else:
                            model_input_test = batch_X.unsqueeze(1)  # (batch, 1, channels, time)
                    else:
                        # Other braindecode models
                        if len(batch_X.shape) == 3:  # (batch, channels, time)
                            model_input_test = batch_X.unsqueeze(1)  # (batch, 1, channels, time)
                        else:
                            model_input_test = batch_X
                    
                    # Handle time dimension mismatch
                    target_time = 250  # 1s at 250Hz
                    current_time = model_input_test.shape[-1]  # 1751
                    
                    if current_time != target_time:
                        if current_time > target_time:
                            # Crop from center
                            start_idx = (current_time - target_time) // 2
                            model_input_test = model_input_test[:, :, :, start_idx:start_idx + target_time]
                        elif current_time < target_time:
                            # Pad with zeros
                            pad_size = target_time - current_time
                            model_input_test = F.pad(model_input_test, (0, pad_size), mode='constant', value=0)
                    
                    outputs = model(model_input_test)
                    
                    # Handle model output format
                    if model_name == 'ExtendedSemanticEEGNet':
                        # For ExtendedSemanticEEGNet, request attention weights during test
                        if hasattr(model, 'forward') and 'return_attention' in model.forward.__code__.co_varnames:
                            outputs = model(model_input_test, return_attention=True)
                        
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
                    else:
                        logits = outputs
                        glob = None
                        band_weights = None
                        temporal_attention = None
                    
                    # Get semantic predictions and probabilities
                    probs = torch.softmax(logits, dim=1)
                    sem_preds = torch.argmax(logits, dim=1)
                    all_preds.extend(sem_preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    if band_weights is not None:
                        all_band_weights.extend(band_weights.cpu().numpy())
                    if glob is not None:
                        all_embeddings.extend(glob.cpu().numpy())
                    if temporal_attention is not None:
                        all_temporal_attention.extend(temporal_attention.cpu().numpy())
                
            except Exception as e:
                print(f"    ❌ Error in evaluation: {e}")
                return None
        
        # Calculate metrics
        y_test_np = y_test.cpu().numpy()
        semantic_accuracy = accuracy_score(y_test_np, all_preds)
        
        # Calculate additional metrics
        additional_metrics = self.calculate_additional_metrics(y_test_np, all_preds, np.array(all_probs))
        
        # Save test results for confusion matrix and band weight visualization
        test_results_save_path = self.test_results_dir / f"test_results_kor_{self._current_subject_id}_{model_name}.npz"
        np.savez(
            test_results_save_path,
            y_true=y_test_np,
            y_pred=np.array(all_preds),
            y_probs=np.array(all_probs),
            embeddings=np.array(all_embeddings) if all_embeddings else None,
            band_weights=np.array(all_band_weights) if all_band_weights else None,
            temporal_attention=np.array(all_temporal_attention) if all_temporal_attention else None,
            model_name=model_name,
            subject_id=self._current_subject_id
        )
        print(f"      💾 Saved test results to: {test_results_save_path.name}")
        
        result = {
            'model': model_name,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'semantic_accuracy': semantic_accuracy,
            'macro_f1': additional_metrics['macro_f1'],
            'top2_accuracy': additional_metrics['top2_accuracy'],
            'top3_accuracy': additional_metrics['top3_accuracy'],
            'n_classes': len(torch.unique(y_train)),
            'model_params': model_params,
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'final_val_acc': training_history['val_acc'][-1] if training_history['val_acc'] else 0.0
        }
        
        print(f"    ✅ {model_name} Results:")
        print(f"      📈 Test Accuracy: {semantic_accuracy:.3f}")
        print(f"      🏃 Epochs Trained: {epoch + 1}/{config['max_epochs']}")
        print(f"      📊 Best Val Loss: {best_val_loss:.4f}")
        
        return result
    
    def run_all_subjects(self):
        """Execute analysis for all subjects with Korean data"""
        print(f"\n🚀 Starting Baseline Korean Analysis for All Subjects")
        
        # Get all subject IDs with Korean data
        subjects = set()
        
        # Find files for Korean data
        files = self.loader.find_eeg_files(
            language='kor',
            tasks=self.selected_tasks
        )
        
        for file_path in files:
            filename = Path(file_path).stem
            subject_id = filename.split('_')[1]
            subjects.add(subject_id)
        
        subjects = sorted(list(subjects))
        print(f"📋 Found {len(subjects)} subjects with Korean data: {subjects}")
        
        all_results = {}
        
        for subject_id in subjects:
            try:
                subject_results = self.run_subject_analysis(subject_id)
                all_results[subject_id] = subject_results
            except Exception as e:
                print(f"❌ Error processing subject {subject_id}: {e}")
        
        # Summarize results
        self._summarize_results(all_results)
        
        return all_results
    
    def run_first_subject_test(self):
        """Execute analysis for the first subject only (for testing)"""
        print(f"\n🧪 Starting Baseline Korean Analysis for First Subject (Testing Mode)")
        
        # Get all subject IDs with Korean data
        subjects = set()
        
        files = self.loader.find_eeg_files(
            language='kor',
            tasks=self.selected_tasks
        )
        
        for file_path in files:
            filename = Path(file_path).stem
            subject_id = filename.split('_')[1]
            subjects.add(subject_id)
        
        subjects = sorted(list(subjects))
        if not subjects:
            print("❌ No subjects found!")
            return None
            
        # Test only the first subject
        first_subject = subjects[0]
        print(f"🎯 Testing with first subject: {first_subject}")
        print(f"📋 Total subjects available: {len(subjects)}")
        
        try:
            subject_results = self.run_subject_analysis(first_subject)
            all_results = {first_subject: subject_results}
            
            # Summarize results
            self._summarize_results(all_results)
            
            return all_results
            
        except Exception as e:
            print(f"❌ Error processing subject {first_subject}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _summarize_results(self, all_results):
        """Summarize and print results for all models"""
        print(f"\n{'='*80}")
        print(f"BASELINE KOREAN: BRAINDECODE MODELS COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Process results for each subject
        for subject_id, subject_results in all_results.items():
            print(f"\n📊 Subject {subject_id} Results:")
            print(f"{'='*50}")
            
            if subject_results is None:
                print(f"  ❌ No data available for subject {subject_id}")
                continue
            
            model_summaries = []
            
            # Summarize each model's performance
            for model_name in self.model_names:
                if model_name in subject_results and subject_results[model_name]:
                    result = subject_results[model_name]
                    model_summaries.append({
                        'model': model_name,
                        'accuracy': result['semantic_accuracy'],
                        'n_params': result['model_params'],
                        'epochs': result['epochs_trained'],
                        'val_loss': result['best_val_loss']
                    })
                    
                    print(f"  🤖 {model_name:<15}: {result['semantic_accuracy']:.3f} "
                          f"({result['model_params']:,} params, {result['epochs_trained']} epochs)")
                else:
                    print(f"  ❌ {model_name:<15}: Failed")
            
            # Sort by performance and show ranking
            if model_summaries:
                model_summaries.sort(key=lambda x: x['accuracy'], reverse=True)
                print(f"\n🏆 Ranking for Subject {subject_id}:")
                for i, summary in enumerate(model_summaries, 1):
                    print(f"  {i}. {summary['model']:<15}: {summary['accuracy']:.3f}")
        
        print(f"\n{'='*80}")
        
        # Save results to file
        self._save_results_to_file(all_results)
    
    def _save_results_to_file(self, all_results):
        """Save experiment results to text and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed text report
        txt_filename = self.results_dir / f"BaselineKorean_Results_{timestamp}.txt"
        csv_filename = self.results_dir / f"BaselineKorean_Results_{timestamp}.csv"
        
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BASELINE KOREAN RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Tested: {', '.join(self.model_names)}\n")
            f.write(f"Data Split: 7:1.5:1.5 (train:valid:test)\n")
            f.write(f"Language: Korean only\n")
            f.write("-"*80 + "\n\n")
            
            # Process each subject
            for subject_id, subject_results in all_results.items():
                f.write(f"SUBJECT {subject_id} RESULTS:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Model':<15} {'Accuracy':<10} {'Params':<12} {'Epochs':<8} {'ValLoss':<10}\n")
                f.write("-"*80 + "\n")
                
                model_summaries = []
                
                for model_name in self.model_names:
                    if model_name in subject_results and subject_results[model_name]:
                        result = subject_results[model_name]
                        
                        f.write(f"{model_name:<15} {result['semantic_accuracy']:<10.3f} {result['model_params']:<12,} ")
                        f.write(f"{result['epochs_trained']:<8} {result['best_val_loss']:<10.4f}\n")
                        
                        model_summaries.append({
                            'model': model_name,
                            'accuracy': result['semantic_accuracy']
                        })
                    else:
                        f.write(f"{model_name:<15} {'FAIL':<10} {'N/A':<12} {'N/A':<8} {'N/A':<10}\n")
                
                # Ranking
                if model_summaries:
                    model_summaries.sort(key=lambda x: x['accuracy'], reverse=True)
                    f.write(f"\nRANKING:\n")
                    for i, summary in enumerate(model_summaries, 1):
                        f.write(f"  {i}. {summary['model']}: {summary['accuracy']:.3f}\n")
                
                f.write("\n" + "="*60 + "\n\n")
        
        # Create CSV file for easy data analysis
        csv_data = []
        csv_headers = [
            'Subject', 'Model', 'Semantic_Accuracy', 'Macro_F1', 'Top2_Accuracy', 'Top3_Accuracy',
            'Train_Size', 'Val_Size', 'Test_Size', 'Model_Params',
            'Epochs_Trained', 'Best_Val_Loss', 'Final_Val_Acc'
        ]
        
        for subject_id, subject_results in all_results.items():
            if subject_results is not None:
                for model_name, result in subject_results.items():
                    if result:
                        csv_data.append([
                            subject_id,
                            model_name,
                            result['semantic_accuracy'],
                            result['macro_f1'],
                            result['top2_accuracy'],
                            result['top3_accuracy'],
                            result['train_size'],
                            result['val_size'],
                            result['test_size'],
                            result['model_params'],
                            result['epochs_trained'],
                            result['best_val_loss'],
                            result['final_val_acc']
                        ])
                    csv_data.append([
                        subject_id,
                        model_name,
                        result['semantic_accuracy'],
                        result['train_size'],
                        result['val_size'],
                        result['test_size'],
                        result['model_params'],
                        result['epochs_trained'],
                        result['best_val_loss'],
                        result['final_val_acc']
                    ])
        
        # Add AVERAGE and STD rows
        if csv_data:
            # Convert to DataFrame for easy statistics calculation
            df = pd.DataFrame(csv_data, columns=csv_headers)
            
            # Calculate statistics for numeric columns only
            numeric_cols = ['Semantic_Accuracy', 'Train_Size', 'Val_Size', 'Test_Size', 
                          'Model_Params', 'Epochs_Trained', 'Best_Val_Loss', 'Final_Val_Acc']
            
            # Calculate average row
            avg_row = ['AVERAGE', 'All Models']
            for col in numeric_cols:
                if col in df.columns:
                    avg_row.append(df[col].mean())
                else:
                    avg_row.append('')
            
            # Calculate std row
            std_row = ['STD', 'All Models']
            for col in numeric_cols:
                if col in df.columns:
                    std_row.append(df[col].std())
                else:
                    std_row.append('')
            
            csv_data.append(avg_row)
            csv_data.append(std_row)
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            writer.writerows(csv_data)
        
        print(f"\n💾 Results saved to:")
        print(f"  📄 Detailed report: {txt_filename}")
        print(f"  📊 CSV data: {csv_filename}")

if __name__ == "__main__":
    # Execute Baseline Korean
    baseline_kor = BaselineKorean()
    
    print("🧪 Running Baseline Korean Analysis")
    print("🎯 Testing mode: First subject only")
    print(f"🤖 Models to test: {', '.join(baseline_kor.model_names)}")
    
    # Test with first subject
    # results = baseline_kor.run_first_subject_test()
    results = baseline_kor.run_all_subjects()
    
    if results:
        print("\n✅ Test completed successfully!")
        print("🔄 To run all subjects, use: baseline_kor.run_all_subjects()")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
