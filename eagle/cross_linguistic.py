"""
Track C: Subject-Dependent Cross-Lingual Transfer

SD-K→E: Training with Korean data from one subject, testing with English data from the same subject
SD-E→K: Symmetric opposite direction
Purpose: Test whether models can transfer across languages within the same subject,
providing evidence for cross-linguistic patterns in individual brains
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

from eagle import eagle_w_labram

import warnings
warnings.filterwarnings('ignore')

class TrackC_ZeroShot:
    def __init__(self, data_path="D:\\SemanticDecoding\\Dataset\\", 
                 use_target_val=False, target_val_ratio=0.1):
        """
        Args:
            data_path: Path to dataset
            use_target_val: Whether to use target language data for validation
            target_val_ratio: Ratio of target language data to use for validation (e.g., 0.1 = 10%)
        """
        self.data_path = data_path
        self.loader = EEGDataLoader(data_path)
        # Use common parameters from utils
        self.selected_tasks = SELECTED_TASKS  # task1, task2, task4 only
        self.stimulus_mapping = STIMULUS_MAPPING
        self.epoch_tmin = EPOCH_TMIN
        self.epoch_tmax = EPOCH_TMAX
        
        # Target validation configuration
        self.use_target_val = use_target_val
        self.target_val_ratio = target_val_ratio
        
        # EEG data configuration
        self.n_channels = 64
        self.n_classes = 15  # Semantic classes for tasks 1, 2, 4
        self.input_window_samples = 1500  # 3.0s * 500Hz = 1500 samples
        self.sfreq = 500.0
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Print target validation configuration
        if self.use_target_val:
            print(f"🔧 Target Validation: Enabled ({self.target_val_ratio*100:.1f}% of target data)")
        else:
            print(f"🔧 Target Validation: Disabled (zero-shot)")
        
        # Training hyperparameters for ExtendedSemanticEEGNet
        self.model_config = {
            'max_epochs': 300,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'patience': 40,
            'warmup_epochs': 20,
            'scheduler_factor': 0.5,
            'scheduler_patience': 25,
            'batch_size': 16
        }
        
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
        
    def find_available_subjects(self):
        """Find available subjects without loading data"""
        print(f"\n🔍 Finding available subjects...")
        
        # Collect all subject IDs from both languages
        kor_files = self.loader.find_eeg_files(language='kor', tasks=self.selected_tasks)
        eng_files = self.loader.find_eeg_files(language='eng', tasks=self.selected_tasks)
        
        kor_subjects = set()
        eng_subjects = set()
        
        for file_path in kor_files:
            filename = Path(file_path).stem
            subject_id = filename.split('_')[1]
            kor_subjects.add(subject_id)
        
        for file_path in eng_files:
            filename = Path(file_path).stem
            subject_id = filename.split('_')[1]
            eng_subjects.add(subject_id)
        
        # Find subjects with both languages
        subjects_with_both = kor_subjects.intersection(eng_subjects)
        
        print(f"📋 Found subjects:")
        print(f"  🇰🇷 Korean: {len(kor_subjects)} subjects")
        print(f"  🇺🇸 English: {len(eng_subjects)} subjects")
        print(f"  ✅ Both languages: {len(subjects_with_both)} subjects")
        print(f"     {sorted(list(subjects_with_both))}")
        
        return sorted(list(subjects_with_both))
    
    def load_single_subject(self, subject_id):
        """Load data for a single subject (both languages) with session metadata preserved"""
        print(f"\n📂 Loading subject: {subject_id}")
        
        subject_data = {'kor': None, 'eng': None}
        
        try:
            all_epochs, session_info = load_subject_sessions_data(self.loader, subject_id, self.selected_tasks)
            
            # Extract English and Korean data
            eng_epochs = all_epochs.get('eng', [])
            kor_epochs = all_epochs.get('kor', [])
            
            # Combine epochs for each language while preserving session metadata
            if kor_epochs:
                # Add session date metadata to each epoch set before combining
                for epoch_set in kor_epochs:
                    # Extract date from first file in this session
                    if hasattr(epoch_set, 'metadata') and 'filename' in epoch_set.metadata.columns:
                        first_filename = epoch_set.metadata['filename'].iloc[0]
                        session_date = first_filename.split('_')[0]
                        epoch_set.metadata['session_date'] = session_date
                
                kor_combined = mne.concatenate_epochs(kor_epochs)
                subject_data['kor'] = kor_combined
                
                # Print session dates
                if 'session_date' in kor_combined.metadata.columns:
                    unique_sessions = kor_combined.metadata['session_date'].unique()
                    print(f"  ✅ KOR: {len(kor_combined)} epochs from {len(unique_sessions)} sessions {list(unique_sessions)}")
                else:
                    print(f"  ✅ KOR: {len(kor_combined)} epochs")
            
            if eng_epochs:
                # Add session date metadata to each epoch set before combining
                for epoch_set in eng_epochs:
                    if hasattr(epoch_set, 'metadata') and 'filename' in epoch_set.metadata.columns:
                        first_filename = epoch_set.metadata['filename'].iloc[0]
                        session_date = first_filename.split('_')[0]
                        epoch_set.metadata['session_date'] = session_date
                
                eng_combined = mne.concatenate_epochs(eng_epochs)
                subject_data['eng'] = eng_combined
                
                # Print session dates
                if 'session_date' in eng_combined.metadata.columns:
                    unique_sessions = eng_combined.metadata['session_date'].unique()
                    print(f"  ✅ ENG: {len(eng_combined)} epochs from {len(unique_sessions)} sessions {list(unique_sessions)}")
                else:
                    print(f"  ✅ ENG: {len(eng_combined)} epochs")
                
        except Exception as e:
            print(f"  ❌ Error loading subject {subject_id}: {e}")
            return None
        
        return subject_data
    
    def free_subject_data(self, subject_data):
        """Explicitly free memory for subject data"""
        if subject_data:
            del subject_data
        torch.cuda.empty_cache()  # Clear GPU cache if using CUDA
        import gc
        gc.collect()  # Force garbage collection
    
    def create_model(self):
        """Create ExtendedSemanticEEGNet model"""
        model = ExtendedSemanticEEGNet(
            nb_classes=self.n_classes,
            Chans=self.n_channels,
            Samples=1500,  # 3.0s at 500Hz (model requirement)
            sfreq=500.0,
            dropoutRate=0.2,
            F1=16,
            D=2,
            gamma_F1=32,
            reduced_dim=48,
            band_defs=((8, 30), (30, 60), (60, 80)),
            use_transformer=True
        )
        return model.to(self.device)
    
    def prepare_data_tensors(self, epochs):
        """Convert MNE epochs to PyTorch tensors"""
        # Get EEG data and labels
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        y_semantic_raw = epochs.metadata['class_label'].values
        
        # Remap semantic labels to 0-14 range
        y_semantic = np.copy(y_semantic_raw)
        task_nums = epochs.metadata['task'].values
        
        for i, (label, task) in enumerate(zip(y_semantic_raw, task_nums)):
            if task == 4:  # task4 labels need remapping from 15-19 to 10-14
                y_semantic[i] = label - 5
        
        # Validate semantic labels
        if np.min(y_semantic) < 0 or np.max(y_semantic) >= 15:
            print(f"    ⚠️  Warning: Invalid semantic labels - min: {np.min(y_semantic)}, max: {np.max(y_semantic)}")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_semantic_tensor = torch.LongTensor(y_semantic)
        
        return X_tensor, y_semantic_tensor
    
    def extract_session_balanced_validation(self, epochs, val_ratio=0.1):
        """Extract validation set with equal proportion from each session
        
        Args:
            epochs: MNE epochs object with 'session_date' in metadata
            val_ratio: Ratio of data to extract from each session
            
        Returns:
            val_indices, remaining_indices: Arrays of indices for validation and remaining data
        """
        if 'session_date' not in epochs.metadata.columns:
            print(f"    ⚠️  No session_date metadata, using random split")
            # Fallback to random split
            n_samples = len(epochs)
            n_val = int(n_samples * val_ratio)
            all_indices = np.arange(n_samples)
            np.random.shuffle(all_indices)
            return all_indices[:n_val], all_indices[n_val:]
        
        # Get unique sessions
        session_dates = epochs.metadata['session_date'].values
        unique_sessions = np.unique(session_dates)
        
        print(f"    📅 Extracting validation from {len(unique_sessions)} sessions: {list(unique_sessions)}")
        
        val_indices = []
        remaining_indices = []
        
        # Extract validation from each session with equal proportion
        for session in unique_sessions:
            session_mask = session_dates == session
            session_indices = np.where(session_mask)[0]
            
            # Calculate validation size for this session
            n_session = len(session_indices)
            n_val_session = max(1, int(n_session * val_ratio))
            
            # Randomly select validation indices from this session
            np.random.seed(42)
            np.random.shuffle(session_indices)
            
            val_indices.extend(session_indices[:n_val_session])
            remaining_indices.extend(session_indices[n_val_session:])
            
            print(f"      • {session}: {n_val_session}/{n_session} epochs to validation")
        
        return np.array(val_indices), np.array(remaining_indices)

    
    def run_zero_shot_analysis(self):
        """Execute Subject-Dependent Cross-Lingual analysis with memory-efficient loading"""
        print(f"\n{'='*60}")
        print(f"TRACK C: Subject-Dependent Cross-Lingual Analysis")
        print(f"{'='*60}")
        
        # Find available subjects (without loading data)
        subjects_with_both = self.find_available_subjects()
        
        if len(subjects_with_both) < 1:
            print(f"❌ No subjects with both language data for subject-dependent analysis")
            return None
        
        results = []
        
        # Process each subject one at a time
        for idx, subject in enumerate(subjects_with_both, 1):
            print(f"\n{'='*60}")
            print(f"Processing Subject {idx}/{len(subjects_with_both)}: {subject}")
            print(f"{'='*60}")
            
            # Load this subject's data
            subject_data = self.load_single_subject(subject)
            
            if subject_data is None:
                print(f"  ⚠️  Skipping subject {subject} - loading failed")
                continue
            
            if subject_data['kor'] is None or subject_data['eng'] is None:
                print(f"  ⚠️  Skipping subject {subject} - missing language data")
                self.free_subject_data(subject_data)
                continue
            
            # SD-K→E: Train on Korean, test on English (same subject)
            print(f"\n🇰🇷➡️🇺🇸 Korean → English for subject {subject}")
            result_kor_eng = self._train_and_evaluate_single_subject(
                subject_data['kor'], subject_data['eng'], 'kor', 'eng', subject
            )
            if result_kor_eng:
                results.append(result_kor_eng)
            
            # SD-E→K: Train on English, test on Korean (same subject)
            print(f"\n🇺🇸➡️🇰🇷 English → Korean for subject {subject}")
            result_eng_kor = self._train_and_evaluate_single_subject(
                subject_data['eng'], subject_data['kor'], 'eng', 'kor', subject
            )
            if result_eng_kor:
                results.append(result_eng_kor)
            
            # Free memory after processing this subject
            print(f"\n🗑️  Freeing memory for subject {subject}")
            self.free_subject_data(subject_data)
        
        # Summarize results
        self._summarize_zero_shot_results(results)
        
        return results
    
    def _train_and_evaluate_single_subject(self, train_epochs, test_epochs, 
                                           train_lang, test_lang, subject):
        """Train and evaluate for a single subject (one direction)"""
        print(f"  📈 Training ({train_lang.upper()}): {len(train_epochs)} epochs")
        print(f"  📊 Testing ({test_lang.upper()}): {len(test_epochs)} epochs")
        
        # Check if we have enough data
        if len(train_epochs) < 10 or len(test_epochs) < 5:
            print(f"  ⚠️  Insufficient data (train: {len(train_epochs)}, test: {len(test_epochs)})")
            return None
        
        # Train and evaluate model
        result = self._train_and_evaluate_zero_shot(
            train_epochs, test_epochs, train_lang, test_lang, subject
        )
        
        return result
    
    def _train_and_evaluate_zero_shot(self, train_epochs, test_epochs, 
                                     train_lang, test_lang, test_subject):
        """Train and evaluate zero-shot model with ExtendedSemanticEEGNet"""
        print(f"  🤖 Training ExtendedSemanticEEGNet: {train_lang.upper()} → {test_lang.upper()}")
        
        # Prepare training data
        X_train_full, y_train_full = self.prepare_data_tensors(train_epochs)
        
        # Split training data: 85% train, 15% validation (source language)
        X_train, X_val_src, y_train, y_val_src = train_test_split(
            X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
        )
        
        print(f"    📊 Source ({train_lang.upper()}) - Train: {len(X_train)}, Val: {len(X_val_src)}")
        
        # Handle target language data based on use_target_val option
        if self.use_target_val:
            print(f"    🎯 Target validation: Enabled ({self.target_val_ratio*100:.1f}%)")
            
            # Extract session-balanced validation from target language
            val_indices, test_indices = self.extract_session_balanced_validation(
                test_epochs, val_ratio=self.target_val_ratio
            )
            
            # Get validation data from target language
            test_epochs_val = test_epochs[val_indices]
            test_epochs_test = test_epochs[test_indices]
            
            X_val_tgt, y_val_tgt = self.prepare_data_tensors(test_epochs_val)
            X_test, y_test = self.prepare_data_tensors(test_epochs_test)
            
            # Combine source and target validation sets
            X_val = torch.cat([X_val_src, X_val_tgt], dim=0)
            y_val = torch.cat([y_val_src, y_val_tgt], dim=0)
            
            print(f"    📊 Target ({test_lang.upper()}) - Val: {len(X_val_tgt)}, Test: {len(X_test)}")
            print(f"    📊 Combined Validation: {len(X_val)} (Source: {len(X_val_src)}, Target: {len(X_val_tgt)})")
        else:
            print(f"    🎯 Target validation: Disabled (pure zero-shot)")
            
            # Use only source language validation, all target data for testing
            X_val = X_val_src
            y_val = y_val_src
            X_test, y_test = self.prepare_data_tensors(test_epochs)
            
            print(f"    📊 Validation: {len(X_val)} (Source only)")
            print(f"    📊 Test: {len(X_test)} (Target, full dataset)")
        
        # Check class distribution
        train_classes = torch.unique(y_train)
        test_classes = torch.unique(y_test)
        
        print(f"    📊 Train: {len(X_train)} epochs, Val: {len(X_val)} epochs, Test: {len(X_test)} epochs")
        print(f"    📊 Train classes: {len(train_classes)}/15, Test classes: {len(test_classes)}/15")
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        # Create model
        model = self.create_model()
        model_params = sum(p.numel() for p in model.parameters())
        print(f"    📊 Model parameters: {model_params:,}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=self.model_config['scheduler_factor'],
            patience=self.model_config['scheduler_patience'],
            verbose=False,
            min_lr=1e-7
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.model_config['batch_size'], 
            shuffle=True
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        print(f"    🏋️  Training for max {self.model_config['max_epochs']} epochs...")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(self.model_config['max_epochs']):
            # Training phase
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Apply warmup learning rate
            if epoch < self.model_config['warmup_epochs']:
                warmup_lr = self.model_config['learning_rate'] * (epoch + 1) / self.model_config['warmup_epochs']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Prepare 4D input for ExtendedSemanticEEGNet
                # Data is already 500Hz/1500 samples, just add channel dimension
                model_input = batch_X.unsqueeze(1)  # (batch, 1, channels, time)
                
                # Forward pass - training mode (no attention extraction)
                outputs = model(model_input, return_attention=False)
                
                # Handle tuple output (logits, features)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Use only logits
                
                # Compute loss
                if len(batch_y.shape) > 1:
                    batch_y = batch_y.squeeze()
                
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches == 0:
                continue
            
            avg_train_loss = total_loss / num_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), self.model_config['batch_size']):
                    batch_X_val = X_val[i:i+self.model_config['batch_size']]
                    batch_y_val = y_val[i:i+self.model_config['batch_size']]
                    
                    # Prepare 4D input (data is already 500Hz/1500 samples)
                    model_input_val = batch_X_val.unsqueeze(1)  # (batch, 1, channels, time)
                    
                    # Forward pass - validation mode (no attention extraction)
                    outputs = model(model_input_val, return_attention=False)
                    
                    # Handle tuple output
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Compute validation loss
                    if len(batch_y_val.shape) > 1:
                        batch_y_val = batch_y_val.squeeze()
                    
                    loss = criterion(outputs, batch_y_val)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y_val.size(0)
                    val_correct += (predicted == batch_y_val).sum().item()
                
                avg_val_loss = val_loss / (len(X_val) // self.model_config['batch_size'] + 1)
                val_accuracy = val_correct / val_total
                
                # Update training history
                training_history['train_loss'].append(avg_train_loss)
                training_history['val_loss'].append(avg_val_loss)
                training_history['val_acc'].append(val_accuracy)
                
                # Learning rate scheduling (after warmup)
                if epoch >= self.model_config['warmup_epochs']:
                    scheduler.step(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Print progress
                if (epoch + 1) % 20 == 0 or epoch < 5:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"      Epoch [{epoch+1}/{self.model_config['max_epochs']}] "
                          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                          f"Val Acc: {val_accuracy:.3f}, LR: {current_lr:.2e}")
                
                # Early stopping
                if patience_counter >= self.model_config['patience']:
                    print(f"      🛑 Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"      ✅ Loaded best model (val_loss: {best_val_loss:.4f})")
            
            # Save best model to disk
            direction_str = f"{train_lang}2{test_lang}"
            model_filename = self.models_dir / f"best_model_trackc_{test_subject}_{direction_str}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'best_val_loss': best_val_loss,
                'direction': direction_str,
                'subject': test_subject,
                'use_target_val': self.use_target_val,
                'target_val_ratio': self.target_val_ratio,
                'model_config': {
                    'n_classes': self.n_classes,
                    'n_channels': self.n_channels,
                    'samples': self.input_window_samples,
                    'sfreq': self.sfreq,
                    'band_defs': model.band_defs,
                    'kernel_lengths': model.kernel_lengths
                }
            }, model_filename)
            print(f"      💾 Model saved: {model_filename.name}")
            
            # Save training history
            history_filename = self.training_history_dir / f"training_history_trackc_{test_subject}_{direction_str}.csv"
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
        all_probs = []  # Also collect probabilities
        all_band_weights = []  # Collect band attention weights
        all_embeddings = []  # Collect latent embeddings (glob) for UMAP/t-SNE
        all_temporal_attention = []  # Collect temporal attention weights for Band×Time heatmap
        
        with torch.no_grad():
            for i in range(0, len(X_test), self.model_config['batch_size']):
                batch_X = X_test[i:i+self.model_config['batch_size']]
                
                # Prepare 4D input (data is already 500Hz/1500 samples)
                model_input_test = batch_X.unsqueeze(1)  # (batch, 1, channels, time)
                
                # Forward - test mode with attention extraction
                outputs = model(model_input_test, return_attention=True)
                
                # Handle tuple output (logits, glob, band_weights, temporal_attention)
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
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())  # Use extend instead of append
                
                # Collect attention weights and embeddings
                if band_weights is not None:
                    all_band_weights.extend(band_weights.cpu().numpy())  # Use extend
                if glob is not None:
                    all_embeddings.extend(glob.cpu().numpy())  # Use extend
                if temporal_attention is not None:
                    all_temporal_attention.extend(temporal_attention.cpu().numpy())  # Use extend
        
        # Calculate test accuracy
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        y_true = y_test.cpu().numpy()
        test_accuracy = accuracy_score(y_true, all_preds)
        
        # Save test results for confusion matrix and band weight visualization
        direction_str = f"{train_lang}2{test_lang}"
        test_results_filename = self.test_results_dir / f"test_results_trackc_{test_subject}_{direction_str}.npz"
        np.savez(test_results_filename,
                 y_true=y_true,
                 y_pred=all_preds,
                 y_probs=all_probs,
                 embeddings=np.array(all_embeddings) if all_embeddings else None,
                 band_weights=np.array(all_band_weights) if all_band_weights else None,
                 temporal_attention=np.array(all_temporal_attention) if all_temporal_attention else None,
                 subject_id=test_subject,
                 direction=direction_str,
                 train_lang=train_lang,
                 test_lang=test_lang,
                 use_target_val=self.use_target_val,
                 target_val_ratio=self.target_val_ratio if self.use_target_val else 0)
        print(f"    💾 Test results saved: {test_results_filename.name}")
        
        result = {
            'direction': f"{train_lang.upper()}→{test_lang.upper()}",
            'subject': test_subject,
            'train_lang': train_lang,
            'test_lang': test_lang,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'val_src_size': len(X_val_src) if self.use_target_val else len(X_val),
            'val_tgt_size': len(X_val_tgt) if self.use_target_val else 0,
            'test_size': len(X_test),
            'train_classes': len(train_classes),
            'test_classes': len(test_classes),
            'accuracy': test_accuracy,
            'model_params': model_params,
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'use_target_val': self.use_target_val,
            'target_val_ratio': self.target_val_ratio if self.use_target_val else 0
        }
        
        print(f"    ✅ {train_lang.upper()}→{test_lang.upper()} Test Accuracy: {test_accuracy:.3f}")
        
        return result
    
    def _summarize_zero_shot_results(self, results):
        """Summarize subject-dependent cross-lingual results"""
        print(f"\n{'='*60}")
        print(f"TRACK C: SUBJECT-DEPENDENT CROSS-LINGUAL SUMMARY")
        print(f"{'='*60}")
        
        if not results:
            print("❌ No valid results to summarize")
            return
        
        # Analyze results by direction
        directions = {}
        for result in results:
            direction = result['direction']
            if direction not in directions:
                directions[direction] = []
            directions[direction].append(result['accuracy'])
        
        print(f"📊 Subject-Dependent Transfer Results:")
        
        for direction, accuracies in directions.items():
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"\n🔄 {direction}:")
            print(f"   • Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
            print(f"   • Best: {max(accuracies):.3f}")
            print(f"   • Worst: {min(accuracies):.3f}")
            print(f"   • N subjects: {len(accuracies)}")
        
        # Overall average
        all_accuracies = [r['accuracy'] for r in results]
        overall_mean = np.mean(all_accuracies)
        overall_std = np.std(all_accuracies)
        
        print(f"\n🎯 Overall Subject-Dependent Performance: {overall_mean:.3f} ± {overall_std:.3f}")
        
        # Individual results
        print(f"\n📋 Detailed Results:")
        for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            val_info = f"val: {result['val_size']}"
            if result.get('use_target_val', False):
                val_info = f"val: {result['val_size']} (src: {result['val_src_size']}, tgt: {result['val_tgt_size']})"
            
            print(f"  {result['direction']} {result['subject']}: {result['accuracy']:.3f} "
                  f"(train: {result['train_size']}, {val_info}, test: {result['test_size']}, "
                  f"epochs: {result['epochs_trained']})")
        
        print(f"{'='*60}")
        
        # Save results to file
        self._save_results_to_file(results)
    
    def _save_results_to_file(self, results):
        """Save experiment results to text and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed text report
        txt_filename = self.results_dir / f"TrackC_SubjectDependent_Results_{timestamp}.txt"
        csv_filename = self.results_dir / f"TrackC_SubjectDependent_Results_{timestamp}.csv"
        
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TRACK C: SUBJECT-DEPENDENT CROSS-LINGUAL RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: ExtendedSemanticEEGNet (EAGLE)\n")
            f.write(f"Analysis Type: Subject-Dependent (within-subject cross-lingual transfer)\n")
            if results and results[0].get('use_target_val', False):
                f.write(f"Target Validation: Enabled ({results[0]['target_val_ratio']*100:.1f}% of target data)\n")
            else:
                f.write(f"Target Validation: Disabled (pure zero-shot)\n")
            f.write("-"*80 + "\n\n")
            
            # Group by direction
            directions = {}
            for result in results:
                direction = result['direction']
                if direction not in directions:
                    directions[direction] = []
                directions[direction].append(result)
            
            for direction, dir_results in directions.items():
                f.write(f"\n{direction} RESULTS:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Subject':<15} {'Accuracy':<10} {'Train':<8} {'Val':<8} {'Test':<8} {'Epochs':<8}\n")
                f.write("-"*80 + "\n")
                
                for result in dir_results:
                    f.write(f"{result['subject']:<15} {result['accuracy']:<10.3f} "
                           f"{result['train_size']:<8} {result['val_size']:<8} "
                           f"{result['test_size']:<8} {result['epochs_trained']:<8}\n")
                
                # Direction summary
                accuracies = [r['accuracy'] for r in dir_results]
                f.write("-"*80 + "\n")
                f.write(f"Mean: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}\n")
                f.write(f"Best: {max(accuracies):.3f}, Worst: {min(accuracies):.3f}\n")
                f.write(f"N subjects: {len(dir_results)}\n")
            
            # Overall summary
            all_accuracies = [r['accuracy'] for r in results]
            f.write("\n" + "="*80 + "\n")
            f.write(f"OVERALL SUBJECT-DEPENDENT PERFORMANCE\n")
            f.write("="*80 + "\n")
            f.write(f"Mean Accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}\n")
            f.write(f"Total Experiments: {len(results)}\n")
        
        # Create CSV file
        csv_data = []
        csv_headers = [
            'Direction', 'Subject', 'Train_Lang', 'Test_Lang',
            'Accuracy', 'Train_Size', 'Val_Size', 'Val_Src_Size', 'Val_Tgt_Size', 'Test_Size', 
            'Train_Classes', 'Test_Classes',
            'Model_Params', 'Epochs_Trained', 'Best_Val_Loss',
            'Use_Target_Val', 'Target_Val_Ratio'
        ]
        
        for result in results:
            csv_data.append([
                result['direction'],
                result['subject'],
                result['train_lang'],
                result['test_lang'],
                result['accuracy'],
                result['train_size'],
                result['val_size'],
                result.get('val_src_size', result['val_size']),
                result.get('val_tgt_size', 0),
                result['test_size'],
                result['train_classes'],
                result['test_classes'],
                result['model_params'],
                result['epochs_trained'],
                result['best_val_loss'],
                result.get('use_target_val', False),
                result.get('target_val_ratio', 0)
            ])
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            writer.writerows(csv_data)
            
            # Add summary statistics (average and std) by direction if there are valid results
            if csv_data:
                writer.writerow([])  # Empty row for separation
                
                # Group results by direction
                directions_data = {}
                for row in csv_data:
                    direction = row[0]  # Direction is first column
                    if direction not in directions_data:
                        directions_data[direction] = []
                    directions_data[direction].append(row)
                
                # Calculate statistics for each direction
                for direction, dir_rows in directions_data.items():
                    # Average row for this direction
                    avg_row = [f'AVG_{direction}', '']
                    # Numeric columns: Accuracy (4), Train_Size (5), Val_Size (6), etc.
                    numeric_cols = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # Accuracy to Best_Val_Loss
                    
                    for col_idx in numeric_cols:
                        values = [row[col_idx] for row in dir_rows if row[col_idx] != '']
                        if values:
                            avg_row.append(np.mean(values))
                        else:
                            avg_row.append('')
                    
                    # Add remaining columns
                    avg_row.extend(['', ''])  # Use_Target_Val, Target_Val_Ratio
                    writer.writerow(avg_row)
                    
                    # Standard deviation row for this direction
                    std_row = [f'STD_{direction}', '']
                    for col_idx in numeric_cols:
                        values = [row[col_idx] for row in dir_rows if row[col_idx] != '']
                        if values:
                            std_row.append(np.std(values, ddof=1) if len(values) > 1 else 0)
                        else:
                            std_row.append('')
                    
                    std_row.extend(['', ''])
                    writer.writerow(std_row)
                
                # Overall statistics (across all directions)
                writer.writerow([])  # Empty row
                
                # Overall average
                overall_avg_row = ['OVERALL_AVG', '']
                for col_idx in numeric_cols:
                    values = [row[col_idx] for row in csv_data if row[col_idx] != '']
                    if values:
                        overall_avg_row.append(np.mean(values))
                    else:
                        overall_avg_row.append('')
                overall_avg_row.extend(['', ''])
                writer.writerow(overall_avg_row)
                
                # Overall std
                overall_std_row = ['OVERALL_STD', '']
                for col_idx in numeric_cols:
                    values = [row[col_idx] for row in csv_data if row[col_idx] != '']
                    if values:
                        overall_std_row.append(np.std(values, ddof=1) if len(values) > 1 else 0)
                    else:
                        overall_std_row.append('')
                overall_std_row.extend(['', ''])
                writer.writerow(overall_std_row)
        
        print(f"\n💾 Results saved to:")
        print(f"  📄 Detailed report: {txt_filename}")
        print(f"  📊 CSV data (with summary stats by direction): {csv_filename}")
        print(f"  📂 Models directory: {self.models_dir}")
        print(f"  📈 Training history directory: {self.training_history_dir}")
        print(f"  📋 Test results directory: {self.test_results_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Track C: Zero-Shot Cross-Lingual Transfer')
    parser.add_argument('--direction', type=str, choices=['eng2kor', 'kor2eng', 'both'], 
                       default='both',
                       help='Transfer direction: eng2kor (English→Korean), kor2eng (Korean→English), or both')
    parser.add_argument('--test_subject', type=str, default=None,
                       help='Test with specific subject only (for quick testing)')
    parser.add_argument('--use_target_val', action='store_true',
                       help='Use target language data for validation (session-balanced sampling)')
    parser.add_argument('--target_val_ratio', type=float, default=0.1,
                       help='Ratio of target language data to use for validation (default: 0.1 = 10%%)')
    
    args = parser.parse_args()
    
    # Execute Track C
    track_c = TrackC_ZeroShot(
        use_target_val=args.use_target_val,
        target_val_ratio=args.target_val_ratio
    )
    
    if args.test_subject:
        print(f"🎯 Testing with single subject: {args.test_subject}")
        subjects_to_process = [args.test_subject]
    else:
        # Find all available subjects
        subjects_to_process = track_c.find_available_subjects()
    
    if not subjects_to_process:
        print("❌ No valid subjects for analysis")
    else:
        results = []
        
        # Process each subject one at a time
        for idx, subject in enumerate(subjects_to_process, 1):
            print(f"\n{'='*60}")
            print(f"Processing Subject {idx}/{len(subjects_to_process)}: {subject}")
            print(f"{'='*60}")
            
            # Load this subject's data
            subject_data = track_c.load_single_subject(subject)
            
            if subject_data is None or subject_data['kor'] is None or subject_data['eng'] is None:
                print(f"  ⚠️  Skipping subject {subject} - insufficient data")
                if subject_data:
                    track_c.free_subject_data(subject_data)
                continue
            
            # Process based on direction
            if args.direction in ['kor2eng', 'both']:
                print(f"\n🇰🇷➡️🇺🇸 Korean → English for subject {subject}")
                result = track_c._train_and_evaluate_single_subject(
                    subject_data['kor'], subject_data['eng'], 'kor', 'eng', subject
                )
                if result:
                    results.append(result)
            
            if args.direction in ['eng2kor', 'both']:
                print(f"\n🇺🇸➡️🇰🇷 English → Korean for subject {subject}")
                result = track_c._train_and_evaluate_single_subject(
                    subject_data['eng'], subject_data['kor'], 'eng', 'kor', subject
                )
                if result:
                    results.append(result)
            
            # Free memory after processing this subject
            print(f"\n🗑️  Freeing memory for subject {subject}")
            track_c.free_subject_data(subject_data)
        
        # Summarize results
        if results:
            track_c._summarize_zero_shot_results(results)
    
    print("\n✅ Track C Subject-Dependent Analysis Complete!")