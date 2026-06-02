"""
Cross-Linguistic EEG Data Analysis Utilities
============================================

This module contains common functions for cross-linguistic EEG data analysis,
specifically for Korean and English speech imagery tasks.

Author: EEG Analysis Team
Date: 2025-09-02
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict
import mne
from scipy import signal, stats
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


# Common experimental parameters for all tracks
SELECTED_TASKS = [1, 2, 4]  # task1, task2, task4 only
STIMULUS_MAPPING = {
    'Stimulus/S  1': 0, 'Stimulus/S  2': 1, 'Stimulus/S  3': 2, 'Stimulus/S  4': 3, 'Stimulus/S  5': 4
}
EPOCH_TMIN = 0.0  # No baseline correction - start from stimulus onset
EPOCH_TMAX = 3.0


def create_epochs_from_raw(raw):
    """
    Create epochs from raw EEG data with 15-class semantic labeling
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data object
        
    Returns:
    --------
    mne.Epochs or None
        Epochs object with metadata, or None if failed
    """
    try:
        # Debug: print filename for tracking
        filename = Path(raw.filenames[0]).stem if raw.filenames else "Unknown"
        # print(f"    Processing file: {filename}")
        
        # Find events from annotations
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        # print(f"      Found {len(events)} events, event_ids: {list(event_id.keys())}")
        
        # Filter only relevant stimulus events
        valid_events = []
        valid_event_ids = {}
        
        for event_name, event_code in event_id.items():
            if event_name in STIMULUS_MAPPING:
                valid_events.append([events[events[:, 2] == event_code]])
                valid_event_ids[event_name] = event_code
        
        # print(f"      Valid stimulus events: {valid_event_ids}")
        
        if not valid_event_ids:
            # print(f"      ❌ No valid stimulus events found in {filename}")
            return None
        
        # Select only valid events
        valid_event_mask = np.isin(events[:, 2], list(valid_event_ids.values()))
        filtered_events = events[valid_event_mask]
        
        # print(f"      Filtered {len(filtered_events)} valid events from {len(events)} total")
        
        if len(filtered_events) == 0:
            # print(f"      ❌ No events match stimulus mapping in {filename}")
            return None
        
        # Create epochs
        epochs = mne.Epochs(
            raw, filtered_events, 
            event_id=valid_event_ids,
            tmin=EPOCH_TMIN, 
            tmax=EPOCH_TMAX,
            baseline=None,  # No baseline correction for now
            preload=True,
            verbose=False
        )
        
        # print(f"      Created {len(epochs)} epochs")
        
        # Extract task information from filename
        filename = Path(raw.filenames[0]).stem
        task_num = int(filename.split('task')[1].split('_')[0])
        
        # Generate 15-class labels (task * 5 + stimulus) using events array
        labels = []
        stimulus_indices = []
        
        # Get event descriptions from the event_id mapping
        reverse_event_id = {v: k for k, v in epochs.event_id.items()}
        
        for event_code in epochs.events[:, 2]:  # Third column contains event codes
            event_name = reverse_event_id[event_code]
            stimulus_idx = STIMULUS_MAPPING[event_name]
            class_label = (task_num - 1) * 5 + stimulus_idx  # task1->0-4, task2->5-9, task4->10-14
            labels.append(class_label)
            stimulus_indices.append(stimulus_idx)
        
        # Add metadata with label information
        lang_code = 'kor' if '_kor' in filename else 'eng'
        epochs.metadata = pd.DataFrame({
            'task': task_num,
            'stimulus': stimulus_indices,
            'class_label': labels,
            'subject': filename.split('_')[1],
            'language': lang_code,
            'lang_code': lang_code  # Add lang_code for compatibility
        })
        
        # print(f"      ✅ Successfully created epochs for {filename}: {len(epochs)} epochs")
        return epochs
        
    except Exception as e:
        print(f"    ❌ Error creating epochs for {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_subject_sessions_data(loader, subject_id, tasks=None):
    """
    Load all session data for a specific subject
    
    Parameters:
    -----------
    loader : EEGDataLoader
        Data loader instance
    subject_id : str
        Subject identifier
    tasks : list, optional
        List of tasks to load (default: SELECTED_TASKS)
        
    Returns:
    --------
    tuple
        (all_epochs, session_info) where all_epochs is dict with 'kor'/'eng' keys
        containing lists of epochs, and session_info contains session indices
    """
    if tasks is None:
        tasks = SELECTED_TASKS
        
    print(f"\n🔄 Loading data for subject: {subject_id}")
    
    all_epochs = {'kor': [], 'eng': []}
    session_info = {'kor': [], 'eng': []}
    
    for session_idx in [1, 2, 3]:  # 3 sessions
        try:
            # Load both languages once per session, then split them locally.
            session_data = loader.load_cross_linguistic_data(
                korean_tasks=tasks,
                english_tasks=tasks,
                subject_pattern=f'*{subject_id}*',
                session_days=session_idx
            )

            for lang, data_key in [('kor', 'korean'), ('eng', 'english')]:
                if session_data[data_key]:
                    epochs_list = []

                    for data_dict in session_data[data_key]:
                        # Extract the actual Raw object from the dictionary
                        raw = data_dict['raw'] if isinstance(data_dict, dict) else data_dict
                        epochs = create_epochs_from_raw(raw)
                        if epochs is not None:
                            epochs_list.append(epochs)

                    if epochs_list:
                        # Combine epochs from the session
                        session_epochs = mne.concatenate_epochs(epochs_list)
                        all_epochs[lang].append(session_epochs)
                        session_info[lang].append(session_idx)
                    else:
                        print(f"  ❌ {lang.upper()} Session {session_idx}: No valid epochs")
        except Exception as e:
            print(f"  ❌ Error loading session {session_idx}: {e}")
    
    return all_epochs, session_info


def load_all_subjects_combined_data(loader, tasks=None):
    """
    Load and combine all session data for all subjects
    
    Parameters:
    -----------
    loader : EEGDataLoader
        Data loader instance
    tasks : list, optional
        List of tasks to load (default: SELECTED_TASKS)
        
    Returns:
    --------
    tuple
        (subject_data, subjects) where subject_data is dict mapping subject_id to combined epochs
    """
    if tasks is None:
        tasks = SELECTED_TASKS
        
    print(f"\n🔄 Loading data for all subjects...")
    
    # Collect all subject IDs
    all_files = loader.find_eeg_files(
        tasks=tasks,
        languages=['kor', 'eng']
    )
    
    subjects = set()
    for files in all_files.values():
        for file_path in files:
            filename = Path(file_path).stem
            subject_id = filename.split('_')[1]
            subjects.add(subject_id)
    
    subjects = sorted(list(subjects))
    print(f"📋 Found {len(subjects)} subjects: {subjects}")
    
    subject_data = {}
    
    for subject_id in subjects:
        print(f"\n  Loading subject: {subject_id}")
        subject_epochs = {'kor': [], 'eng': []}
        
        # Load all sessions data
        for session_idx in [1, 2, 3]:
            for lang in ['kor', 'eng']:
                try:
                    session_data = loader.load_cross_linguistic_data(
                        tasks=tasks,
                        subject_pattern=f'*{subject_id}*',
                        session_idx=session_idx
                    )
                    
                    if session_data[lang]:
                        epochs_list = []
                        
                        for raw in session_data[lang]:
                            epochs = create_epochs_from_raw(raw)
                            if epochs is not None:
                                epochs_list.append(epochs)
                        
                        if epochs_list:
                            session_epochs = mne.concatenate_epochs(epochs_list)
                            subject_epochs[lang].append(session_epochs)
                            print(f"    ✅ {lang.upper()} Session {session_idx}: {len(session_epochs)} epochs")
                
                except Exception as e:
                    print(f"    ❌ Error loading {lang} session {session_idx}: {e}")
        
        # Combine all sessions for subject
        combined_epochs = []
        for lang in ['kor', 'eng']:
            if subject_epochs[lang]:
                lang_combined = mne.concatenate_epochs(subject_epochs[lang])
                combined_epochs.append(lang_combined)
        
        if combined_epochs:
            subject_data[subject_id] = mne.concatenate_epochs(combined_epochs)
            print(f"  📊 Subject {subject_id}: {len(subject_data[subject_id])} total epochs")
        else:
            print(f"  ❌ No valid data for subject {subject_id}")
    
    return subject_data, subjects


def load_subjects_by_language(loader, tasks=None):
    """
    Load subject data separated by language for zero-shot analysis
    
    Parameters:
    -----------
    loader : EEGDataLoader
        Data loader instance
    tasks : list, optional
        List of tasks to load (default: SELECTED_TASKS)
        
    Returns:
    --------
    tuple
        (language_data, subjects) where language_data has structure:
        {'kor': {subject_id: epochs}, 'eng': {subject_id: epochs}}
    """
    if tasks is None:
        tasks = SELECTED_TASKS
        
    print(f"\n🔄 Loading data separated by language...")
    
    # Collect all subject IDs
    all_files = loader.find_eeg_files(
        tasks=tasks,
        languages=['kor', 'eng']
    )
    
    subjects = set()
    for files in all_files.values():
        for file_path in files:
            filename = Path(file_path).stem
            subject_id = filename.split('_')[1]
            subjects.add(subject_id)
    
    subjects = sorted(list(subjects))
    print(f"📋 Found {len(subjects)} subjects: {subjects}")
    
    # Collect data by language
    language_data = {'kor': {}, 'eng': {}}
    
    for subject_id in subjects:
        print(f"\n  Loading subject: {subject_id}")
        
        for lang in ['kor', 'eng']:
            subject_lang_epochs = []
            
            # Load all sessions for this language only
            for session_idx in [1, 2, 3]:
                try:
                    session_data = loader.load_cross_linguistic_data(
                        tasks=tasks,
                        subject_pattern=f'*{subject_id}*',
                        session_idx=session_idx
                    )
                    
                    if session_data[lang]:
                        epochs_list = []
                        
                        for raw in session_data[lang]:
                            epochs = create_epochs_from_raw(raw)
                            if epochs is not None:
                                epochs_list.append(epochs)
                        
                        if epochs_list:
                            session_epochs = mne.concatenate_epochs(epochs_list)
                            subject_lang_epochs.append(session_epochs)
                            print(f"    ✅ {lang.upper()} Session {session_idx}: {len(session_epochs)} epochs")
                
                except Exception as e:
                    print(f"    ❌ Error loading {lang} session {session_idx}: {e}")
            
            # Combine subject's language-specific data
            if subject_lang_epochs:
                language_data[lang][subject_id] = mne.concatenate_epochs(subject_lang_epochs)
                print(f"  📊 Subject {subject_id} {lang.upper()}: {len(language_data[lang][subject_id])} epochs")
    
    return language_data, subjects


class EEGDataLoader:
    """
    Cross-linguistic EEG data loader for Korean and English speech imagery data
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the EEG data loader
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset directory containing EEG files
        """
        self.data_path = data_path
        self.supported_languages = ['kor', 'eng']
        self.supported_tasks = list(range(7))  # task0 to task6
        
    def find_eeg_files(self, language: str = 'kor', tasks: Union[int, List[int]] = None, 
                       subject_pattern: str = '*', session_days: Union[int, List[int]] = None,
                       max_subjects: int = None) -> List[str]:
        """
        Find EEG files based on language, tasks, subject pattern, and session days
        
        Parameters:
        -----------
        language : str
            Language code ('kor' or 'eng')
        tasks : int or list of int, optional
            Task numbers to include (0-6). If None, includes all tasks
        subject_pattern : str
            Subject pattern for filtering (default: '*' for all subjects)
        session_days : int or list of int, optional
            Session day numbers (1=first day, 2=second day, 3=third day)
            If None, includes all sessions
        max_subjects : int, optional
            Maximum number of subjects to include
            
        Returns:
        --------
        list of str
            List of file paths matching the criteria
        """
        if language not in self.supported_languages:
            raise ValueError(f"Language must be one of {self.supported_languages}")
        
        # Handle single task number
        if isinstance(tasks, int):
            tasks = [tasks]
        elif tasks is None:
            tasks = self.supported_tasks
            
        # Handle single session day
        if isinstance(session_days, int):
            session_days = [session_days]
            
        # Validate task numbers
        invalid_tasks = [t for t in tasks if t not in self.supported_tasks]
        if invalid_tasks:
            raise ValueError(f"Invalid task numbers: {invalid_tasks}. Must be 0-6")
            
        # Validate session days
        if session_days is not None:
            invalid_sessions = [s for s in session_days if s not in [1, 2, 3]]
            if invalid_sessions:
                raise ValueError(f"Invalid session days: {invalid_sessions}. Must be 1, 2, or 3")
        
        found_files = []
        
        for task in tasks:
            # Pattern: YYYYMMDD_subject_taskX_lang.vhdr
            pattern = os.path.join(self.data_path, "**", f"*{subject_pattern}*_task{task}_{language}*.vhdr")
            files = glob.glob(pattern, recursive=True)
            found_files.extend(files)
        
        # Remove duplicates and sort
        found_files = sorted(list(set(found_files)))
        
        # Group files by subject and sort by date
        subject_files = self._group_files_by_subject_and_session(found_files, max_subjects, session_days)
        
        # Flatten the grouped files
        final_files = []
        for subject_data in subject_files.values():
            for session_data in subject_data.values():
                final_files.extend(session_data)
        
        print(f"Found {len(final_files)} {language.upper()} EEG files for tasks {tasks}:")
        print(f"  • Subjects: {len(subject_files)}")
        if session_days:
            print(f"  • Session days: {session_days}")
        # for i, file in enumerate(final_files):
        #     print(f"  {i+1}. {os.path.basename(file)}")
            
        return final_files
    
    def _group_files_by_subject_and_session(self, files: List[str], max_subjects: int = None, 
                                           session_days: List[int] = None) -> Dict:
        """
        Group files by subject and session (date), sorted chronologically
        
        Parameters:
        -----------
        files : list of str
            List of file paths
        max_subjects : int, optional
            Maximum number of subjects to include
        session_days : list of int, optional
            Session day numbers to include (1, 2, 3)
            
        Returns:
        --------
        dict
            Nested dict: {subject: {session_day: [files]}}
        """
        subject_sessions = {}
        
        for file_path in files:
            filename = os.path.basename(file_path)
            
            # Extract date, subject, task, language from filename
            # Expected format: YYYYMMDD_subject_taskX_lang.vhdr
            match = re.match(r'(\d{8})_([^_]+)_task(\d+)_([^.]+)\.vhdr', filename)
            if not match:
                continue
                
            date_str, subject, task, language = match.groups()
            
            if subject not in subject_sessions:
                subject_sessions[subject] = {}
            
            if date_str not in subject_sessions[subject]:
                subject_sessions[subject][date_str] = []
            
            subject_sessions[subject][date_str].append(file_path)
        
        # Sort subjects and their sessions chronologically
        sorted_subjects = {}
        subject_count = 0
        
        for subject in sorted(subject_sessions.keys()):
            if max_subjects and subject_count >= max_subjects:
                break
                
            # Sort sessions by date for this subject
            sorted_dates = sorted(subject_sessions[subject].keys())
            
            subject_data = {}
            for session_idx, date in enumerate(sorted_dates, 1):
                # Only include requested session days
                if session_days is None or session_idx in session_days:
                    subject_data[session_idx] = sorted(subject_sessions[subject][date])
            
            if subject_data:  # Only add if we have sessions for this subject
                sorted_subjects[subject] = subject_data
                subject_count += 1
        
        return sorted_subjects
    
    def load_eeg_data(self, file_paths: List[str], verbose: bool = True) -> List[mne.io.Raw]:
        """
        Load EEG data from BrainVision files
        
        Parameters:
        -----------
        file_paths : list of str
            List of BrainVision .vhdr file paths
        verbose : bool
            Whether to print loading information
            
        Returns:
        --------
        list of mne.io.Raw
            List of loaded Raw objects
        """
        raw_data = []
        
        for file_path in file_paths:
            try:
                # Load BrainVision file
                raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
                
                if verbose:
                    print(f"  ✓ Loaded: {os.path.basename(file_path)} (Subject: {os.path.basename(file_path).split('_')[1]}, Date: {os.path.basename(file_path).split('_')[0]})")
                    # print(f"  - Channels: {raw.info['nchan']}")
                    # print(f"  - Sampling rate: {raw.info['sfreq']} Hz")
                    # print(f"  - Duration: {raw.times[-1]:.2f} seconds")
                    # print(f"  - Channel names: {raw.ch_names[:5]}..." if len(raw.ch_names) > 5 else f"  - Channel names: {raw.ch_names}")
                
                raw_data.append(raw)
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # if verbose:
        #     print(f"\nSuccessfully loaded {len(raw_data)} EEG recordings")
        
        return raw_data
    
    def load_cross_linguistic_data(self, korean_tasks: Union[int, List[int]] = 4, 
                                   english_tasks: Union[int, List[int]] = 4,
                                   subject_pattern: str = '*',
                                   max_subjects: int = None,
                                   session_days: Union[int, List[int]] = None) -> Dict:
        """
        Load cross-linguistic EEG data for comparative analysis with session support
        
        Parameters:
        -----------
        korean_tasks : int or list of int
            Task numbers to load for Korean data
        english_tasks : int or list of int  
            Task numbers to load for English data
        subject_pattern : str
            Pattern to filter subjects (e.g., 'hgpark', '*park*', '*')
        max_subjects : int, optional
            Maximum number of subjects to load per language
        session_days : int or list of int, optional
            Session day numbers to include (1=first day, 2=second day, 3=third day)
            If None, includes all sessions
            
        Returns:
        --------
        dict
            Dictionary containing loaded data with keys 'korean' and 'english'
            Each contains list of dicts with 'raw', 'filename', 'filepath', 'subject', 'session_day'
        """
        print("Loading cross-linguistic EEG data...")
        
        # Find files for both languages with session support
        korean_files = self.find_eeg_files(language='kor', tasks=korean_tasks, 
                                         subject_pattern=subject_pattern,
                                         session_days=session_days,
                                         max_subjects=max_subjects)
        english_files = self.find_eeg_files(language='eng', tasks=english_tasks,
                                          subject_pattern=subject_pattern,
                                          session_days=session_days,
                                          max_subjects=max_subjects)
        
        # Load the data
        korean_data = []
        english_data = []
        
        print(f"\nLoading {len(korean_files)} Korean files...")
        for i, file_path in enumerate(korean_files):
            try:
                raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
                
                # Extract subject and session info from filename
                filename = os.path.basename(file_path)
                subject_info = self._extract_subject_session_info(filename)
                
                korean_data.append({
                    'raw': raw,
                    'filename': filename,
                    'filepath': file_path,
                    'subject': subject_info['subject'],
                    'session_day': subject_info['session_day'],
                    'date': subject_info['date']
                })
                print(f"  ✓ Loaded: {filename} (Subject: {subject_info['subject']}, Date: {subject_info['date']})")
            except Exception as e:
                print(f"  ✗ Failed to load {os.path.basename(file_path)}: {e}")
        
        print(f"\nLoading {len(english_files)} English files...")
        for i, file_path in enumerate(english_files):
            try:
                raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
                
                # Extract subject and session info from filename
                filename = os.path.basename(file_path)
                subject_info = self._extract_subject_session_info(filename)
                
                english_data.append({
                    'raw': raw,
                    'filename': filename,
                    'filepath': file_path,
                    'subject': subject_info['subject'],
                    'session_day': subject_info['session_day'],
                    'date': subject_info['date']
                })
                print(f"  ✓ Loaded: {filename} (Subject: {subject_info['subject']}, Date: {subject_info['date']})")
            except Exception as e:
                print(f"  ✗ Failed to load {os.path.basename(file_path)}: {e}")
        
        data = {
            'korean': korean_data,
            'english': english_data
        }
        
        # Print session summary
        self._print_session_summary(data)
        
        print(f"\nLoaded {len(korean_data)} Korean and {len(english_data)} English datasets")
        return data
    
    def _extract_subject_session_info(self, filename: str) -> Dict:
        """
        Extract subject and session information from filename
        
        Parameters:
        -----------
        filename : str
            EEG filename (e.g., 20231201_hgpark_task4_kor.vhdr)
            
        Returns:
        --------
        dict
            Dictionary with subject, session_day, and date information
        """
        # Extract date, subject, task, language from filename
        match = re.match(r'(\d{8})_([^_]+)_task(\d+)_([^.]+)\.vhdr', filename)
        if not match:
            return {'subject': 'unknown', 'session_day': 0, 'date': 'unknown'}
            
        date_str, subject, task, language = match.groups()
        
        return {
            'subject': subject,
            'session_day': 1,  # Will be updated in grouping function
            'date': date_str
        }
    
    def _print_session_summary(self, data: Dict):
        """Print summary of loaded sessions per subject"""
        print("\n=== Session Summary ===")
        
        for language, datasets in data.items():
            if not datasets:
                continue
                
            print(f"\n{language.upper()} Data:")
            
            # Group by subject
            subjects = {}
            for dataset in datasets:
                subject = dataset['subject']
                if subject not in subjects:
                    subjects[subject] = []
                subjects[subject].append(dataset)
            
            for subject, subject_data in subjects.items():
                dates = sorted(set(d['date'] for d in subject_data))
                print(f"  • {subject}: {len(dates)} sessions ({', '.join(dates)})")
        
        print("========================")


class EEGVisualizer:
    """
    EEG data visualization utilities
    """
    
    @staticmethod
    def plot_raw_eeg_data(raw: mne.io.Raw, duration: float = 10, start: float = 0, 
                         n_channels: int = 8, title: str = None):
        """
        Visualize raw EEG data
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw EEG data object
        duration : float
            Duration to display in seconds
        start : float
            Start time in seconds
        n_channels : int
            Number of channels to display
        title : str, optional
            Plot title
        """
        plt.ion()
        
        if title is None:
            title = 'EEG Raw Data'
        
        fig = raw.plot(duration=duration, start=start, n_channels=n_channels, 
                      scalings='auto', show=True, block=False, title=title)
        
        return fig
    
    @staticmethod
    def plot_psd_analysis(raw: mne.io.Raw, fmax: float = 50, title_prefix: str = ""):
        """
        Power Spectral Density analysis and visualization
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw EEG data object
        fmax : float
            Maximum frequency for analysis
        title_prefix : str
            Prefix for plot titles
        """
        plt.ioff()
        
        # Compute PSD
        psd = raw.compute_psd(method='welch', fmax=fmax)
        psds, freqs = psd.get_data(return_freqs=True)
        
        # 1. All channels PSD plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        for i, ch_name in enumerate(raw.ch_names):
            ax1.plot(freqs, 10 * np.log10(psds[i]), alpha=0.7, 
                    label=ch_name if i < 5 else "")
        
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (dB)')
        ax1.set_title(f'{title_prefix}Power Spectral Density - All Channels')
        ax1.grid(True, alpha=0.3)
        if len(raw.ch_names) <= 5:
            ax1.legend()
        plt.tight_layout()
        plt.show()
        
        # 2. Frequency band topography
        frequency_bands = {
            'Alpha (8-12 Hz)': (8, 12),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-50 Hz)': (30, 50)
        }
        
        fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (band_name, (fmin, fmax_band)) in enumerate(frequency_bands.items()):
            try:
                freq_mask = (freqs >= fmin) & (freqs <= fmax_band)
                if np.any(freq_mask):
                    band_power = np.mean(psds[:, freq_mask], axis=1)
                    
                    im, _ = mne.viz.plot_topomap(band_power, raw.info, axes=axes[i], 
                                               show=False, cmap='viridis')
                    axes[i].set_title(f'{title_prefix}{band_name}')
                else:
                    axes[i].text(0.5, 0.5, f'No data in\n{band_name} range', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{title_prefix}{band_name} - No Data')
                    
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{title_prefix}{band_name} - Error')
        
        plt.suptitle(f'{title_prefix}Frequency Band Topography')
        plt.tight_layout()
        plt.show()
        
        return psd, fig1, fig2
    
    @staticmethod
    def plot_channel_statistics(raw: mne.io.Raw, title_prefix: str = ""):
        """
        Visualize channel statistics
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw EEG data object
        title_prefix : str
            Prefix for plot titles
        """
        plt.ioff()
        
        data = raw.get_data()
        
        # Calculate statistics
        channel_stats = {
            'means': np.mean(data, axis=1),
            'stds': np.std(data, axis=1),
            'mins': np.min(data, axis=1),
            'maxs': np.max(data, axis=1),
            'rms': np.sqrt(np.mean(data**2, axis=1))
        }
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Means
        axes[0, 0].bar(range(len(raw.ch_names)), channel_stats['means'], 
                       color='skyblue', alpha=0.7)
        axes[0, 0].set_title(f'{title_prefix}Channel Means')
        axes[0, 0].set_xlabel('Channel Index')
        axes[0, 0].set_ylabel('Mean Amplitude (V)')
        
        # Standard deviations
        axes[0, 1].bar(range(len(raw.ch_names)), channel_stats['stds'], 
                       color='lightcoral', alpha=0.7)
        axes[0, 1].set_title(f'{title_prefix}Channel Standard Deviations')
        axes[0, 1].set_xlabel('Channel Index')
        axes[0, 1].set_ylabel('Std Amplitude (V)')
        
        # RMS values
        axes[0, 2].bar(range(len(raw.ch_names)), channel_stats['rms'], 
                       color='lightgreen', alpha=0.7)
        axes[0, 2].set_title(f'{title_prefix}Channel RMS Values')
        axes[0, 2].set_xlabel('Channel Index')
        axes[0, 2].set_ylabel('RMS Amplitude (V)')
        
        # Min/Max values
        width = 0.35
        x_pos = np.arange(len(raw.ch_names))
        axes[1, 0].bar(x_pos - width/2, channel_stats['maxs'], width, 
                       alpha=0.7, label='Max', color='green')
        axes[1, 0].bar(x_pos + width/2, channel_stats['mins'], width, 
                       alpha=0.7, label='Min', color='red')
        axes[1, 0].set_title(f'{title_prefix}Channel Min/Max Values')
        axes[1, 0].set_xlabel('Channel Index')
        axes[1, 0].set_ylabel('Amplitude (V)')
        axes[1, 0].legend()
        
        # Overall distribution
        axes[1, 1].hist(data.flatten(), bins=100, alpha=0.7, color='purple')
        axes[1, 1].set_title(f'{title_prefix}Overall Data Distribution')
        axes[1, 1].set_xlabel('Amplitude (V)')
        axes[1, 1].set_ylabel('Frequency')
        
        # Channel variance distribution
        axes[1, 2].hist(channel_stats['stds'], bins=20, alpha=0.7, color='orange')
        axes[1, 2].set_title(f'{title_prefix}Channel Variance Distribution')
        axes[1, 2].set_xlabel('Standard Deviation (V)')
        axes[1, 2].set_ylabel('Number of Channels')
        
        plt.tight_layout()
        plt.show()
        
        return channel_stats


class ConnectivityAnalyzer:
    """
    Brain connectivity analysis utilities for cross-linguistic comparison
    """
    
    def __init__(self):
        self.connectivity_methods = ['coh', 'pli', 'wpli', 'plv']
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def compute_connectivity(self, raw: mne.io.Raw, method: str = 'wpli', 
                           fmin: float = 8, fmax: float = 12, 
                           epoch_length: float = 1.0) -> Optional[np.ndarray]:
        """
        Compute connectivity between channels
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw EEG data
        method : str
            Connectivity method ('coh', 'pli', 'wpli', 'plv')
        fmin, fmax : float
            Frequency range for analysis
        epoch_length : float
            Length of epochs in seconds
            
        Returns:
        --------
        numpy array or None
            Connectivity matrix
        """
        try:
            # Try multiple import methods for mne-connectivity
            spectral_connectivity_epochs = None
            
            # Method 1: Direct import from mne_connectivity
            try:
                from mne_connectivity import spectral_connectivity_epochs
            except ImportError:
                # Method 2: Import from mne.connectivity (older versions)
                try:
                    from mne.connectivity import spectral_connectivity_epochs
                except ImportError:
                    print(f"❌ MNE-connectivity not available for {method}")
                    return None
            
            # Create fixed-length events
            events = mne.make_fixed_length_events(raw, duration=epoch_length)
            epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_length, 
                               baseline=None, preload=True, verbose=False)
            
            # Compute connectivity
            con = spectral_connectivity_epochs(
                epochs.get_data(), method=method, 
                mode='multitaper', sfreq=raw.info['sfreq'], 
                fmin=fmin, fmax=fmax, verbose=False
            )
            
            # Return connectivity matrix
            if hasattr(con, 'get_data'):
                return con.get_data()[:, :, 0]  # First frequency band
            else:
                return con[0]  # Handle different return formats
            
        except Exception as e:
            print(f"❌ Connectivity computation error ({method}): {e}")
            return None
    
    def plot_connectivity_matrix(self, con_matrix: np.ndarray, raw: mne.io.Raw, method: str, 
                               fmin: float, fmax: float, title_prefix: str = ""):
        """
        Plot connectivity matrix
        
        Parameters:
        -----------
        con_matrix : np.ndarray
            Connectivity matrix
        raw : mne.io.Raw
            Raw EEG data for channel names
        method : str
            Connectivity method used
        fmin, fmax : float
            Frequency range
        title_prefix : str
            Prefix for plot title
        """
        if con_matrix is None:
            print(f"❌ No connectivity matrix to plot for {method}")
            return None
            
        plt.ioff()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(con_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'{title_prefix}Channel Connectivity ({method.upper()}) - {fmin}-{fmax} Hz')
        
        # Set channel names
        ax.set_xticks(range(len(raw.ch_names)))
        ax.set_yticks(range(len(raw.ch_names)))
        ax.set_xticklabels(raw.ch_names, rotation=45)
        ax.set_yticklabels(raw.ch_names)
        
        plt.colorbar(im, ax=ax, label='Connectivity')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def compare_connectivity(self, kor_data: List[mne.io.Raw], eng_data: List[mne.io.Raw], 
                           method: str = 'wpli', frequency_band: str = 'alpha') -> Dict:
        """
        Compare connectivity between Korean and English data
        
        Parameters:
        -----------
        kor_data : list of mne.io.Raw
            Korean EEG data
        eng_data : list of mne.io.Raw
            English EEG data
        method : str
            Connectivity method
        frequency_band : str
            Frequency band name
            
        Returns:
        --------
        dict
            Comparison results
        """
        if frequency_band not in self.frequency_bands:
            raise ValueError(f"Frequency band must be one of {list(self.frequency_bands.keys())}")
        
        fmin, fmax = self.frequency_bands[frequency_band]
        
        results = {
            'kor_connectivity': [],
            'eng_connectivity': [],
            'kor_networks': [],
            'eng_networks': [],
            'statistics': {}
        }
        
        # Compute connectivity for Korean data
        print(f"Computing {method.upper()} connectivity for Korean data...")
        for i, raw in enumerate(kor_data):
            con_matrix = self.compute_connectivity(raw, method=method, fmin=fmin, fmax=fmax)
            if con_matrix is not None:
                results['kor_connectivity'].append(con_matrix)
                # Network metrics
                results['kor_networks'].append(self._compute_network_metrics(con_matrix))
        
        # Compute connectivity for English data
        print(f"Computing {method.upper()} connectivity for English data...")
        for i, raw in enumerate(eng_data):
            con_matrix = self.compute_connectivity(raw, method=method, fmin=fmin, fmax=fmax)
            if con_matrix is not None:
                results['eng_connectivity'].append(con_matrix)
                # Network metrics
                results['eng_networks'].append(self._compute_network_metrics(con_matrix))
        
        # Statistical comparison
        results['statistics'] = self._compare_network_statistics(
            results['kor_networks'], results['eng_networks']
        )
        
        return results
    
    def _compute_network_metrics(self, connectivity_matrix: np.ndarray) -> Dict:
        """
        Compute network metrics from connectivity matrix
        """
        # Create graph
        G = nx.from_numpy_array(connectivity_matrix)
        
        # Compute metrics
        metrics = {
            'global_efficiency': nx.global_efficiency(G),
            'local_efficiency': nx.local_efficiency(G),
            'clustering': nx.average_clustering(G),
            'path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else np.inf,
            'density': nx.density(G),
            'mean_connectivity': np.mean(connectivity_matrix),
            'std_connectivity': np.std(connectivity_matrix)
        }
        
        return metrics
    
    def _compare_network_statistics(self, kor_networks: List[Dict], 
                                  eng_networks: List[Dict]) -> Dict:
        """
        Statistical comparison between Korean and English networks
        """
        if not kor_networks or not eng_networks:
            return {}
        
        # Extract metrics
        kor_metrics = {key: [net[key] for net in kor_networks] for key in kor_networks[0].keys()}
        eng_metrics = {key: [net[key] for net in eng_networks] for key in eng_networks[0].keys()}
        
        # Statistical tests
        comparison = {}
        for metric in kor_metrics.keys():
            kor_values = np.array(kor_metrics[metric])
            eng_values = np.array(eng_metrics[metric])
            
            # Remove infinite values
            kor_finite = kor_values[np.isfinite(kor_values)]
            eng_finite = eng_values[np.isfinite(eng_values)]
            
            if len(kor_finite) > 0 and len(eng_finite) > 0:
                t_stat, p_value = stats.ttest_ind(kor_finite, eng_finite)
                
                comparison[metric] = {
                    'kor_mean': np.mean(kor_finite),
                    'kor_std': np.std(kor_finite),
                    'eng_mean': np.mean(eng_finite),
                    'eng_std': np.std(eng_finite),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return comparison


def display_eeg_summary(raw_data_list: List[mne.io.Raw], language: str = ""):
    """
    Display summary of EEG data
    
    Parameters:
    -----------
    raw_data_list : list of mne.io.Raw
        List of Raw objects
    language : str
        Language identifier for display
    """
    if not raw_data_list:
        print("No data to summarize")
        return
    
    print("=" * 60)
    print(f"{language.upper()} EEG DATA SUMMARY")
    print("=" * 60)
    
    total_duration = 0
    all_channels = set()
    sfreqs = []
    
    for i, raw in enumerate(raw_data_list):
        print(f"\nFile {i+1}:")
        print(f"  Duration: {raw.times[-1]:.2f} seconds")
        print(f"  Sampling Rate: {raw.info['sfreq']} Hz")
        print(f"  Channels ({raw.info['nchan']}): {', '.join(raw.ch_names)}")
        
        total_duration += raw.times[-1]
        all_channels.update(raw.ch_names)
        sfreqs.append(raw.info['sfreq'])
    
    print(f"\n" + "=" * 40)
    print(f"OVERALL SUMMARY:")
    print(f"  Total files: {len(raw_data_list)}")
    print(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"  Unique channels: {len(all_channels)}")
    print(f"  Sampling rates: {set(sfreqs)}")
    print(f"  All channel names: {sorted(list(all_channels))}")


def plot_cross_linguistic_comparison(kor_data: List[mne.io.Raw], eng_data: List[mne.io.Raw],
                                   analysis_type: str = 'psd', **kwargs):
    """
    Plot cross-linguistic comparison
    
    Parameters:
    -----------
    kor_data : list of mne.io.Raw
        Korean EEG data
    eng_data : list of mne.io.Raw
        English EEG data
    analysis_type : str
        Type of analysis ('psd', 'connectivity')
    **kwargs : dict
        Additional parameters for specific analysis
    """
    visualizer = EEGVisualizer()
    
    if analysis_type == 'psd':
        print("Cross-linguistic PSD comparison...")
        if kor_data:
            print("\nKorean PSD Analysis:")
            visualizer.plot_psd_analysis(kor_data[0], title_prefix="Korean - ")
        
        if eng_data:
            print("\nEnglish PSD Analysis:")
            visualizer.plot_psd_analysis(eng_data[0], title_prefix="English - ")
    
    elif analysis_type == 'connectivity':
        analyzer = ConnectivityAnalyzer()
        comparison_results = analyzer.compare_connectivity(kor_data, eng_data, **kwargs)
        
        # Plot results
        if comparison_results['statistics']:
            print("\nCross-linguistic connectivity comparison results:")
            for metric, stats in comparison_results['statistics'].items():
                print(f"\n{metric}:")
                print(f"  Korean: {stats['kor_mean']:.4f} ± {stats['kor_std']:.4f}")
                print(f"  English: {stats['eng_mean']:.4f} ± {stats['eng_std']:.4f}")
                print(f"  t-statistic: {stats['t_statistic']:.4f}")
                print(f"  p-value: {stats['p_value']:.4f}")
                print(f"  Significant: {'Yes' if stats['significant'] else 'No'}")
        
        return comparison_results
    
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")


# Convenience functions for backward compatibility
def load_korean_eeg_data(file_paths: List[str]) -> List[mne.io.Raw]:
    """Backward compatibility function for Korean data loading"""
    loader = EEGDataLoader("")
    return loader.load_eeg_data(file_paths)


def load_english_eeg_data(file_paths: List[str]) -> List[mne.io.Raw]:
    """Load English EEG data"""
    loader = EEGDataLoader("")
    return loader.load_eeg_data(file_paths)
