from pathlib import Path
import json
from shock.utils import h5Dataset
import mne

savePath = Path('D:\\SemanticDecoding\\Dataset\\h5dataset\\')
rawDataPath = Path('D:\\SemanticDecoding\\Dataset\\')
group = sorted(rawDataPath.glob('*.vhdr'))

# preprocessing parameters
l_freq = 0.1
h_freq = 75.0
rsfreq = 200
notch_freq = 60.0
trim_tail_sec = 10
overwrite_output = True

def preprocessing_brainvision(vhdrFilePath, l_freq=0.1, h_freq=75.0, sfreq=200, notch=60.0):
    raw = mne.io.read_raw_brainvision(vhdrFilePath, preload=True, verbose=False)
    orig_sfreq = float(raw.info['sfreq'])

    # Drop common non-EEG channels if they exist.
    drop_candidates = ['M1', 'M2', 'VEO', 'HEO', 'ECG']
    drop_channels = [ch for ch in drop_candidates if ch in raw.ch_names]
    if drop_channels:
        raw.drop_channels(drop_channels)

    # Keep only EEG channels for stable channel consistency.
    raw.pick_types(eeg=True)

    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    if notch is not None:
        raw = raw.notch_filter(notch)
    raw = raw.resample(sfreq, n_jobs=5)

    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except ValueError:
        events = None
        event_id = {}

    eegData = raw.get_data(units='uV')
    chOrder = [s.upper() for s in raw.ch_names]
    return eegData, chOrder, events, event_id, orig_sfreq

savePath.mkdir(parents=True, exist_ok=True)
output_file = savePath / 'dataset.hdf5'
if overwrite_output and output_file.exists():
    output_file.unlink()

dataset = h5Dataset(savePath, 'dataset')
for vhdrFile in group:
    vmrkFile = vhdrFile.with_suffix('.vmrk')
    eegFile = vhdrFile.with_suffix('.eeg')

    if not vmrkFile.exists() or not eegFile.exists():
        print(f'skipping {vhdrFile.name}: missing paired .vmrk or .eeg')
        continue

    print(f'processing {vhdrFile.name}')
    eegData, chOrder, events, event_id, orig_sfreq = preprocessing_brainvision(
        vhdrFile,
        l_freq,
        h_freq,
        rsfreq,
        notch_freq,
    )

    trim_samples = int(trim_tail_sec * rsfreq)
    if eegData.shape[1] > trim_samples:
        eegData = eegData[:, :-trim_samples]
        if events is not None and len(events) > 0:
            events = events[events[:, 0] < eegData.shape[1]]

    grp = dataset.addGroup(grpName=vhdrFile.stem)
    ch_chunk = min(62, eegData.shape[0])
    t_chunk = min(rsfreq, eegData.shape[1])
    chunks = (max(1, ch_chunk), max(1, t_chunk))
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'notchFreq', notch_freq)
    dataset.addAttributes(dset, 'origSfreq', orig_sfreq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)
    dataset.addAttributes(dset, 'sourceVhdr', vhdrFile.name)
    dataset.addAttributes(dset, 'sourceVmrk', vmrkFile.name)
    dataset.addAttributes(dset, 'sourceEeg', eegFile.name)

    if events is not None and len(events) > 0:
        ev = dataset.addDataset(grp, 'events', events, chunks=(min(1024, len(events)), 3))
        dataset.addAttributes(ev, 'columns', ['sample', 'previous', 'event_code'])
        dataset.addAttributes(ev, 'event_id', json.dumps(event_id, ensure_ascii=True))
    else:
        dataset.addAttributes(grp, 'event_id', json.dumps({}, ensure_ascii=True))
        dataset.addAttributes(grp, 'events', 'none')

dataset.save()
