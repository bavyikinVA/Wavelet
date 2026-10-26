"""Durable ML operation journal; successful outputs are verified before reuse."""
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from history.atomic_io import atomic_write_text
from compute.knn.result_format import is_disk_knn_payload
from .validation import validate_ml_parameters


def file_digest(path, check):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            check()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# Bump whenever feature extraction/model/output semantics change.
ML_IMPLEMENTATION_VERSION = 'knn-clustering-1'


class OperationJournal:
    def __init__(self, folder, check, warn):
        self.path = Path(folder) / '.ml_operations.json'
        self.check = check
        self.entries = {}
        self.sources = {}
        self.inputs = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding='utf-8'))
                if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
                    raise ValueError('invalid operation journal')
                self.entries = data
            except (OSError, ValueError) as error:
                warn(f'Журнал ML недоступен; операции будут проверены повторным запуском: {error}')

    def identity(self, config, selected):
        cfg = validate_ml_parameters(config)
        fields = ['algorithm', 'dataset_key', 'point_filter', 'feature_set', 'standardize', 'random_state']
        defaults = dict(point_filter='all', feature_set='knn_raw', standardize=True, random_state=42)
        fields += (['n_clusters'] if cfg['algorithm'] == 'kmeans' else
                   ['eps', 'min_samples'])
        parameters = {key: cfg.get(key, defaults.get(key)) for key in fields}
        source_key = tuple(config['dataset_key'])
        if source_key not in self.sources:
            digest = hashlib.sha256()
            for key, payload in selected.items():
                self.check()
                digest.update(repr(key).encode('utf-8'))
                if is_disk_knn_payload(payload):
                    digest.update(file_digest(payload['path'], self.check).encode('ascii'))
                else:
                    class Sink:
                        def write(sink, block):
                            self.check()
                            digest.update(block)
                    pickle.Pickler(Sink(), protocol=5).dump(payload)
            self.sources[source_key] = digest.hexdigest()
        identity = dict(parameters=parameters, source=self.sources[source_key],
                        implementation=ML_IMPLEMENTATION_VERSION)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.inputs[key] = identity
        return key

    def reusable(self, identity):
        entry = self.entries.get(identity, {})
        if entry.get('status') not in {'completed', 'saving'}:
            return None
        try:
            result = entry['result']
            artifacts = entry['artifacts']
            if not isinstance(result, dict) or not isinstance(artifacts, dict) or len(artifacts) != 3:
                return None
            for field in ('table_path', 'summary_path', 'feature_plot_path'):
                if file_digest(result[field], self.check) != artifacts[field]:
                    return None
            return result
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def save(self, identity, status, **details):
        now = datetime.now(timezone.utc).isoformat()
        previous = self.entries.get(identity, {})
        entry = {**previous, **details, 'status': status, 'updated_at_utc': now,
                 'implementation_version': ML_IMPLEMENTATION_VERSION}
        if identity in self.inputs:
            entry['inputs'] = self.inputs[identity]
        event = dict(status=status, at_utc=now)
        if status == 'running':
            entry['started_at_utc'] = now
            entry['attempts'] = previous.get('attempts', 0) + 1
            for key in ('result', 'artifacts', 'error', 'finished_at_utc', 'duration_seconds'):
                entry.pop(key, None)
        elif status in {'saving', 'completed', 'failed', 'cancelled'}:
            finished = previous.get('finished_at_utc', now) if previous.get('status') == 'saving' else now
            entry['finished_at_utc'] = finished
            if entry.get('started_at_utc'):
                entry['duration_seconds'] = max(0, (datetime.fromisoformat(finished) -
                    datetime.fromisoformat(entry['started_at_utc'])).total_seconds())
        entry['events'] = [*previous.get('events', []), event]
        updated = {**self.entries, identity: entry}
        atomic_write_text(self.path, json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False))
        self.entries = updated

    def completion_details(self, result):
        artifacts = {field: file_digest(result[field], self.check)
                     for field in ('table_path', 'summary_path', 'feature_plot_path')}
        fields = ('algorithm', 'algorithm_name', 'point_count', 'metrics', 'output_dir',
                  'table_path', 'summary_path', 'feature_plot_path', 'dbscan_backend', 'dbscan_fallback_reason', 'dataset')
        compact = {field: result[field] for field in fields if field in result}
        return dict(result=compact, artifacts=artifacts)

    def complete(self, identity, result):
        self.save(identity, 'completed', **self.completion_details(result))

    def record_reuse(self, identity, config):
        now = datetime.now(timezone.utc).isoformat()
        entry = self.entries[identity]
        updated = {**self.entries, identity: {**entry, 'configuration': config,
            'last_reused_at_utc': now, 'reuse_count': entry.get('reuse_count', 0) + 1,
            'events': [*entry.get('events', []), dict(status='reused', at_utc=now)]}}
        atomic_write_text(self.path, json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False))
        self.entries = updated
