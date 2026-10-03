"""Viraat's Yelp preprocessing utilities. No model implementations.

Assistant-assisted implementation, starting from Viraat's checked Cell 6.
Revision 2 normalizes Yelp's literal escaped whitespace before tokenization.
Team notebooks were consulted only for the split and metric conventions.
API references: nltk.org/api/nltk.stem.porter.html;
scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html;
numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html.
"""
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

NEGATIONS = frozenset({'no', 'not', 'nor', 'never', 'neither',
                       'without', 'hardly', 'barely', 'scarcely'})
RESERVED = {'<pad>': 0, '<unk>': 1, '<empty>': 2}
TOKEN_PATTERN = r'[^\W\d_]+'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def build_preprocessor(stop_words):
    stop_words = frozenset(stop_words) - NEGATIONS
    stemmer = PorterStemmer(mode=PorterStemmer.NLTK_EXTENSIONS)
    words = re.compile(TOKEN_PATTERN, re.UNICODE)
    special = {'cannot': 'can not', "can't": 'can not', "won't": 'will not',
               "shan't": 'shall not', "ain't": 'is not'}
    special_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in special) + r')\b')
    contraction = re.compile(r"\b([^\W\d_]+)n't\b", re.UNICODE)

    @lru_cache(maxsize=100_000)
    def stem(token):
        return token if token in NEGATIONS else stemmer.stem(token)

    def tokenize(review):
        if not isinstance(review, str):
            raise TypeError('Review must be a string; quarantine invalid raw rows separately.')
        text = unicodedata.normalize('NFKC', html.unescape(review)).lower()
        text = text.replace('\u2019', "'").replace('\u2018', "'")
        # Yelp stores some line breaks as two literal characters: backslash + n.
        # Handle only whitespace escapes; do not use unicode_escape on Unicode text.
        text = re.sub(r'\\[nrt]', ' ', text)
        text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
        text = re.sub(r'</?[a-z][^>]*>', ' ', text)
        text = special_pattern.sub(lambda match: special[match.group(0)], text)
        text = contraction.sub(lambda match: match.group(1) + ' not', text)
        return [stem(token) for token in words.findall(text) if token not in stop_words]

    return tokenize


def make_membership(labels, validation_fraction, seed):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Training labels must be a one-dimensional binary array.')
    indices = np.arange(len(labels), dtype=np.int32)
    training, validation = train_test_split(
        indices, test_size=validation_fraction, random_state=seed,
        shuffle=True, stratify=labels)
    training = np.sort(np.asarray(training, dtype=np.int32))
    validation = np.sort(np.asarray(validation, dtype=np.int32))
    if not np.array_equal(np.sort(np.concatenate([training, validation])), indices):
        raise AssertionError('Membership does not partition the official training population.')
    return training, validation


def cache_tokens(dataset, training, validation, tokenize, cache_root, cache_key):
    """One raw-training pass; only training membership updates the vocabulary counter."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_root / 'token_cache_manifest.json'
    targets = {name: cache_root / f'{name}_tokens.jsonl'
               for name in ('train', 'validation')}
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text())
        if prior['cache_key'] != cache_key:
            raise RuntimeError('Cached preprocessing differs. Use a new run ID for changed settings.')
        for name, path in targets.items():
            if not path.exists() or sha256_file(path) != prior['files_sha256'][name]:
                raise RuntimeError('Token cache integrity failed: ' + name)
        return prior, Counter(prior['training_token_counts'])

    membership = np.full(len(dataset), -1, dtype=np.int8)
    membership[training] = 0
    membership[validation] = 1
    if (membership < 0).any():
        raise AssertionError('Incomplete membership.')
    counts = Counter()
    text_counts = {'train': Counter(), 'validation': Counter()}
    row_counts = {'train': 0, 'validation': 0}
    empty = {'train': 0, 'validation': 0}
    temporary = {name: path.with_name(path.name + '.part') for name, path in targets.items()}
    handles = {name: path.open('w', encoding='utf-8') for name, path in temporary.items()}
    row_index = 0
    try:
        batches = dataset.iter(batch_size=2048)
        for batch in tqdm(batches, total=(len(dataset) + 2047) // 2048,
                          desc='Tokenize official training population once'):
            for text, label in zip(batch['text'], batch['label']):
                if not isinstance(text, str) or not text.strip() or label not in (0, 1):
                    write_json(cache_root / 'unexpected_invalid_row.json',
                               {'source_index': row_index, 'reason': 'invalid text or label'})
                    raise ValueError(f'Unexpected invalid official training row {row_index}; halted.')
                name = 'train' if membership[row_index] == 0 else 'validation'
                tokens = tokenize(text)
                handles[name].write(json.dumps([row_index, int(label), tokens],
                                              ensure_ascii=False, separators=(',', ':')) + '\n')
                row_counts[name] += 1
                empty[name] += int(not tokens)
                # Exact stripped-text duplicates are audited; official rows are retained.
                digest = hashlib.sha256(text.strip().encode('utf-8')).hexdigest()
                text_counts[name][digest] += 1
                if name == 'train':
                    counts.update(tokens)
                row_index += 1
    finally:
        for handle in handles.values():
            handle.close()
    if row_counts != {'train': len(training), 'validation': len(validation)}:
        raise AssertionError('Tokenized row counts do not match membership.')
    for name in targets:
        temporary[name].replace(targets[name])
    train_hashes, val_hashes = text_counts['train'], text_counts['validation']
    metadata = {
        'cache_key': cache_key, 'row_counts': row_counts,
        'empty_after_preprocessing': empty,
        'training_token_counts': dict(counts),
        'files_sha256': {name: sha256_file(path) for name, path in targets.items()},
        'duplicate_audit': {
            'definition': 'SHA-256 of exact stripped raw training text; rows retained',
            'extra_duplicate_rows_train': sum(c - 1 for c in train_hashes.values()),
            'extra_duplicate_rows_validation': sum(c - 1 for c in val_hashes.values()),
            'unique_texts_shared_between_train_and_validation': len(train_hashes.keys() & val_hashes.keys()),
            'official_test_text_inspected': False,
            'limitation': 'Shared duplicate texts can make validation optimistic. '
                          'Retained to preserve the common official-row split; report the audit.'}}
    write_json(metadata_path, metadata)
    return metadata, counts


def fit_vocabulary(training_counts, max_size, min_frequency):
    if max_size < len(RESERVED) or min_frequency < 1:
        raise ValueError('Invalid vocabulary limits.')
    eligible = sorted(((t, n) for t, n in training_counts.items() if n >= min_frequency),
                      key=lambda item: (-item[1], item[0]))
    mapping = dict(RESERVED)
    for token, count in eligible[:max_size - len(RESERVED)]:
        mapping[token] = len(mapping)
    return mapping


def encode_tokens(tokens, vocabulary, sequence_length):
    """Right-pad, keep the first L tokens; preserve an explicit token for empty reviews."""
    if sequence_length < 1:
        raise ValueError('Sequence length must be positive.')
    original_length = len(tokens)
    kept = tokens[:sequence_length] if tokens else ['<empty>']
    ids = np.asarray([vocabulary.get(token, RESERVED['<unk>']) for token in kept], dtype=np.int32)
    padded = np.zeros(sequence_length, dtype=np.int32)
    padded[:len(ids)] = ids
    oov = int(np.count_nonzero(ids == RESERVED['<unk>']))
    return padded, {
        'lengths': len(ids), 'processed_lengths': original_length,
        'oov_counts': oov, 'oov_rates': oov / len(ids),
        'contains_negation': any(token in NEGATIONS for token in kept)}


def encode_cache(cache_path, output_root, name, row_count, vocabulary, sequence_length, cache_key):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / f'{name}_encoding_manifest.json'
    encoding_key = fingerprint({'cache_key': cache_key, 'vocabulary': vocabulary,
                                'sequence_length': sequence_length, 'split': name})
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior['encoding_key'] != encoding_key:
            raise RuntimeError('Existing encoding settings differ; use a new run ID.')
        for filename, expected in prior['files_sha256'].items():
            if sha256_file(output_root / filename) != expected:
                raise RuntimeError('Encoded artifact hash mismatch: ' + filename)
        return prior
    specifications = {
        'input_ids': ('int32', (row_count, sequence_length)),
        'labels': ('int8', (row_count,)), 'source_indices': ('int32', (row_count,)),
        'lengths': ('int32', (row_count,)), 'processed_lengths': ('int32', (row_count,)),
        'oov_counts': ('int32', (row_count,)), 'oov_rates': ('float64', (row_count,)),
        'contains_negation': ('bool', (row_count,))}
    paths = {field: output_root / f'{name}_{field}.npy' for field in specifications}
    temporary = {field: path.with_name(path.name + '.part') for field, path in paths.items()}
    arrays = {field: np.lib.format.open_memmap(temporary[field], mode='w+', dtype=dtype, shape=shape)
              for field, (dtype, shape) in specifications.items()}
    seen = 0
    with Path(cache_path).open(encoding='utf-8') as handle:
        for row, line in enumerate(tqdm(handle, total=row_count, desc='Encode ' + name)):
            if row >= row_count:
                raise AssertionError('More cached rows than expected.')
            source_index, label, tokens = json.loads(line)
            ids, features = encode_tokens(tokens, vocabulary, sequence_length)
            arrays['input_ids'][row] = ids
            arrays['labels'][row] = label
            arrays['source_indices'][row] = source_index
            for field, value in features.items():
                arrays[field][row] = value
            seen += 1
    if seen != row_count:
        raise AssertionError('Fewer cached rows than expected.')
    summary = {
        'split': name, 'rows': row_count, 'encoding_key': encoding_key,
        'input_shape': [row_count, sequence_length],
        'class_counts': np.bincount(arrays['labels'], minlength=2).tolist(),
        'empty_after_preprocessing': int(np.count_nonzero(arrays['processed_lengths'] == 0)),
        'truncated_reviews': int(np.count_nonzero(arrays['processed_lengths'] > sequence_length)),
        'mean_encoded_oov_rate': float(np.mean(arrays['oov_rates']))}
    for field, array in arrays.items():
        array.flush()
        array._mmap.close()
    arrays.clear()
    for field, path in paths.items():
        temporary[field].replace(path)
    summary['files_sha256'] = {path.name: sha256_file(path) for path in paths.values()}
    write_json(marker, summary)
    return summary


def derive_slice_definitions(training_lengths, training_oov_rates):
    training_lengths = np.asarray(training_lengths)
    training_oov_rates = np.asarray(training_oov_rates)
    short = int(np.quantile(training_lengths, 0.25, method='higher'))
    medium_upper = int(np.quantile(training_lengths, 0.75, method='higher'))
    oov_threshold = float(np.quantile(training_oov_rates, 0.95, method='higher'))
    return {
        'derived_from': 'training membership only; effective post-truncation token lengths',
        'quantile_method': 'higher', 'short_upper_inclusive': short,
        'medium_upper_inclusive': medium_upper,
        'high_oov_threshold': oov_threshold,
        'high_oov_comparator': '>= threshold and > 0',
        'negation_words': sorted(NEGATIONS),
        'negation_definition': 'At least one preserved negation in retained processed tokens',
        'oov_definition': 'UNK count / encoded non-PAD length; EMPTY is not UNK',
        'empty_review_effective_length': 1,
        'team_comparison_limit': 'Per-slice membership depends on each member tokenizer. '
                                 'Use overall test metrics for direct team comparison.'}


def slice_masks(lengths, oov_rates, negation_flags, definitions):
    lengths, oov_rates = np.asarray(lengths), np.asarray(oov_rates)
    short, upper = definitions['short_upper_inclusive'], definitions['medium_upper_inclusive']
    result = {
        'short_reviews': lengths <= short,
        'medium_reviews': (lengths > short) & (lengths <= upper),
        'long_reviews': lengths > upper,
        'contains_negation': np.asarray(negation_flags, dtype=bool),
        'high_oov_rate': (oov_rates >= definitions['high_oov_threshold']) & (oov_rates > 0)}
    partition = sum(result[name].astype(np.int8) for name in ('short_reviews', 'medium_reviews', 'long_reviews'))
    if not (partition == 1).all():
        raise AssertionError('Length slices must partition all reviews.')
    return result
