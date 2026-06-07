import math
import re
import random
from collections import defaultdict

class AI20Engine:
def __init__(self, context_window=10, decay_factor=0.95):
self.context_window = context_window
self.decay_factor = decay_factor
self.word_to_idx = {}
self.idx_to_word = {}
self.next_free_index = 1
self.word_to_meta = defaultdict(list)
self.meta_to_words = defaultdict(list)
self.token_frequencies = defaultdict(int)
self.total_processed_tokens = 0
self.distance_matrix = defaultdict(float)
self.meta_distance_matrix = defaultdict(float)

def register_meta_category(self, meta_name: str, words: list[str]):
meta_idx = hash(meta_name) & 0xFFFFFFFF
for word in words:
w_idx = self._get_or_create_index(word.lower().strip())
if meta_idx not in self.word_to_meta[w_idx]:
self.word_to_meta[w_idx].append(meta_idx)
self.meta_to_words[meta_idx].append(w_idx)

def _get_or_create_index(self, word: str) -> int:
if word not in self.word_to_idx:
self.word_to_idx[word] = self.next_free_index
self.idx_to_word[self.next_free_index] = word
self.next_free_index += 1
return self.word_to_idx[word]

def _clean_and_parse(self, text: str) -> list[str]:
text = text.lower().strip()
cleaned = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"\']', '', text)
return cleaned.split()

def tokenize_sequence(self, raw_text: str) -> list[int]:
words = self._clean_and_parse(raw_text)
return [self._get_or_create_index(w) for w in words]

def _get_omega(self, idx: int) -> float:
if self.total_processed_tokens == 0:
return 1.0
freq = self.token_frequencies[idx]
prob = float(freq) / float(self.total_processed_tokens)
return max(0.01, 1.0 - prob)

def mirror_training_corpus(self, training_data: str):
indices = self.tokenize_sequence(training_data)
sequence_length = len(indices)

for idx in indices:
self.token_frequencies[idx] += 1
self.total_processed_tokens += sequence_length

if sequence_length < 2:
return

for i in range(sequence_length):
idx_a = indices[i]
omega_a = self._get_omega(idx_a)
lookahead_limit = min(sequence_length, i + self.context_window + 1)

for j in range(i + 1, lookahead_limit):
idx_b = indices[j]
omega_b = self._get_omega(idx_b)
spatial_distance = j - i

p_weight = (omega_a * omega_b) / float(spatial_distance)

self.distance_matrix[(idx_a, idx_b)] += p_weight
self.distance_matrix[(idx_b, idx_a)] += p_weight

meta_a_list = self.word_to_meta[idx_a]
meta_b_list = self.word_to_meta[idx_b]

for m_a in meta_a_list:
self.meta_distance_matrix[(m_a, idx_b)] += p_weight * 0.5
self.meta_distance_matrix[(idx_b, m_a)] += p_weight * 0.5
for m_b in meta_b_list:
self.meta_distance_matrix[(idx_a, m_b)] += p_weight * 0.5
self.meta_distance_matrix[(m_b, idx_a)] += p_weight * 0.5
for m_a in meta_a_list:
for m_b in meta_b_list:
self.meta_distance_matrix[(m_a, m_b)] += p_weight * 0.25
self.meta_distance_matrix[(m_b, m_a)] += p_weight * 0.25

def generate_response(self, prompt: str, max_generation_tokens: int = 15, temperature: float = 0.0) -> str:
active_sequence = self.tokenize_sequence(prompt)
if not active_sequence:
return "[System Alert: Empty Index Space Detected]"

for _ in range(max_generation_tokens):
last_token_idx = active_sequence[-1]
candidate_pool = {}

for candidate_idx in self.idx_to_word.keys():
if len(active_sequence) >= 2 and candidate_idx == active_sequence[-2]:
continue

base_link_weight = self.distance_matrix[(last_token_idx, candidate_idx)]

last_meta_list = self.word_to_meta[last_token_idx]
cand_meta_list = self.word_to_meta[candidate_idx]

for m_l in last_meta_list:
base_link_weight += self.meta_distance_matrix[(m_l, candidate_idx)] * 0.3
for m_c in cand_meta_list:
base_link_weight += self.meta_distance_matrix[(last_token_idx, m_c)] * 0.3

if base_link_weight == 0.0:
for m_l in last_meta_list:
for m_c in cand_meta_list:
transitive_bridge = self.meta_distance_matrix[(m_l, m_c)]
if transitive_bridge > 0.0:
base_link_weight += transitive_bridge * 0.1

if base_link_weight == 0.0:
continue

historical_gravity_score = 0.0
history_length = len(active_sequence) - 1

for history_pos in range(history_length):
historical_token_idx = active_sequence[history_pos]
steps_back = history_length - history_pos + 1

cross_attraction = self.distance_matrix[(historical_token_idx, candidate_idx)]
hist_meta_list = self.word_to_meta[historical_token_idx]

for m_h in hist_meta_list:
cross_attraction += self.meta_distance_matrix[(m_h, candidate_idx)] * 0.3
for m_c in cand_meta_list:
cross_attraction += self.meta_distance_matrix[(historical_token_idx, m_c)] * 0.3

historical_gravity_score += (cross_attraction / float(steps_back)) * (self.decay_factor ** steps_back)

total_path_attraction = base_link_weight + historical_gravity_score
candidate_pool[candidate_idx] = total_path_attraction

if not candidate_pool:
break

if temperature <= 0.0:
selected_next_idx = max(candidate_pool, key=candidate_pool.get)
else:
candidates = list(candidate_pool.keys())
raw_weights = []
for c in candidates:
try:
scaled_score = math.exp(candidate_pool[c] / temperature)
except OverflowError:
scaled_score = float('inf')
raw_weights.append(scaled_score)
if sum(raw_weights) == 0:
selected_next_idx = max(candidate_pool, key=candidate_pool.get)
else:
selected_next_idx = random.choices(candidates, weights=raw_weights, k=1)

active_sequence.append(selected_next_idx)

output_words = [self.idx_to_word[idx] for idx in active_sequence]
return " ".join(output_words)
