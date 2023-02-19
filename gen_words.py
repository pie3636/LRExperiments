# <ID>  <SET_TYPE>  <KEY_WORD>  <RELATION>  <TARGET_WORD1>  <TARGET_WORD2>  ...
# <ANNOTATED_WORD> := <WORD> (<POS>,<FREQ>,<COUNT>)

import random
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings

from nltk.corpus import wordnet as wn
from tqdm import tqdm
from transformers import AutoTokenizer
from wordfreq import zipf_frequency

random.seed(42)

tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-large-cased')
vocab = tokenizer.get_vocab().keys()
word2count = {}
str_alphabet = 'abcdefghijklmnoprstuwyząćęłńóśżź'

out_file = open('WNLaMPro_pl.txt', 'w')
id_counter = 0

def save_data(word, word_pos, word_data, word_count, out_file):
	global id_counter
	for relation, targets in word_data.items():
		if targets:
			set_type = 'dev' if random.randint(1, 10) == 1 else 'test'
			freq = zipf_frequency(word, 'fr')
			out_str = f'{id_counter}\t{set_type}\t{word} ({word_pos},{freq},{word_count})\t{relation}\t'
			for name, pos, count in targets:
				freq = zipf_frequency(name, 'fr')
				out_str += f'{name} ({pos},{freq},{count})\t'
			id_counter += 1
			out_str = out_str[:-1]
			out_file.write(out_str + '\n')

# Count frequencies
for line in tqdm(open('words_plwiki-2022-08-29.txt'), desc='Counting frequencies'):
	word, count = line.split(' ')
	word2count[word] = int(count)

for word in tqdm(word2count, desc='Creating dataset'):
	word_save = word
	synsets = wn.synsets(word, lang='pol')
	word_data = {'antonym': [], 'hypernym': [], 'cohyponym': [], 'corruption': []}

	# Sort synsets per frequency
	sorted_ss = []
	for s in synsets:
		if s.pos() not in ['n', 'a']:
			continue
		freq = 0
		for name in s.lemma_names(lang='pol'):
			if '_' in name or name not in word2count:
				continue
			freq += word2count[name]
		if freq:
			sorted_ss.append((freq, s))
	if not sorted_ss:
		continue

	sorted_ss.sort()
	best_freq, best_ss = sorted_ss[0]

	# Get antonyms of lemmas in most frequent synset
	antonyms = set()
	for synlemma in best_ss.lemmas():
		antlist = synlemma.antonyms()
		if antlist:
			for antonym in antlist:
				for lemma in antonym.synset().lemmas(lang='pol'):
					name = lemma.name()
					# Add only once per lemma name, and keep intersection between BERT vocab and corpus
					# Keep single-word lemma names only
					if '_' not in name and name in word2count and name not in antonyms and name in vocab:
						antonyms.add(name)
						word_data['antonym'].append((name, lemma.synset().pos(), word2count[name]))

	# Get hypernyms and cohyponyms from most two frequent synsets
	hypernyms = set()
	cohyponyms = set()
	for s_freq, s in sorted_ss[:2]:
		for path in s.hypernym_paths():
			# Take only hypernyms of depth >= 6 and path lengths <= 3
			for hyp in path[5:][-4:-1]:
				for lemma in hyp.lemmas(lang='pol'):
					name = lemma.name()
					if '_' not in name and name in word2count and (lemma, word2count[name]) not in hypernyms and name in vocab:
						hypernyms.add((lemma, word2count[name]))
				# Take only hypernyms of depth >= 6 and path lengths <= 2
				if hyp not in path[5:][-4:-3]:
					with warnings.catch_warnings():
						warnings.filterwarnings("ignore", category=UserWarning)
						# And get hyponyms of depth <= 4
						for hyponym in hyp.closure(lambda x: x.hyponyms(), depth=4):
							for lemma in hyponym.lemmas(lang='pol'):
								name = lemma.name()
								if '_' not in name and name in word2count and (lemma, word2count[name]) not in cohyponyms and name in vocab:
									cohyponyms.add((lemma, word2count[name]))

	# Keep from 3 to 30 hypernyms, sorted by frequency
	if len(hypernyms) >= 3:
		hypernyms = sorted(hypernyms, key=lambda x: x[1], reverse=True)[:20]
		for lemma, freq in hypernyms:
			name = lemma.name()
			word_data['hypernym'].append((name, lemma.synset().pos(), word2count[name]))

	# Keep from 10 to 50 hypernyms, sorted by frequency
	if len(cohyponyms) >= 10:
		cohyponyms = sorted(cohyponyms, key=lambda x: x[1], reverse=True)[:50]
		for lemma, freq in cohyponyms:
			name = lemma.name()
			word_data['cohyponym'].append((name, lemma.synset().pos(), word2count[name]))

	# Corruptions
	if word2count[word] >= 100 and word in vocab:
		operation = random.randint(1, 3)
		if operation == 1: # Insertion
			pos = random.randint(0, len(word) + 1)
			word = word[:pos] + random.choice(str_alphabet) + word[pos:]
			word_data['corruption'].append((word, best_ss.pos(), 0))
		elif operation == 2: # Deletion
			if len(word) >= 2:
				pos = random.randint(0, len(word) - 1)
				word = word[:pos] + word[pos + 1:]
				word_data['corruption'].append((word, best_ss.pos(), 0))
		else: # Swap
			if len(word) >= 2:
				pos = random.randint(0, len(word) - 2)
				word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
				word_data['corruption'].append((word, best_ss.pos(), 0))

	save_data(word_save, best_ss.pos(), word_data, word2count[word_save], out_file)

out_file.close()
