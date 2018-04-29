
# set file paths
REF_PATH='/Users/luciayu/Documents/maskgan/original_inputs.txt'
PREDICT_PATH='/Users/luciayu/Documents/maskgan/reviews.txt'

###################
def parse_sentence(filepath):
	result = []
	with open(filepath, 'r') as f:
		lines = f.readlines()
		for line in lines:
			newline = []
			for word in line.split():
				newline.append(word)
			result.append(newline)
	return result


def bleu(reference, predict):
	"""Compute sentence-level bleu score.

	Args:
		reference (list[str])
		predict (list[str])
	"""
	from nltk.translate import bleu_score

	if len(predict) == 0:
		if len(reference) == 0:
			return 1.0
		else:
			return 0.0

	# TODO(kelvin): is this quite right?
	# use a maximum of 4-grams. If 4-grams aren't present, use only lower n-grams.
	n = min(4, len(reference), len(predict))
	weights = tuple([1. / n] * n)  # uniform weight on n-gram precisions
	return bleu_score.sentence_bleu([reference], predict, weights, emulate_multibleu=True)

def gleu(reference, predict):
	"""Compute sentence-level gleu score.

	Args:
		reference (list[str])
		predict (list[str])
	"""
	from nltk.translate import gleu_score

	if len(predict) == 0:
		if len(reference) == 0:
			return 1.0
		else:
			return 0.0

	return gleu_score.sentence_gleu([reference], predict)

def ribes(reference, hypothesis):
	"""Compute sentence-level ribes score with kendall_tau correlation.
	Args:
		reference (list[str])
		hypothesis (list[str])
	"""
	from nltk.translate import ribes_score
	import math

	# if both are empty return full score, else return zero
	if len(hypothesis) == 0:
		if len(reference) == 0:
			return 1.0
		else:
			return 0.0

	# ribes_score.sentence_ribes modified for single reference and hypothesis
	# found at: http://www.nltk.org/_modules/nltk/translate/ribes_score.html#sentence_ribes
	best_ribes = -1.0
	alpha = 0.25
	beta = 0.10
	
	# Collects the *worder* from the ranked correlation alignments.
	worder = ribes_score.word_rank_alignment(reference, hypothesis)

	# if worder matches are sparse, then spearman_rho correlation will be NaN
	# rho = 1 - sum_d_square / choose(worder_len+1, 3)
	if len(worder) < 2:
		return 0.0
	else:
		nkt = ribes_score.spearman_rho(worder)
		
	# Calculates the brevity penalty
	bp = min(1.0, math.exp(1.0 - len(reference)/len(hypothesis)))
	
	# Calculates the unigram precision, *p1*
	p1 = len(worder) / len(hypothesis)
	
	_ribes = nkt * (p1 ** alpha) *  (bp ** beta)
	
	if _ribes > best_ribes: # Keeps the best score.
		best_ribes = _ribes
		
	return best_ribes

def chrf(reference, predict):
	"""Compute sentence-level chrf score.

	Args:
		reference (list[str])
		predict (list[str])
	"""
	from nltk.translate import chrf_score

	if len(predict) == 0:
		if len(reference) == 0:
			return 1.0
		else:
			return 0.0

	return chrf_score.sentence_chrf(reference, predict)

def main():
	import numpy as np
	# parse corpus into sentences with 1:1 comparison
	ref = parse_sentence(REF_PATH)
	pred = parse_sentence(PREDICT_PATH)
	# print(ref[:5], '\n', pred[:5])

	bleus = []
	gleus = []
	ribeses = []
	chrfs = []
	for item in range(0, len(ref)):
		bleus.append(bleu(ref[item], pred[item]))
		gleus.append(gleu(ref[item], pred[item]))
		ribeses.append(ribes(ref[item], pred[item]))
		# chrfs.append(chrf(ref[item], pred[item]))

	print(np.mean(bleus))
	print(np.mean(gleus))
	print(np.mean(ribeses))
	# print(np.mean(chrfs))
		
if __name__== "__main__":
  main()










