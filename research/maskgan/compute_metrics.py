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
	return bleu_score.sentence_bleu([reference], predict, weights)

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

	return gleu_score.sentence_gleu(reference, predict)

def ribes(reference, hypothesis):
	"""Compute sentence-level ribes score with kendall_tau correlation.
	Args:
		reference (list[str])
		hypothesis (list[str])
	"""
	from nltk.translate import ribes_score
	import math

	try: 
		best_ribes = ribes_score.sentence_ribes(reference, hypothesis)
	except ZeroDivisionError:
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


	try:
		result=chrf_score.sentence_chrf(reference, predict)
	except ZeroDivisionError:
		result=0.0
	
	return result

def main(step_size=1000, max_step=200000):
	import numpy as np
	# set file paths
	import os
	cwd = os.getcwd()
	REF_PATH=cwd+'/generated-outputs/original_inputs.txt'

	predict_paths = []
	i = 0
	while i < max_step:
		print('adding ', cwd+'/generated-outputs/reviews-'+str(i)+'.txt')
		predict_paths.append(cwd+'/generated-outputs/reviews-'+str(i)+'.txt')
		i += step_size

	# parse corpus into sentences with 1:1 comparison
	ref = parse_sentence(REF_PATH)
	pred_list = [parse_sentence(single_path) for single_path in predict_paths]

	# print(ref[:5], '\n\n', pred[:5])
	print('len(ref)', len(ref))

	all_bleus = []
	all_gleus = []
	all_ribes = []
	all_chrfs = []
	for pred in pred_list:
		print('len(pred) ', len(pred))
		bleus = []
		gleus = []
		ribeses = []
		chrfs = []
		for item in range(0, len(ref)):
			# print('ref[item] ', ref[item])
			# print('pred[item] ', pred[item])
			bleus.append(bleu(ref[item], pred[item]))
			gleus.append(gleu(ref[item], pred[item]))
			ribeses.append(ribes(ref[item], pred[item]))
			chrfs.append(chrf(ref[item], pred[item]))
		all_bleus.append(np.mean(bleus))
		all_gleus.append(np.mean(gleus))
		all_ribes.append(np.mean(ribeses))
		all_chrfs.append(np.mean(chrfs))

	print('len(all_bleus) ', len(all_bleus))
	print('all_bleus ', all_bleus)

	print('len(all_gleus) ', len(all_gleus))
	print('all_gleus ', all_gleus)

	print('len(all_ribes) ', len(all_ribes))
	print('all_ribes ', all_ribes)

	print('len(all_chrfs) ', len(all_chrfs))
	print('all_chrfs ', all_chrfs)

	with open('bleu_metric.txt', 'w') as f:
		for item in all_bleus:
			f.write(str(item)+'\n')
		f.close()
	with open('gleu_metric.txt', 'w') as f:
		for item in all_gleus:
			f.write(str(item)+'\n')
		f.close()
	with open('ribes_metric.txt', 'w') as f:
		for item in all_ribes:
			f.write(str(item)+'\n')
		f.close()
	with open('chrf_metric.txt', 'w') as f:
		for item in all_chrfs:
			f.write(str(item)+'\n')
		f.close()

		
if __name__== "__main__":
  main()










