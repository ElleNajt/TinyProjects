import numpy as np
import scipy
from scipy import spatial
import random

def noise(list, p):
	#If you want to make voters have some probability of getting confused; making them less perceptive.
	newlist = []
	for x in list:
		if np.random.rand(1)[0] < p:
			newlist.append(-x)
		else:
			newlist.append(x)
	return newlist

dim = 3
sample_size = 400

sample = list(np.random.normal(0,1,[sample_size,dim]))
true_classifier = np.random.normal(0,1,dim)
true_classes = [np.sign(np.dot(true_classifier,x)) for x in sample]

voters = []
m = 3
for k in range(100):
	worldview = random.sample(sample,m)
	classes = [np.sign(np.dot(true_classifier, x)) for x in worldview]
	p = .2
	#Increase p (say to .2) to make voters less perceptive.
	classes = noise(classes,p)
	voter_classifier = np.linalg.lstsq(worldview,classes)[0]
	voters.append(voter_classifier)
	
population_subconscious_guess = np.mean(voters,0)


test = list(np.random.normal(0,1,[sample_size,dim]))
test_true_types = [np.sign(np.dot(true_classifier, x)) for x in test]

subconscious_classes = [np.sign(np.dot(population_subconscious_guess, x)) for x in test]
subconscious_errors = scipy.spatial.distance.hamming(subconscious_classes, test_true_types)

def majority_vote(voters, vector):
	votes = [np.sign(np.dot(k, vector)) for k in voters]
	value = np.sign(np.mean(votes))
	return value

test_majority_vote = [majority_vote(voters, vector) for vector in test]
majority_vote_errors = scipy.spatial.distance.hamming(test_majority_vote, test_true_types)

random_voters = random.sample(voters, 50)
individual_voter_accuracy = []

for x in random_voters:
	ballot = [np.sign(np.dot(x, k)) for k in test]
	true = [np.sign(np.dot(true_classifier, k)) for k in test]
	accuracy = scipy.spatial.distance.hamming(ballot, true)
	individual_voter_accuracy.append(accuracy)
	

majority_vs_subconscious = scipy.spatial.distance.hamming(test_majority_vote, subconscious_classes)
	
print("Probability of errors in subconscious vector", subconscious_errors)
print("Probability of errors in majority voting", majority_vote_errors)
print("Probability of errors on average for random individual voters", np.mean(individual_voter_accuracy))
print("Probability that subconscious vector and majority vote disagree", majority_vs_subconscious)

#Example run, when m = 3 and dim = 5.

# ## -- End pasted text --
# Probability of errors in subconscious vector 0.065
# Probability of errors in majority voting 0.056
# Probability of errors on average for random individual voters 0.2975
# Probability that subconscious vector and majority vote disagree 0.038

#With p = .2, so that voters have a chance of being mis-educated about some of their observations.
# ## -- End pasted text --
# Probability of errors in subconscious vector 0.144
# Probability of errors in majority voting 0.069
# Probability of errors on average for random individual voters 0.38794
# Probability that subconscious vector and majority vote disagree 0.119

##The general pattern is that majority voting and the subconscious vector work roughly as well. In any case, in practice we don't have access to the collective subconscious (in Rousseou, the 'will of the people'), so we use majority voting. Note that m = 3, so each voter here is incredibly ignorant, as is reflected in the error probability for a random voter. However, each voter is smart, because it is least squares optimizing its classification vector based on its worldview, and choosing the vector of least norm that works.

#(Note; if you show that a random voter has a probability > 1/2 of classifying new samples correctly, then the law of large numbers implies that majority voting converges to a perfect classifier. Showing that random voters achieve this if they have large enough world views is a combinatorics / calculus exercise - it shouldn't be too hard, though I haven't worked it out. It's experimentally true.)

#There are some configurations when majority voting seems to work *way* better than the subconscious vote. For example, here p = .2, m = 3, dim = 3, sample_size = 400, and there are 100 voters.... so, here we have a lot of ignorant, sometimes sloppily observant voters exploring the world, but somehow their democratic process manages to make good decisions.

# ## -- End pasted text --
# Probability of errors in subconscious vector 0.375
# Probability of errors in majority voting 0.06
# Probability of errors on average for random individual voters 0.40125
# Probability that subconscious vector and majority vote disagree 0.405
