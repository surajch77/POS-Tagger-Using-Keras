import sys
import timeit

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, datasets, metrics

from sklearn import metrics
import pickle
import csv

# from MLP_sgd import plot_on_dataset, various_sgd

f1 = sys.argv[1]

#filename = 'final_model.sav'
output_model = sys.argv[2]

# function to define arr with (word, lang, tag)
def make_arr(f1):
    twitter_file = open(f1, "r")
    sentences = []
    sent = []
    for line in twitter_file:
        temp = line.split('\t')
        
        if temp[0] == '\n':
            sentences.append(sent)
            sent = []
            continue

        check = list(temp[2])
        if '\n' in check:
            check.remove('\n')

        temp[2] = ''.join(check)
        sent.append((temp[0], temp[1], temp[2]))


    return sentences

tagged_sentences = make_arr(f1)

print tagged_sentences[:2]

# define the feature of each sentence
def features1(sentence, index):
    #sentence: [(w1, l1), (w2l2) , ...], index: the index of the word  and l1 l2 are word level language tag 
	return {
        'word': sentence[index][0],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0][0].upper() == sentence[index][0][0],
        'is_all_caps': sentence[index][0].upper() == sentence[index][0],
        'is_all_lower': sentence[index][0].lower() == sentence[index][0],
        'prefix-1': sentence[index][0][0],
        'prefix-2': sentence[index][0][:2],
        'prefix-3': sentence[index][0][:3],
        'suffix-1': sentence[index][0][-1],
        'suffix-2': sentence[index][0][-2:],
        'suffix-3': sentence[index][0][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1][0],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1][0],
        'has_hyphen': '-' in sentence[index][0],
        'is_numeric': sentence[index][0].isdigit(),
        'capitals_inside': sentence[index][0][1:].lower() != sentence[index][0][1:],
		'lang' : sentence[index][1],
		'has_@' : sentence[index][0][0] == '@',
		'has_#' : sentence[index][0][0] == '#'
    }
# define the feature of each sentence

def features2(sent, i):
    word = sent[i][0]
    # postag = sent[i][2]
    lang = sent[i][1]
    features = {
        'word' : 'bias',
        'word.lower' : word.lower(),
        'word[-3:]' : word[-3:],
        'word[-2:]' : word[-2:],
        'word[-1:]' : word[-1:],
        'word[0:1]' : word[0:1], 
        'word[0:2]' : word[0:2], 
        'word[0:3]' : word[0:3],
        'word.isupper' : word.isupper(),
        'word.istitle' : word.istitle(),
        'word.isdigit' : word.isdigit(),
        'word.islower' : word.islower(),
        'is_capitalized' : word[0].upper() == word[0],
        'has_hyphen' : '-' in word,
        'capitals_inside' : word[1:].lower() != word[1:],
        'has_@' : word[0] == '@',
        'has_#' : word[0] == '#',
        'lang' : lang,
        # 'postag=' + postag,
        
        # 'lang[:2]=' + lang[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        # postag1 = sent[i-1][2]
        lang1 = sent[i-1][1]
        features1 = {
            '-1:word' : word1,
            '-1:word.lower' : word1.lower(),
            '-1:word.istitle' : word1.istitle(),
            '-1:word.isupper' : word1.isupper(),
            '-1:word[-3:]' : word1[-3:],
            '-1:word[-2:]' : word1[-2:],
            '-1:word[-1:]' : word1[-1:],
            '-1:word[0:1]' : word1[0:1], 
            '-1:word[0:2]' : word1[0:2], 
            '-1:word[0:3]' : word1[0:3],
            '-1:word.islower' : word1.islower(),
            '-1:is_capitalized' : word1[0].upper() == word1[0],
            '-1:has_hyphen' : '-' in word1,
            '-1:capitals_inside' : word1[1:].lower() != word1[1:],
            '-1:has_@' : word1[0] == '@',
            '-1:has_#' : word1[0] == '#',
            # '-1:postag=' + postag1,
            '-1:lang' : lang,
            # '-1:lang[:2]=' + lang[:2],
        }
        features.update(features1)
    else:
        features1 = {'word' : 'BOS'}
        features.update(features1)
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        # postag1 = sent[i+1][2]
        lang2 = sent[i+1][1]
        features2 = {
            '+1:word' : word1,
            '+1:word.lower' : word1.lower(),
            '+1:word.istitle' : word1.istitle(),
            '+1:word.isupper' : word1.isupper(),
            '+1:word[-3:]' : word1[-3:],
            '+1:word[-2:]' : word1[-2:],
            '+1:word[-1:]' : word1[-1:],
            '+1:word[0:1]' : word1[0:1], 
            '+1:word[0:2]' : word1[0:2], 
            '+1:word[0:3]' : word1[0:3],
            '+1:word.islower' : word1.islower(),
            '+1:is_capitalized' : word1[0].upper() == word1[0],
            '+1:has_hyphen=' : '-' in word1,
            '+1:capitals_inside' : word1[1:].lower() != word1[1:],
            '+1:has_@' : word1[0] == '@',
            '+1:has_#' : word1[0] == '#',
            # '+1:postag=' + postag1,
            '+1:lang=' : lang2,
            # '+1:lang[:2]=' + lang[:2],
       	}
       	features.update(features2)
    else:
        features2 = {'word': 'EOS'}
        features.update(features2)
                
    return features

# untag the sentences and return only sentences
def untag(tagged_sentence):
    return [(w, l) for w, l, t in tagged_sentence]


# temp = features1(untag(tagged_sentences[0]), 0) 

temp = features2(untag(tagged_sentences[0]), 0)

# for i in temp:
# 	print i, temp[i]


# Split the dataset for training and testing
L = 10
cutoff =  int(.7 *  L)
#cutoff =  int(.7 *  len(tagged_sentences))

#cutoff =  int(.7 *  len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:L+1]
 
print "Number of Training sentences : ", len(training_sentences)   
print "Number of Testing sentences : ", len(test_sentences) 
print 
def transform_to_matrix(tagged_sentences):
    X, Y = [], []
 
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features1(untag(tagged), index))
            Y.append(tagged[index][2])
 
    return X, Y

X_train, Y_train = transform_to_matrix(training_sentences)
X_test, Y_test = transform_to_matrix(test_sentences)

print "X_train ", len(X_train)
print "Y_train ",len(Y_train)
print "X_test ",len(X_test)
print "Y_test ", len(Y_test)
print 

# call the MLPClassifier
print "MLPClassifier : "
print


MLP = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', MLPClassifier(activation='tanh', learning_rate='constant', solver='sgd', learning_rate_init=0.01,
    	alpha=1e-4, hidden_layer_sizes=(500, 500, 500, 500, 500, 500, 500, 500, 500, 600, 700, 800), random_state=1, batch_size=120, 
    	verbose= True, max_iter=500, warm_start=True, shuffle=True))  
])

# Start the Training
print "Training Started"

start = timeit.default_timer() 

# Train 

MLP.fit(X_train, Y_train)

pickle.dump(MLP, open(output_model, 'wb'))

stop = timeit.default_timer()

Y_pred = MLP.predict(X_test)
tt = zip(Y_test[:20], Y_pred[:20])

for i, j in tt:
	print i + "		" + j

print metrics.classification_report(Y_test, MLP.predict(X_test))
print "Accuracy : ",  metrics.accuracy_score(Y_test, Y_pred)

print 'Training completed ', (stop - start),  " sec"

# loaded_model = pickle.load(open(output_model, 'rb'))
# result = loaded_model.score(X_test, Y_test)

# Y_pred = loaded_model.predict(X_test)
# tt = zip(Y_test[:20], Y_pred[:20])

# for i, j in tt:
# 	print i, j

# print result