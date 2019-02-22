import tweepy
import csv
import collections
import nltk.classify.util, nltk.metrics
from nltk.metrics import precision as precision
from nltk.metrics import recall as recall
from nltk.metrics import f_measure as f_measure

consumer_key = "Enter your key"
consumer_secret = "Enter your secret"
access_key = "Enter your key"
access_secret = "Enter your secret"
// create an API token from your Tweepy account get get those credentials.

def get_all_tweets(Screen_name):

  
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)


    alltweets = []  

    new_tweets = api.user_timeline(Screen_name = Screen_name,count=200)

    alltweets.extend(new_tweets)

    oldest = alltweets[-1].id - 1


    c=1
    while c > 0:
        print("getting tweets before %s" % (oldest))

        new_tweets = api.user_timeline(Screen_name = Screen_name,count=200,max_id=oldest)

        alltweets.extend(new_tweets)

        oldest = alltweets[-1].id - 1

#print("...%s tweets downloaded so far" % (len(alltweets)))
        c=0 

    outtweets = [[tweet.text.encode("utf-8")] for tweet in alltweets]

-    #write the csv  
    with open('%s_tweets.csv' %Screen_name, 'w') as f:
        writer = csv.writer(f)
        #writer.writerow(["text"])
        writer.writerows(outtweets)

#get_all_tweets(NarendraModi)
#get_all_tweets(realDonaldTrump)
#get_all_tweets(imVkohli)


# load text
get_all_tweets("NarendraModi_tweets")

filename = 'NarendraModi_tweets.csv'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split("\n")
words = [word.lower() for word in words]
print(words[:100])
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in words]
print(stripped[:100])

# create cleaned file
file = open("cleaned.csv","w")
for i in stripped:
    file.write(i+"\n")
    
file.close() 

New_tweets=[]
with open('cleaned.csv', 'r') as myfile:    
    reader = csv.reader(myfile)
    for val in reader:
      try:  
        New_tweets.append(val[0])
      except:
          print("empty list encountered")
          
for i in New_tweets:
    print(i)
    
def word_split(data):    
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new    

def word_feats(words):    
    return dict([(word, True) for word in words])


my_feats  = [(word_feats(f), 'NA') for f in word_split(New_tweets)]
negfeats = [(word_feats(f), 'neg') for f in word_split(New_tweets)]
posfeats = [(word_feats(f), 'pos') for f in word_split(New_tweets)]
        
negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]


testsets_New_tweets = collections.defaultdict(set)
refsets = collections.defaultdict(set)

classifier = nltk.NaiveBayesClassifier.train(trainfeats)

for i, (feats, label) in enumerate(trainfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets_New_tweets[observed].add(i)
        accuracy = nltk.classify.util.accuracy(classifier,trainfeats)
pos_precision = nltk.precision(refsets['pos'], testsets_New_tweets['pos'])

pos_recall = recall(refsets['pos'], testsets_New_tweets['pos'])
pos_fmeasure =f_measure(refsets['pos'], testsets_New_tweets['pos'])
neg_precision =precision(refsets['neg'], testsets_New_tweets['neg'])
neg_recall = recall(refsets['neg'], testsets_New_tweets['neg'])
neg_fmeasure = f_measure(refsets['neg'], testsets_New_tweets['neg'])


print(len(testsets_New_tweets['pos']))
print(len(testsets_New_tweets['neg']))
p=(testsets_New_tweets['pos'])

n=(testsets_New_tweets['neg'])

(p/(p+n)*100)

(n/(n+p)*100)


