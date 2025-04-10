# Install NLTK (if not installed)
!pip install nltk

#Import required libraries
import nltk

# Download required datasets
nltk.download('punkt') # For tokenization
nltk.download('universal_tagset') # For general POS tags
nltk.download('indian') # For Indian Language datasets (including Marathi)
# Import necessary modules
from nltk.corpus import indian # Load Indian Language corpus
from nltk.tag import BigramTagger # Import Bigrom POS Tagger

#Load Marathi POS-tagged dataset from the Indian corpus
marathi_sents = indian.tagged_sents('marathi.pos')

#Split the dataset into training (90%) and testing (10%) sets
train_marathi = marathi_sents[:int(len(marathi_sents) * 0.9)] # Training data (90%)
test_marathi = marathi_sents[int(len(marathi_sents) * 0.9):]  # Testing data (10%)

#Create a Bigram Tagger using training data
bigram_tagger_marathi = BigramTagger(train_marathi)
# Example Marathi sentence (code-mixed with English)
marathi_sentence = "भारत एक महान इंडिया आहे"

#Tokenize the sentence (split into words)
tokenized_marathi = nltk.word_tokenize(marathi_sentence)

#Apply the trained Bigram POS Tagger
tagged_marathi = bigram_tagger_marathi.tag(tokenized_marathi)

print("Tagged Marathi Sentence:", tagged_marathi)
from nltk.tag import UnigramTagger, DefaultTagger

#Define a default tagger (assigns 'NN' if nothing is known)
default_tagger = DefaultTagger('NN')

#Train a Unigram Tagger with backoff to the Default Tagger
unigram_tagger = UnigramTagger(train_marathi, backoff=default_tagger)

#Train a Bigram Tagger with backoff to the Unigram Tagger
bigram_tagger_marathi = BigramTagger(train_marathi, backoff=unigram_tagger)
import stanza
#DownLoad and initialize the Hindi model
stanza.download('hi') # Download the Hindi model (anly needs to be run ance)
nlp = stanza.Pipeline('hi') # Create the pipeline for Hindi Language
#List of sentences for testing, including those with Proper Nouns (PN) and English words
sentences = [
    "इंडिया एक महान देश है",  # India is a great country
    "राजीव गांधी भारत के प्रधानमंत्री थे",  # Rajiv Gandhi was the Prime Minister of India
    "मुझे मुंबई बहुत पसंद है",  # I really Like Mumbai
    "शाहरुख़ ख़ान एक प्रसिद्ध अभिनेता हैं",  # Shah Rukh Khan is a famous actor
    "गांधी जी ने भारतीय स्वतंत्रता संग्राम में महत्वपूर्ण भूमिका निभाई", #Gandhi ji played an important role in India's freedom struggl
    "नरेंद्र मोदी वर्तमान प्रधानमंत्री हैं",  # Narendra Modi is the current Prime Minister
    "दिल्ली भारत की राजधानी है",  # Delhi is the capital of India
    "ताज महल भारत के प्रमुख स्मारकों में से एक है",  # The Taj Mahal is one of India's major monuments
    "भारत और पाकिस्तान के बीच संबंध जटिल हैं",  # The relationship between India and Pakistan is complicated
    "भारत के स्वतंत्रता संग्राम में सुभाष चंद्र बोस की भूमिका महत्वपूर्ण थी",  # Subhas Chandra Bose's role in India's freedom struggle w
    "I love भारत और मैंने मुंबई में छुट्टियाँ बिताई", # I Love India and I spent holidays in Mumbai
    "शाहरुख़ ख़ान is a famous actor",  # Shah Rukh Khan is a famous actor
    "भारत का भारत सरकार has implemented new policies",  # The Indian government has implemented new policies
    "आजकल, में online classes attend कर रहा हूँ",  # These days, I am attending online classes
    "The Taj Mahal is located in भारत",  # The Taj Mahal is located in India
    "I want to visit दिल्ली someday",  # I want to visit Delhi someday
    "Mumbai is a major financial center of भारत",  # Mumbai is a major financial center of India

]
#Process and print tokenized and tagged words for each sentence
for hindi_sentence in sentences:
    doc = nlp(hindi_sentence)  # Process the sentence using the pipeline
    print(f"\nTagged Hindi Sentence: '{hindi_sentence}'")
    for sentence in doc.sentences:
        for word in sentence.words:
            print(f"({word.text}): {word.upos}")
# Define some manually tagged Minglish sentences for training
minglish_tagged_sentences = [
    [('इंडिया', 'NNP'), ('ची', 'PREP'), ('कल्चर', 'NN'), ('खूप', 'RB'), ('रिच', 'JJ'), ('आहे', 'VBP')],
    [('माझ्या', 'PRP$'), ('मुंबईच्या', 'NNP'), ('नाइटलाइफला', 'NN'), ('खूप', 'RB'), ('आवड', 'NN'), ('आहे', 'VBP')],
    [('क्या', 'PRP'), ('तुमने', 'PRP'), ('चेन्नईची', 'NNP'), ('बिर्याणी', 'NN'), ('ट्राय', 'VBP'), ('केली', 'VBP')]
]
#Combine new sentences with the existing training dataset
extended_train_data = train_marathi + minglish_tagged_sentences

#Train a new Bigram POS Tagger
bigram_tagger_marathi = BigramTagger(extended_train_data, backoff=unigram_tagger)
import nltk
from nltk.corpus import indian
from nltk.tag import BigramTagger, UnigramTagger, DefaultTagger

#Load the Marathi POS-tagged dataset
marathi_sents = indian.tagged_sents('marathi.pos')
#Define Backoff Taggers
default_tagger = DefaultTagger('NN')  # Assigns 'NN' to unknown words
unigram_tagger = UnigramTagger(train_marathi, backoff=default_tagger)  # Single-word tagger
bigram_tagger_marathi = BigramTagger(train_marathi, backoff=unigram_tagger)  # Trained Bigram Tagger
#Define a sample sentence (Minglish)
sentence = "मी ऑनलाइन क्लासेस अटेंड करत आहे."
#Tokenize properly
tokenized_sentence = nltk.word_tokenize(sentence)
#Apply the improved tagger
tagged_sentence = bigram_tagger_marathi.tag(tokenized_sentence)
#Print the output
for word, tag in tagged_sentence:
    print(word, tag)
# Define a List of test sentences (MingLish format)
sentences = [
    "इंडिया ची कल्चर खूप रिच आहे.",  # India's culture is very rich.
    "माझ्या मुंबईच्या नाइटलाइफला खूप आवड आहे.",  # I Love Mumbai's nightlife.
    "या वर्षी दिवालीवर आम्ही गोव्यात जात आहोत.",  # This year we are going to Goa for Diwali.
    "क्या तुमने चेन्नईची बिर्याणी ट्राय केली का?",  # Have you tried Chennal's biryani?
    "दिल्लीचा वेदर आज खूप प्लेजंट आहे.",  # Delhi's weather is very pleasant today.
    "मी काल बेंगळुरूवरून फ्लाइट घेतली होती.",  # I took a flight from Bengaluru yesterday.
    "इंडिया ची क्रिकेट टीम खूप स्ट्रॉग आहे.",  # India's cricket team is very strong.
    "मी वीकेंडवर पुण्यात जात आहे.",  # I am going to Pune on the weekend.
    "क्या तुम्हें आगरा का ताजमहल पसंद आहे?",  # Do you Like the Taj Mahal in Agra?
    "मी ऑनलाइन क्लासेस अटेंड करत आहे.",  # I am attending online classes.
]
# Process each sentence: tokenize, tag, and print results
for marathi_sentence in sentences:
    tokenized_sentence = nltk.word_tokenize(marathi_sentence) #tokenization
    tagged_sentence = bigram_tagger_marathi.tag(tokenized_sentence)  # POS tagging
    print(f"\nTagged Sentence: '{marathi_sentence}'")
    print(tagged_sentence)
# Evaluate the accuracy of the Bigram Tagger using the test dataset
accuracy = bigram_tagger_marathi.evaluate(test_marathi)

print("\nMarathi Bigram Tagger Accuracy:", accuracy)

#Print final status
print("\nModels saved successfully!")
