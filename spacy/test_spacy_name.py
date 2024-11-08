import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("models/model_name")

# Test sentence
test_sentence = "this is luca and luka"

# Process the sentence
doc = nlp(test_sentence)

# Print detected entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)
