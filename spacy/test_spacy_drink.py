import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("models/model_drink")

# Test sentence
test_sentence = "i want a cup of milk please."

# Process the sentence
doc = nlp(test_sentence)

# Print detected entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)
