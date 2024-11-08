import spacy
from spacy.training import Example
from spacy.util import minibatch
import random
from data_spacy.train_data_name import TRAIN_DATA

nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")
ner.add_label("PERSON")

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    for itn in range(80):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=spacy.util.compounding(4.0, 32.0, 1.001))  # Example batching with compounding batch sizes
        for batch in batches:
            texts, annotations = zip(*batch)
            docs = [nlp.make_doc(text) for text in texts]
            examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(docs, annotations)]
            nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)  # Adjust drop as needed
        print(losses)

nlp.to_disk("~/rasa_ws/spacy_model/model_name")
