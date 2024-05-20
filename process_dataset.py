import sys
import json


def extract_entities(sentence):
    # Extract entities from a sentence
    # Tagging scheme: IOB
    # Entities: person, location, group, corporation, product, creative-work
    # Return: { "name": <joined words of the entity>, "type": <entity type without IOB tag> }
    entities = []
    current_entity = None

    for word, tag in sentence:
        if tag == "O":
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
        else:
            iob, entity_type = tag.split("-", 1)
            if iob == "B":
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {"name": word, "type": entity_type}
            else:
                assert current_entity is not None
                current_entity["name"] += " " + word

    if current_entity is not None:
        entities.append(current_entity)

    return entities


sentences = []
current_sentence = []

for line in sys.stdin:
    if line == "\n":
        sentences.append(
            {
                "sentence": " ".join([w for w, t in current_sentence]),
                "entities": extract_entities(current_sentence),
            }
        )
        current_sentence = []
        print(json.dumps(sentences[-1]))
    else:
        word, tag = line.strip().split("\t", 1)
        current_sentence.append((word, tag))
