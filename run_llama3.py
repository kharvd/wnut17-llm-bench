import openai
from openai import RateLimitError
import json
import argparse
import tqdm
import sys
import os
import backoff

from concurrent.futures import ThreadPoolExecutor, as_completed

client = openai.OpenAI(
    base_url="https://api.together.xyz/v1/",
)


SYSTEM_PROMPT = """\
The user will provide a sentence. You must process the sentence and extract a list of entities of the following categories: person, location, group, corporation, product, creative-work.

# Categories
## 1. Person

Names of people (e.g. **Virginia Wade**). Don't mark people that don't have their own name. Include punctuation in the middle of names. Fictional people can be included, as long as they're referred to by name (e.g. "Harry Potter").

#### Examples:

  * "There was a celebration for **Sarkozy**" _correct - Sarkozy is a person's name_

  * "The **driver** is lost" _wrong, because driver isn't the name of a particular person_

  * "We ate with **Ringo Starr** and **he** didn't do any impressions" _Only mark the names of people, not words that just refer to them_

  * "**Dell** shares are up 17%" _In this case, Dell refers to a company, not a person_

  * "The award went to **Ben** and **Jerry**" _wrong, because here Ben and Jerry is an organisation_

  * "I'm so glad @ **MileyCyrus** won the award" _correct - this refers to a person by name_


## 2. Location 

Names that are locations (e.g. **France**). Don't mark locations that don't have their own name. Include punctuation in the middle of names. Fictional locations can be included, as long as they're referred to by name (e.g. "Hogwarts").

#### Examples:

  * "There was a celebration in **London**" _Correct - London is a location name_

  * "The **Eiffel** **Tower** is over 300 metres tall" _Correct - both words of the location are marked_

  * "The **room** is empty" _Wrong, because room isn't the name of a particular location_

  * "Last night's game saw **Liverpool** lose to Manchester United" _Wrong, because in this case Liverpool refers to a sports team_

  * "Liverpool played well at **Old Trafford**" _Correct, OT is a location_

  * "The award went to **Chelsea** Clinton" _Wrong, because here Chelsea refers to a person, not a location_

  * "Just taken my son to **KFC** for dinner #greasybutgood" Right. In this case, KFC is used as a location.

If you can visit it, and the word's used as a location, then mark it like one.


## 3. Corporation

Names of corporations (e.g. **Google**). Don't mark locations that don't have their own name. Include punctuation in the middle of names.

#### Examples:

  * "Stock in **Tesla** is soaring" _Correct - Tesla is a corporation name_

  * "Just taken my son to **KFC** for dinner #greasybutgood" _Wrong. In this case, KFC is used as a location._

  * "I have a **Samsung** Galaxy phone" _This one's out, because it doesn't refer to the corp, just the product._

If you can buy shares in it, or it has employees, and the word's used as a corporation, then mark it like one.


## 4. Product

Name of products (e.g. **iPhone**). Don't mark products that don't have their own name. Include punctuation in the middle of names.

There may be no products mentioned by name in the sentence at all - that's OK. Fictional products can be included, as long as they're referred to by name (e.g. "Everlasting Gobstopper"). It's got to be something you can touch, and it's got to be the official name.

#### Examples:

  * "There was a recall of the **Lexus** **i700**" _correct - Lexus i700 is a product name_

  * "The **apple** is rotten" _wrong, because apple isn't the name of a particular product_

  * "We love our **Kindles**, and never travel without **them**" _Only mark the product names, not words that just refer to them_

  * "I need to go to Crosshill **iPhone** Repair shop" _wrong, because here iPhone is part of another name (for a location)_

  * "Today's plane will be an **Airbus A320**_Correct - the A320 is a product made by Airbus_

  * "I forgot my **phone**" _That's out - this is just the generic term, not the specific product

If you can touch it, you can buy it, and it's the technical or manufacturer name for it, then mark it. 


## 5. Creative work

Names of creative works (e.g. **Bohemian Rhapsody**). Include punctuation in the middle of names. The work should be created by a human, and referred to by its specific name.

#### Examples:

  * "Can't wait til the next **Guardians of the Galaxy** film" _correct - that's a movie_
  * "That was a great **song**!" _wrong, because "song" isn't the name of a particular work_
  * "Man, I love **idubbbz tv**" _correct; this refers to a YouTube channel_
  * "SpaceX names its barges from the **Culture** series" _That's fine - this is a book series.

If it's the specific name of a creative work, for example a movie, song or book, then mark it.


## 6. Group

Names of groups (e.g. **Nirvana**, **San Diego Padres**). Don't mark groups that don't have a specific, unique name, or companies.

There may be no groups mentioned by name in the sentence at all - that's OK. Fictional groups can be included, as long as they're referred to by name.

#### Examples:

  * "**Manchester** played very well on the field tonight" _Correct; Manchester here refers to a sports team_

  * "Closing 8% up on team **Starbucks** today" _This is out; Starbucks here is a company_

  * "Great set from **The Darkness** at Wembley" _This is in - referring to the rock bank_

If it's a special name that refers to a unique, specific group, then mark it. 

# Output format
Produce a JSON that conforms to the following schema:
```typescript
type Response = {
    entities: {
        name: string; // entity verbatim from the provided sentence
        type: "person" | "location" | "group" | "corporation" | "product" | "creative-work";
    }[];
};
```

The JSON MUST conform to the `Response` type. DO NOT produce _anything_ other than the JSON. DO NOT UNDER ANY CIRCUMSTANCES wrap the JSON in markdown or any other format."""


def backoff_handler(details):
    print(
        f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries",
        file=sys.stderr,
    )


@backoff.on_exception(
    backoff.expo,
    RateLimitError,
    on_backoff=backoff_handler,
)
def extract_entities(sentence, model):
    message = client.chat.completions.create(
        model=model,
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"<sentence>\n{sentence}\n</sentence>",
            },
        ],
    )

    try:
        response_text = message.choices[0].message.content or ""
        if "```json" in response_text:
            response_text = response_text.split("```json", 1)[1]
            response_text = response_text.split("```", 1)[0]
        entities = json.loads(response_text)
        entities = entities["entities"]
    except Exception as e:
        print(f"response: {message.choices[0].message.content}", file=sys.stderr)
        print("Failed to parse response", file=sys.stderr)
        entities = []

    return {
        "sentence": sentence,
        "predicted_entities": entities,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8b-chat-hf")
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()

    processed_sentences = set()
    if os.path.exists(args.out_file):
        with open(args.out_file, "r") as f:
            for line in f:
                processed_sentences.add(json.loads(line)["sentence"])

    sentences = []
    for line in sys.stdin:
        sentence = json.loads(line)["sentence"]
        if sentence not in processed_sentences:
            sentences.append(sentence)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [
            executor.submit(extract_entities, sentence, args.model)
            for sentence in sentences
        ]

        for future in tqdm.tqdm(as_completed(futures), total=len(sentences)):
            with open(args.out_file, "a+") as f:
                f.write(json.dumps(future.result()) + "\n")


if __name__ == "__main__":
    main()
