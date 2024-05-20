## Preprocessing

Dataset from https://noisy-text.github.io/2017/emerging-rare-entities.html

```bash
cat emerging.dev.conll | python ./process_dataset.py >| emerging.dev.jsonl
```

## Extracting entities

(Small subset for testing)

```bash
head -n100 ./emerging.dev.jsonl | COHERE_API_KEY=<key> python run_cohere.py --num_threads 3 --out_file emerging.dev.predicted.command-r-plus.jsonl
```

## Evaluation

```bash
python ./eval.py emerging.dev.jsonl emerging.dev.predicted.command-r-plus.jsonl
```
