DO_TRAINING = False

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

from datasets import load_dataset, DatasetDict, concatenate_datasets

# Load the SQUAD dataset
original_datasets = load_dataset("squad")

# Combine 'train' and 'validation' datasets
combined_train_dataset = concatenate_datasets([
    original_datasets['train'],
    original_datasets['validation']
])

# Create a new DatasetDict with only the 'train' dataset
merged_datasets = DatasetDict({
    'train': combined_train_dataset
})

del combined_train_dataset

merged_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)

from transformers import AutoTokenizer

checkpoint = "distilbert/distilbert-base-cased"  # Distilled version of BERT. 65M parameters
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

context = merged_datasets["train"][7777]["context"]
question = merged_datasets["train"][7777]["question"]

inputs = tokenizer(question, context)
print(tokenizer.decode(inputs["input_ids"]))
print(type(inputs))

inputs = tokenizer(
    question,
    context,
    max_length=75,
    truncation="only_second",
    stride=40,
    return_overflowing_tokens=True,
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))

inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

# Look at the keys that are returned with inputs variable
print(inputs.keys())

for k, v in inputs.items():
    print(f"{k}: {v}\n")

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(tokens)

inputs = tokenizer(
    merged_datasets["train"][7777:7781]["question"],
    merged_datasets["train"][7777:7781]["context"],
    max_length=100,
    truncation="only_second",  # Only truncates/windows the context, not the question!
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")

# inputs is a BatchEncoding class type
print(type(inputs))

# .sequence_ids() is a method on the BatchEncoding class type
for i, _ in enumerate(inputs):
    print(inputs.sequence_ids(i))
    
print("\nJust the first one:")
print(inputs.sequence_ids(0))
