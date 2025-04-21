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

answers = merged_datasets["train"][7777:7781]["answers"]

# We'll save the start/end token positions here and check
start_positions = []
end_positions = []

# Iterate over the offset mappings, which is a list of lists.
# It looked like this: offset_mappings = [[(0, 0), (0, 4), (5, 7), (8, 11), (12, 16), ... ]]
# There is one list entry for each of the windows, and the inner lists 
# Each iteration of this main loop works on a single window and turns it into a labeled example
for i, offset in enumerate(inputs["offset_mapping"]):
    print(f"\nDEBUG: Working on offset_mapping #{i}..")
    print(f"DEBUG: offset_mapping={offset}")
    
    # Overflow_to_sample_mapping was the list that tells us which question the
    # window came from, it looked like: [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    print(f"DEBUG: sample_idx={sample_idx}")
    
    # The answer comes from indexing into the 4 example answers
    # according to the current sample (window) we're working on.
    answer = answers[sample_idx]
    print(f"DEBUG: answer={answer}")
    
    # We recall that the answer is a dictionary that contains
    # the text of the answer, as well as where the answer begins.
    # NOTE: answer_start is given as CHARACTER counts relative to the CONTEXT start!
    # So if it was 0, for example, the answer begins with the first
    # character of the context.
    # Answer dictionary looks like this: {'text': ['sire'], 'answer_start': [247]}
    answer_start = answer["answer_start"][0]
    print(f"DEBUG: answer_start={answer_start}")
    answer_end = answer_start + len(answer["text"][0])
    print(f"DEBUG: answer_end={answer_end}")
    
    # Sequence_ids was the mask that tells us what part of the input was
    # the question and what was the context, delineated by special mark tokens
    # e.g. 
    # [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1, 1, 
    # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    # ...
    # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]
    sequence_ids = inputs.sequence_ids(i)
    print(f"DEBUG: sequence_ids={sequence_ids}")
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx    # Found it!  sequence_ids[idx] was 1
    print(f"DEBUG: context_start={context_start}")
    
    # Similarly, scan along again until the mask is *not* 1, which means 
    # we found the end of the context
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1  # Found it!  Decrement to not include 'None' token
    print(f"DEBUG: context_end={context_end}")
    print(f"DEBUG: offset[context_start][0]={offset[context_start][0]}")
    print(f"DEBUG: offset[context_end][1]={offset[context_end][1]}")    
    
    if answer_start < offset[context_start][0] or offset[context_end][1] < answer_end:
        # Straddles or completely outside of context.  Labels are (0, 0) for this window.
        start_positions.append(0)
        end_positions.append(0)
        print("DEBUG: answer is not within the context boundary! Assigning (0, 0) labels.")
    else:
        # Otherwise, the answer has to be fully inside the context!
        print("DEBUG: answer IS completely within the context boundary!")
        idx = context_start
        # Move idx pointer to the right as long as its still within the context,
        # and stop when we find the token that marks the start of the answer.
        while idx <= context_end and offset[idx][0] <= answer_start:
            idx += 1
        start_positions.append(idx - 1)
        print(f"DEBUG: label start_position={(idx - 1)}")

        # Now move our idx pointer to the end and move it backwards, staying inside context
        # window, and stop when it points to the token that ends the answer.
        idx = context_end
        while idx >= context_start and offset[idx][1] >= answer_end:
            idx -= 1
        end_positions.append(idx + 1)
        print(f"DEBUG: label end_position={(idx + 1)}")
print("\n\nstart_positions, end_positions:")
print(start_positions, end_positions)

feature = 0  # Pick the first feature to check

# Get the sample that corresponded to this feature
sample_idx = inputs["overflow_to_sample_mapping"][feature]

# Get the answer dictionary for this sample and index into the answer text
correct_answer = answers[sample_idx]['text'][0]

# Now get the positions from our labels, from the input_ids
start_pos = start_positions[feature]
end_pos = end_positions[feature]
our_label = tokenizer.decode(inputs["input_ids"][feature][start_pos : end_pos+1])

print(f"Correct answer: {correct_answer}, Our label: {our_label}")

feature = 2  # Pick a feature that shouldn't be found inside the context

# Get the sample that corresponded to this feature
sample_idx = inputs["overflow_to_sample_mapping"][feature]

# Get the answer dictionary for this sample and index into the answer text
correct_answer = answers[sample_idx]['text'][0]

# decode the whole feature
decoded_example = tokenizer.decode(inputs["input_ids"][feature])

print(f"Correct answer: {correct_answer}, Decoded example: {decoded_example}")

MAX_LENGTH = 384
STRIDE = 128

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    
    inputs = tokenizer(
        questions,                 # Old: examples["question"],
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",  # Only truncate/window the context, not question!
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",      # Added a padding strategy to make all batches same
    )
    
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    for i, offset in enumerate(inputs["offset_mapping"]):
        sample_idx = inputs["overflow_to_sample_mapping"][i]
        answer = answers[sample_idx]
        answer_start = answer["answer_start"][0]
        answer_end = answer_start + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx    # Found it!  sequence_ids[idx] was 1

        # Similarly, scan along again until the mask is *not* 1, which means 
        # we found the end of the context
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1 
        if answer_start < offset[context_start][0] or offset[context_end][1] < answer_end:
            # Straddles or completely outside of context.  Labels are (0, 0) for this window.
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            # Move idx pointer to the right as long as its still within the context,
            # and stop when we find the token that marks the start of the answer.
            while idx <= context_end and offset[idx][0] <= answer_start:
                idx += 1
            start_positions.append(idx - 1)
            idx = context_end
            while idx >= context_start and offset[idx][1] >= answer_end:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

train_dataset = merged_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=merged_datasets["train"].column_names,
)

from transformers import TFAutoModelForQuestionAnswering

# Load the model with a head for QA using our checkpoint
model = TFAutoModelForQuestionAnswering.from_pretrained(checkpoint)

from transformers import set_seed

# Set the seed for Hugging Face Transformers
set_seed(7)

# Set the seed for TensorFlow
tf.random.set_seed(7)

# Set the seed for NumPy (used by TensorFlow)
np.random.seed(7)

# Set the model seed directly
model.config.seed = 7

BATCH_SIZE = 16

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors='tf')

# Convert the datasets to Tensorflow Datasets
# Also, shuffle and batch
tf_train_dataset = model.prepare_tf_dataset(
    train_dataset,
    collate_fn=data_collator,
    shuffle=True,
    batch_size=BATCH_SIZE,
)

NUM_EPOCHS = 5

num_train_steps = len(tf_train_dataset) * NUM_EPOCHS
print(f"Num training steps: {num_train_steps}")

class PrintLearningRateCB(tf.keras.callbacks.Callback):
    def __init__(self, print_step=1000):
        super(PrintLearningRateCB, self).__init__()
        self.print_step = print_step

    # Remember that every new training step is a new batch, same thing
    def on_batch_begin(self, batch, logs=None):
        if batch % self.print_step == 0:
            # Retrieve the current learning rate from the model's optimizer
            lr = self.model.optimizer.lr
            # If using a learning rate schedule, the learning rate might be a callable
            if callable(lr):
                lr = lr(self.model.optimizer.iterations)
            # TensorFlow 2.x returns a tensor, so evaluate it to get the value
            lr_value = tf.keras.backend.get_value(lr)
            print(f"\nStep: {batch}, Learning Rate: {lr_value:.6f}")

MY_MODEL_DIR = "distilbert-base-cased-squad_v1"

# We need to log into Hugging Face to be able to push the model to the Hub
# and also to load it (since a private repo)
from huggingface_hub import notebook_login

# Play a beep when the login page comes to get my attention
from IPython.display import HTML

def play_sound():
    audio_url = "https://www.soundjay.com/misc/sounds/censor-beep-3.mp3"  # Example URL
    display(HTML(f"<audio src='{audio_url}' autoplay></audio>"))

if DO_TRAINING:
    
    # Beep!
    play_sound()

    # Code will pause here where we can login with our HF credentials
    notebook_login()
    
    from transformers.keras_callbacks import PushToHubCallback
   
    # By default, this uploads to your username and named after the output_dir
    # (though this can all be overridden), so for me this will upload to:
    # "joelwigton/distilbert-base-cased-squad_v1"
    PushToHubCB = PushToHubCallback(output_dir=MY_MODEL_DIR, tokenizer=tokenizer)   

INITIAL_LR = 5e-5
END_LR = 1e-5
power = 2

# Create the polynomial decay learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=INITIAL_LR,
    end_learning_rate=END_LR,
    decay_steps=num_train_steps,    
    power=power,  # This controls how the learning rate decays over time
    cycle=False   # If True, it causes a periodic restart of the decay
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
import matplotlib.pyplot as plt

# We can plot it to see what it's going to do
plt.plot(lr_schedule(tf.range(num_train_steps, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Compile the model
# Will use the default loss already specified by the model!
model.compile(optimizer=optimizer)
# View the model
model.summary()

if DO_TRAINING:
    # We don't need a learning rate callback, it's already built into optimizer
    # But we did add another custom one to show the learning rate at each epoch
    # And another to push to the HF Hub after progress.
    model.fit(tf_train_dataset,
              epochs=NUM_EPOCHS,
              callbacks=[PrintLearningRateCB(print_step=1000),
                         PushToHubCB],
            )
    
    # Beep!
    play_sound()
    print("Training has finished.")
    
# If we skipped training, let's re-load the model from our saved Hub checkpoint instead
else:
    model = TFAutoModelForQuestionAnswering.from_pretrained(f"joelwigton/{MY_MODEL_DIR}")

context = """
    In breeding circles, a male canine is referred to as a dog, while a female is called 
    a bitch (Middle English bicche, from Old English bicce, ultimately from Old Norse 
    bikkja). A group of offspring is a litter. The father of a litter is called the sire, 
    and the mother is called the dam. Offspring are, in general, called pups or puppies, 
    from French poupée, until they are about a year old. The process of birth is whelping, 
    from the Old English word hwelp."""

question = "What is the male who is mother of the pups called?"

def get_answer_simple(question, context):
    # For now, we'll truncate the inputs so that the question plus context fits.
    # But remember, we could be missing information!  We'll fix this later.    
    inputs = tokenizer([question],  # a list
                       [context],   # a list
                       return_tensors='tf', 
                       truncation="only_second", # Only truncate context, not question
                       padding='max_length',
                       # We set this earlier when windowing, it's not the same as what
                       max_length=MAX_LENGTH,  # the model can actually handle
                      )
    # print("inputs type: ", type(inputs))
    
    outputs = model(inputs)

    start_token = int(tf.argmax(outputs.start_logits, axis=1))
    end_token = int(tf.argmax(outputs.end_logits, axis=1))
    
    # print(start_token, end_token)
    if start_token > end_token:
        print("WARNING: start_token came after end_token!")

    answer = inputs["input_ids"][0][start_token : end_token+1]
    return tokenizer.decode(answer)

print(get_answer_simple(question, context))

def get_answer_improved(question, context):
    tokenized_inputs = tokenizer([question],             # a list
                               [context],                # a list
                               return_tensors="tf",
                               truncation="only_second", # Only truncate context
                               padding='max_length',     # This cuts off longer contexts
                               # We set this earlier when windowing, it's not the same as what
                               max_length=MAX_LENGTH,    # the model can actually handle
                      )

    # Run the tokenized inputs through the model
    outputs = model(tokenized_inputs)
    start_logits = outputs.start_logits[0].numpy()  # lists
    end_logits = outputs.end_logits[0].numpy()

    # Sort the list of logits, getting the indices
    # Defaults to sorted ascending, so reverse it, then take best 20 (like BERT paper)
    start_positions = list(np.argsort(start_logits))[::-1][:20]  # Best 20
    end_positions = list(np.argsort(end_logits))[::-1][:20]      # Best 20

    # Now we'll build up a list of potentially good answers.
    # This list will consider not just the best start token like before,
    # but the best answer that maximizes the start PLUS the end logits
    # (since logarithmic, we add them).
    # Thus there could be up to 20 * 20 = 400 answers generated here.
    valid_answers = []
    for start_pos in start_positions:
        for end_pos in end_positions:
            # Only add solutions where the start comes before (or same as) end position
            if start_pos <= end_pos:
                # Add a dictionary entry with the score and resulting text
                valid_answers.append(
                    {
                        "score": start_logits[start_pos] + end_logits[end_pos],
                        "text": tokenizer.decode(tokenized_inputs['input_ids'][0][start_pos:end_pos+1])
                    }
                )
        valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)

    # Return just the best one
    return valid_answers[0]['text'] 

print(get_answer_improved(question, context))

