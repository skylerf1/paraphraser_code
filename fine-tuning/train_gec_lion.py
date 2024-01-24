import torch
from datasets import load_dataset, load_metric
import pandas as pd 
from datasets import DatasetDict, Dataset
import nltk
nltk.download('punkt')
import string
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5ForConditionalGeneration, AutoTokenizer, AutoModel, BertTokenizer, AutoConfig
import numpy as np
import sys
import subprocess
import os
from gleu_mine import GLEU
import scipy.stats
import random
from gleu_mine import gleu_calculator_mine
import os

os.environ["CUDA_VISIBLE_DEVICES"]= '1'  

#CUDA_VISIBLE_DEVICES=1
# source ~/.bashr
#training data
def main():
    if len(sys.argv) != 4:
            print("Usage: python train_model.py filename, model_type, model_dir")
            sys.exit(1)

    filename = sys.argv[1]
    model_type = sys.argv[2]
    model_dir = sys.argv[3]

    try:
        df = pd.read_csv(filename).sample(frac=1).reset_index(drop=True)
    except:
        print('filename should be a csv file with two columns, source and target.')
        sys.exit(1)

    if model_type == 'mbart':
        model_checkpoint = "facebook/mbart-large-50"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,src_lang="nl_XX", tgt_lang="nl_NL")
    elif model_type == 'ul2':
        model_checkpoint = "yhavinga/ul2-large-dutch"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    else:
        print("model_type must be ul2 or mbart")
        sys.exit(1)

    #possibly use auto-batching to fit to your gpu! The ul2 method can not handle fp16 while mbart can. 
    #As a consequence mbart can be trained with a batch size of 2 and ul2 of 1 (large). Maybe more on turtle...

    #model_name = 'mt5_small_finetuning_v1'
    if model_checkpoint == "facebook/mbart-large-50":
        model_name = 'Mbart_large_100k_' + filename[:-4][-2:]
        batch_size = 16 #32
    
    else:
        model_name = 'Ul2_large_100k_'+ filename[:-4][-2:]
        batch_size = 4
    print('batch_size: ',batch_size)
    model_dir = model_dir + model_name #"E:/Floris/Models/" 

    args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,
        save_strategy="steps", 
        save_steps= 1000,
        learning_rate=2e-5, #0.0011 before rsqrt decay. Unsure how many steps till convergence paper had or 0.001 4,490731195102493e-5
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=12,
        #max_steps=750,
        predict_with_generate=True,
        #fp16 true for mbart. only good with large batches
        fp16=False,
        #warmup_steps = 1000,
        #weight_decay=0.01,
        metric_for_best_model="GLEU+",
        report_to="tensorboard",
        save_total_limit = 2,
        load_best_model_at_end=True,
        dataloader_drop_last=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    def get_gleu_stats(scores) :
        mean = np.mean(scores)
        std = np.std(scores)
        ci = scipy.stats.norm.interval(0.95,loc=mean,scale=std)
        return ['%f'%mean,
                '%f'%std,
                '(%.3f,%.3f)'%(ci[0],ci[1])]
    def prepare_text(sentence):
        tokens = nltk.word_tokenize(sentence)
        return ' '.join(tokens)

    def compute_gleu(predictions, references, sources, n=4, num_iterations=500):
        value = 0
        for i in range(len(predictions)):
            if i == 0:
                print('source ', sources[i], 'predictions ', predictions[i], 'references ',  references[i], 'just a bit at the end')
                print('gleu', gleu_calculator_mine(sources[i], predictions[i], [references[i]], n=n, num_iterations=num_iterations)[0])
            value += float(gleu_calculator_mine(sources[i], predictions[i], [references[i]], n=n, num_iterations=num_iterations)[0])
        print('here', value/len(predictions)) 
        return {'GLEU+' : round(value/len(predictions), 3)}
  
    #ENTER DETAILS OF YOUR EVALUATION DF TO ALIGN WITH SOURCES
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        #print(decoded_labels, flush=True)
        sources = [eval_df[eval_df['target'] == target]['source'].values[0] for target in decoded_labels]
        # Replace -100 in the labels as we can't decode them.
        #print('compute metrics ', 'source ' , sources[0], 'prediction', decoded_preds[0], 'reference ', decoded_labels[0])
        #print(' types ' , type(decoded_preds[0]))
        # Rouge expects a newline after each sentence
        result = compute_gleu(decoded_preds, decoded_labels, sources)
        # Compute ROUGE scores
        
        return result

    # Function that returns an untrained model to be trained
    def model_init():
        return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        #return MBartForConditionalGeneration.from_pretrained(model_checkpoint)

    def load_a_model():
        model_dir = model_dir+model_name

        return AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    trainer = Seq2SeqTrainer(
        model_init(),
        args=args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Start TensorBoard
    #subprocess.run(['tensorboard', '--logdir', model_dir])

    config = AutoConfig.from_pretrained(model_checkpoint)

    try: 
        max_length = config.max_position_embeddings
    except:
        max_length = config.n_positions
    print("Maximum input sequence length:", max_length)

    #add prefix if using ul2 model!
    if model_checkpoint == "yhavinga/ul2-large-dutch":
        prefix = "[NLU]"
    else:
        prefix = ''
    
    max_input_length = max_length
    max_target_length = max_length

    def preprocess_data(examples):
        #texts_cleaned = [clean_text(text) for text in examples["Source"]]
        #inputs = [text for text in examples["Source"]]
        inputs = [prefix + text if text is not None else prefix+ "" for text in examples["Source"]]
        #inputs = [prefix + text for text in examples["Source"]]
        #print("Processing data:", inputs)  # Print the first source text
            # ... rest of the preprocessing code ...

        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        #print("Processed data:", model_inputs["input_ids"][-1])  # Print the first few input IDs
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["Target"], max_length=max_target_length, 
                            truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def split_data(df, split=(0.7,0.2,0.1)):
        # Shuffle the data
        df = df.sample(frac=1, random_state=42)

        # Get the number of rows in the data
        n_rows = df.shape[0]

        # Define the train, validation, and test set sizes
        train_size = int(split[0] * n_rows)
        val_size = int(split[2] * n_rows)

        # Split the data into train, validation, and test sets
        train_set = df.iloc[:train_size]
        eval_set = df.iloc[train_size:train_size + val_size]
        test_set = df.iloc[train_size + val_size:]
        return train_set, eval_set, test_set

    def reformat_data(df, type_of_data):
        '''
        type of data is either train, eval or test
        '''
        #reformat
        df = df.rename(columns={'source': 'Source', 'target': 'Target'})
        df = df.reset_index(drop=True)

        dataset_df = Dataset.from_pandas(df)
        # Create a DatasetDict object
        dataset_dict_df = DatasetDict({type_of_data: dataset_df})
        tokenized_dataset_dict_df = dataset_dict_df.map(preprocess_data, batched=True, batch_size=16)
        return tokenized_dataset_dict_df

    #CHECK FOR CORRECT TRAINING DATA
    train_set, _, _ = split_data(df, (1, 0, 0))
    eval_df = pd.read_csv('E:/Data_exploration/GitHub/paraphraser_code/data/benchmarks/annotated_sentences.csv')
    _, eval_set, _ = split_data(eval_df, (0,0,1))

    eval_datasetdict = reformat_data(eval_set, 'eval')
    train_datasetdict= reformat_data(train_set, 'train')

    trainer.eval_dataset = eval_datasetdict['eval']
    trainer.train_dataset = train_datasetdict['train']

    trainer.train()
if __name__ == "__main__":
    main()
