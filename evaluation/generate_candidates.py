#Load UL2 Model
import sys
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5ForConditionalGeneration, AutoTokenizer, AutoModel, BertTokenizer, AutoConfig
import numpy as np
import gleu_mine
import torch
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]= '1' 
#print(len(sys.argv))
#print('0:', sys.argv[0]) 
#print('1:', sys.argv[1])
#print('2:', sys.argv[2])
#print('3:', sys.argv[3])
#print('4:', sys.argv[4])
#print(sys.argv[2],sys.argv[3])

if len(sys.argv) != 6:
        print("Usage: python generate_candidates.py model_dir, model_type, save_dir, benchmarks_dir, save_extension")
        sys.exit(1)

model_dir = sys.argv[1]
model_type = sys.argv[2]
save_dir = sys.argv[3]
benchmark1 = pd.read_csv(sys.argv[4]+'annotated_sentences.csv')
#not actually used so far
benchmark2 = pd.read_csv(sys.argv[4]+'error_type_benchmark_v2.csv')
extension= sys.argv[5]


try:
     benchmark1['source']
except:
     print('benchmark must have "source" column')
     sys.exit(1)
eval_sentences1 = benchmark1['source']
eval_sentences2 = benchmark2['source']

if model_type == 'ul2':
    #model_name = "ul2_finetuning_v1/checkpoint-17500"
    #model_dir = 'E:/Floris/Models/ul2_large_42k_ns/checkpoint-15000'

    model_checkpoint = "yhavinga/ul2-large-dutch"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    max_input_length = 1024
    batch_size=8
    #ul2 prefix
    prefix = "[NLU]"

    eval_sentences1 = [prefix+sentence for sentence in eval_sentences1]
    eval_sentences2 = [prefix+sentence for sentence in eval_sentences2]

if model_type == 'mbart':
    #Load Mbart model
    #model_dir = 'E:/Floris/Models/Mbart_large_42k_ns_4th_attempt/checkpoint-5500'
    model_checkpoint = "facebook/mbart-large-50"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,src_lang="nl_XX", tgt_lang="nl_NL")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    eval_sentences1 = [sentence for sentence in eval_sentences1]
    eval_sentences2 = [sentence for sentence in eval_sentences2]
    batch_size=4

if model_type != 'mbart' and model_type != 'ul2':
      print('Either ul2 or mbart model type')
      sys.exit(1)

def predict_correction(input_texts, batch_size=batch_size, max_length=256):
    decoded_outputs = []

    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i + batch_size]
        #print(len(batch_texts))
        
        # Encode the inputs using the tokenizer and pad them to the same length
        encoded_inputs = tokenizer(batch_texts, padding=True, max_length=max_length, truncation=True, return_tensors="pt")

        # Get the input IDs and attention masks
        input_ids = encoded_inputs["input_ids"].to(device)  # Move to GPU if available
        attention_mask = encoded_inputs["attention_mask"].to(device)  # Move to GPU if available

        # Generate the output
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

        # Decode the output
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

        decoded_outputs.extend(decoded_output)
        torch.cuda.empty_cache()
    input_ids = None
    attention_mask = None
    output= None
    return decoded_outputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the GPU device
model.to(device)

config = AutoConfig.from_pretrained(model_checkpoint)

try: 
    max_length = config.max_position_embeddings
except:
    max_length = config.n_positions

lengths = []
for sentence in eval_sentences1: #_prefixed
    lengths.append(len(sentence))
longest_string = max(lengths)
if longest_string < max_length:
    max_length = longest_string
print("Maximum input sequence length:", max_length)

#Generate candidates

candidates1 = predict_correction(eval_sentences1, max_length=max_length)
benchmark1['candidate_'+extension] = candidates1
print('saving to: ', save_dir+'annotated_sentences_'+extension+'.csv')
benchmark1.to_csv(save_dir+'annotated_sentences_'+extension+'.csv', index=False)
print('first done')
try: 
    max_length = config.max_position_embeddings
except:
    max_length = config.n_positions

lengths = []
for sentence in eval_sentences2: #_prefixed
    lengths.append(len(sentence))
longest_string = max(lengths)
if longest_string < max_length:
    max_length = longest_string
print("Maximum input sequence length:", max_length)

candidates2 = predict_correction(eval_sentences2, max_length=max_length)
benchmark2['candidate_'+extension] = candidates2
benchmark2.to_csv(save_dir+'error_type_benchmark_v2_'+extension+'.csv', index=False)
