--- Generate similar data to a dataset using Chat-GP---
	
* Data Preperation
First use the file 'data cleaning.py' to structure the careplan reports. Specific to our dataset.

Check out vocabulary.py to see some statistics about the data. This gives insight in the distribution of sentence lengths, as well as use the vocabulary of the corpus to support the spellchecker. It also creates the nl_with_corpus.txt file which is used by the spellchecker later. We also preprocessed our corpus with a dictionary found in functions.py. Here we got rid of some noise and abbreviations so the labeler would be informed as much as possible. We also used these preprocessed texts later as sources instead of the original text. 


Then use 'labelling.py' to hand label some data entries similar to your data to avoid sensitivity issues if there are any. These are used to prompt GPT with some examples. 

Then with your labelled data you can use 'generate_(in)correct_data.py' or generate_triplets pipeline.py to generate data. If you have non sensitive data you can use generate labels to generate corrections directly. Please make sure to use your own API key for openai which you can request on their website. Furthermore if you use this code for data other than the medical entries you will need to rewrite a prompt to prompt GPT. This may take some experimentation to get the desired results. If you are using the generate_triplets pipeline you will also need the generated keywords from Diego's model in keywords3_v1.csv. Use transform_triplets.py to generate this keyword file.

*spellchecker
Before we can create our synthetic data we need to gather the common errors in the corpus. First we preprocess and count the words in the nos dataset(kaggle) with text_to_counter.py. We remove words which are less common than 3 to get rid of uncommon names and very uncommon words. For the care_plan dataset we ignore all words less common than 170 instances since this means words occur less than 1/2.500.000. This way we should not add typos to the correct word list which we will spell check with. Very common errors will however possibly not be caught by the spellchecker! Then we add this to the nl_with_corpus.txt file to have a word list with occurences which we can use together with symspell to correct the corpus. with symspell_applied.py we apply the spellchecker to the corpus and save all the reverse error patterns so we can replace them in the error introduction step. This results in the spelling_corrected_corpus_nos_suggestions.pickle file which is used to replace words. 

Use nastraka_error_introduction to introduce spelling errors to the generated text based on your wants and needs. 

error introduction v1 includes:
Word level: insertion(random), replacement (common errors from corpus), deletion(random), swapping(random). 
Character level: insertion(random), replacement(random), deletion(random), swapping(random).

error introduction v2:
same as v1 but
+Interpunction level: probabilistically remove periods and following capitalizations. This is applied to a new section of 2 sentence long data points.
+Interpunction level: remove commas
+Word level: replace bigrams(common bigram errors from corpus). 

error introduction v3:
same as v2 but
-Word level: deletion, swapping
+Span level: deletion, swapping

error introduction v4:
Same as v3 but 
-Span level: deletion
+Word level: NER based deletion

error introduction v5:
(mc4_selection for text selection)
Same as v2 (since it was found as best) but 
-Gpt generated dataset
+dutch mc4 dataset (used mc4_selection.ipynb to create dataset)

All errors are based on probabilities. Our probabilities are found within naplava_spell_checker when we introduce it in v1, v2, v3, v4.

* Fine-tuning
Create a venv with requirements_lion.txt It uses python 3.10, linux and cuda 11.8. More details are in linux_requirements.txt

For training we manually annotated 283 sentences with a target sentence to be able to calculate GLEU+ scores between candidates and target during evaluation. We also did this within labelling.py. This is found in annotated_sentences.csv which is used as the evaluation set.

Finetune multiple large models by useing train_4_models.sh. This script uses the train_gec_lion file to train models. It saves the best and last checkpoint based on GLEU+ scores. Adapt the code for your own batch_sizes and paths to work. Models are evaluated on the annotated_sentences.csv file. This file will be in the benchmark folder. During training stick to the column names used or adapt the code.

You can find your training analytics in the tensorboard run files. You can see the GLEU+ scores for the evaluation set.

*Evaluation

Generate candidate sentences using the model of your choice on the annotated sentences. This is done by using the generate_candidates and generate_all_candidates files.
Then use combine_candidates to create a csv file with the source sentences and all candidates. These can then be judged by humans.

Calculate sentence similarity scores and different evaluation metrics for all generated candidates with semantic_similarity_measures.ipynb. It uses the requirements_310.txt in a python 3.10 environment 


* Human evaluation
To evaluate your results by humans we have some scripts to make this easier.
Use the create_survey file to transform a csv file consisting of source, correction1, correction2... correctionN to a qualtrics survey text format which can be imported to qualtrics as a survey.

Use calculate_qualtrics_results to extract the average scores per model from an exported qualtrics survey csv file for a single user and generate visualizations.

Use apply_model.py to apply the model of your choice to your own text data. In the form of a csv file with appropriate column names.

*Clustering

Clustering contains code to cluster sentences based on contextual sentence embeddings. These were not used in the end but could be used for future work. It also uses the requirements_310.txt in a python 3.10 environment 






