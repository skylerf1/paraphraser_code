import re
import random


def remove_colon_words(text):
    result = []
    words = text.split()
    for word in words:
        #skip P:, O: etc.
        if ':' in word and len(word) <= 3:
            continue
        #remove words with a : within it as happens often with o:meneer
        if re.search(r'\b:\b', word):
            if not word.split(':')[0].isdigit():
                #remove text before the ':'
                word = word.split(':')[1]
        result.append(word)
    return ' '.join(result)

def preprocess(text):
    text2 = remove_colon_words(text)
    return abbr_substitutor(text2)
'''
This function applies all known substitutions to the text
'''
def abbr_substitutor(text):
    abbr_list = {
        "dhr": "meneer",
        "dhr.": "meneer",
        "hr": "meneer",
        "deheer": "meneer",
        "heer": "meneer",
        "mw": "mevrouw",
        "mw.": "mevrouw",
        "mevr.": "mevrouw",
        "mevr": "mevrouw",
        "mvr.": "mevrouw",
        "mvr": "mevrouw",
        "mv": "mevrouw",
        "mv.": "mevrouw",
        "agv": "als gevolg van",
        "ha": "huisarts",
        "h.a": "huisarts",
        "pgb": "persoons gebonden budget",
        "ivb": "in verband met",
        "tgv": "ten gevolge van",
        "evt": "eventueel",
        "ivm": "in verband met",
        "adl": "algemene dagelijkse levensverrichtingen",
        "adl.": "algemene dagelijkse levensverrichtingen",
        'def': 'defecatie',
        "a:": "",
        "e:": "",
        "p:": "",
        "o:": "",
        "s:": "",
        'b:' : '',
        "zkh": "ziekenhuis",
        "cva": "beroerte",
        "zn": "zo nodig",
        "wmo": "wet maatschappelijke ondersteuning",
        "dgs": "daags",
        "mi": "volgens mij",
        "assha": "assistent huisarts",
        "ass": "assistent",
        "li-arm": "linker arm",
        "re-arm": "rechter arm",
        "vpk": "verpleegkundige",
        "vlgs": "volgens",
        "re": "rechter",
        "li": "linker",
        "dr": "dokter",
        "tv": "televisie",
        'b/' : '',
        "p/": '',
        'o/': '',
        's/': '',
        'a/': '',
        'e/': '',
        'b' : '',
        'p': '',
        'o': '',
        's': '',
        'a': '',
        'e': '',
        's.': '',
        'p.': '',
        'o.': '',
        'a.': '',
        'b.': '',
        'e.': '',
        'ao/': '',
        'ao': '',
        'ao:': '',
        'ao.': '',
        'o/p:' : '',
        't:' : '',
        'o=' : '',
        'observatie:': '',
        'participatie:': '',
        'wijkvpk': 'wijkverpleegkundige',
        'advies:': '',
        '/n' : '', 
        'lichamelijke welbevinden:' : '', 
        'zlo' : 'zorgleveringsovereenkomst',
        'clad' : 'centrale landsaccountantsdienst',
        'pcm': 'paracetamol',
        'tbl': 'tablet',
        'hb': 'hemoglobine',
        'rapp.' :'rapportage',
        'rap.' :'rapportage',
        'def.' :'defecatie',
        'nd.' : 'nacht dienst.',
        'nd' : 'nacht dienst',
        'inc' : 'incontinentie',
        'inc.' : 'incontinentie',
        'dd' : 'keer per dag',
        'uro' : 'uroloog',
        'vg' : 'voorgeschiedenis',
        'obs/rapp' : 'oberveer/rapporteer'
        }
    #if re.match(r"^\d+dd$", token)
    #contactpersoon 1cp/2cp
    #ND Vervolg O/ Mw ging totaal 5 maal op de postoel. Mw urineert veel.   
    words = text.split()
    result = []
    for word in words:
        #dd -> keer per dag
        #if re.match(r"^\d+dd$", word.lower()):
        #    num = token[:-2]
        #    word = num + "e contactpersoon"
        #contactpersoon
        #if re.match(r"\d+cp", word.lower()):
        #    num = token[0]
        #    word = num + " keer per dag"

        original_word = word
        if word.lower() in abbr_list:
            word = abbr_list[word.lower()]
            #print(original_word, original_word[0])
            if original_word and original_word[0].isupper():
                word = word.capitalize()
        result.append(word)
    return ' '.join(result)

def simulate_sentence_continuation(text, p):
    sentences = re.split(r'([.!?])', text)
    
    for i in range(1, len(sentences), 2):
        if random.random() < p:
            if i+1 != len(sentences)-1:
                #remove punctuation
                sentences[i] = sentences[i].lstrip('!?.')
                #remove capitalization of first word
                sentences[i+1] = re.sub(r'^([^a-zA-Z]*)([a-zA-Z])', lambda match: match.group(1) + match.group(2).lower(), sentences[i+1])
                if sentences[i+1][0] != ' ':
                    #add space if there isnt one at the start of the next sentence
                    sentences[i+1] = ' '+ sentences[i+1]

    simulated_text = ''.join(sentences)
    return simulated_text

def remove_commas(text, p):
    words = text.split()
    modified_words = []
    
    for word in words:
        if ',' in word:
            if random.random() < p:
                word = word.replace(',', '')
        modified_words.append(word)
    
    modified_text = ' '.join(modified_words)
    return modified_text

def make_sentence_df(df, text_column_name):
    df['sentences'] = df[text_column_name].apply(split_into_sentences)
    df_exploded = df.explode('sentences') 

    df_exploded['num_words'] = df_exploded['sentences'].apply(count_words)
    df_exploded = df_exploded[~df_exploded['num_words'].isna()]
    # Remove the problematic entries from df_exploded
    #df_exploded = df_exploded.dropna(subset=['sentences'])

    return df_exploded

from nltk.tokenize import sent_tokenize
def split_into_sentences(text):
    return sent_tokenize(str(text))

# Calculate the number of words using a custom function
def count_words(sentence):
    try:
        return len(sentence.split())
    except AttributeError:
        return None  # Handle problematic entries as NaN

