#################################
# This file contains all functions which are used in the Knowledge Graph part of eda_kg.ipynb
#################################


# spacy library
import spacy

from spacy.matcher import Matcher 



## inspired by : https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk

def get_entities(sent, nlp):
    """
    Derive entities for knowledge graphs
    """
    # define initial variables
    x, y, prev_dep, prev_txt, prefix, modifier = "", "", "", "", "", ""


    for tok in nlp(sent):
        if tok.dep_ != "punct": # automatically remove punctuation (if available)
            ## prefix: previous compound and compound
            if tok.dep_ == "compound":
                prefix = prev_txt +" "+ tok.text if prev_dep == "compound" else tok.text
            
            ## modifier to the previous compound and mod
            if tok.dep_.endswith("mod") == True:
                modifier = prev_txt +" "+ tok.text if prev_dep == "compound" else tok.text
            ## subject: modifier, prefix and subject
            if tok.dep_.find("subj") == True:
                x = modifier +" "+ prefix + " "+ tok.text
                prefix, modifier, prev_dep, prev_txt = "", "", "", ""
            ## object: modifier, prefix and object
            if tok.dep_.find("obj") == True:
                y = modifier +" "+ prefix +" "+ tok.text
            
            # get dependency ad text
            prev_dep, prev_txt = tok.dep_, tok.text
    # get entities x,y
    x = " ".join([i for i in x.split()])
    y = " ".join([i for i in y.split()])
    return (x.strip(), y.strip())






##  copied from: https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk

def get_relation(sent, nlp):
    """
    Derive relations for knowledge graphs
    """
    doc = nlp(sent)

    # Matcher class object 
    matcher = Matcher(nlp.vocab)

    #define the pattern 
    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"}, 
            {'DEP': 'acomp', 'OP': "?"},
            {'POS':'ADJ','OP':"?"}] 
    

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return(span.text)

    
