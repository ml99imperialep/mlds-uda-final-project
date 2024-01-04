# Unstructured Data Analysis-Final Project: Harry Potter and Friends (?)

_Imperial College London - MSc Machine Learning and Data Science Year 2._

We consider the seven Harry Potter novels written by J.K Rowling and apply text analysis. 

Knowledge graphs and social networks/character networks are created on the basis of this Harry Potter text corpus.

## Dataset(s)

Seven Harry Potter novels downloaded from [jacobs repository](https://github.com/ErikaJacobs/Harry-Potter-Text-Mining/tree/master/Book%20Text), which is a public repository as well.

The second dataset [characters.csv](./data/characters.csv) contains the name of the characters and their variation downloaded from [ohumu repository](https://github.com/ohumu/HarryPotterNetwork/blob/main/characters.csv)

## Github Repository

The code of the project is fully stored in the public repository: 

2 Folders:

- [data](./data/): contains the datasets [characters.csv](./data/characters.csv) and [HPBook1.txt](./data/HPBook1.txt), [HPBook2.txt](./data/HPBook2.txt), [HPBook3.txt](./data/HPBook3.txt), [HPBook4.txt](./data/HPBook4.txt), [HPBook5.txt](./data/HPBook5.txt), [HPBook6.txt](./data/HPBook6.txt), [HPBook7.txt](./data/HPBook7.txt)  (6.9 MB)
- [images](./images/): divided into 3 subfolders:  [eda](./images/eda/),  [knowledge_graph](./images/knowledge_graph/),  [social_network](./images/social_network/) contains all images which are produced in the jupyter notebooks (5.7 MB for the image folder)

2 Jupyter notebooks (Python 3.10.13):

- [result.ipynb](./result.ipynb): produces results of [eda](./images/eda/) and [knowledge_graph](./images/knowledge_graph/)
- [social_network.ipynb](./social_network.ipynb): produces results of [social_network](./images/social_network/)

It must be noted that the two notebooks are independent from each other.

3 Python files (Python 3.10.13):

- [eda_utils.py](./eda_utils.py) ( functions for [result.ipynb](./result.ipynb))
- [kg_utils.py](./kg_utils.py) ( functions for [result.ipynb](./result.ipynb))
- [sn_utils.py](./sn_utils.py) (functions for [social_network.ipynb](./social_network.ipynb))


1 Textfile

- [requirements.txt](./requirements.txt): includes all versions of python libraries (see Running the code section)


## Running the code
It is recommended to use a (python) virtual environment to manage the dependencies called ['pyenv'](https://github.com/pyenv/pyenv) was used to test this project on a desktop computer and can be installed using the [pyenv installer](https://github.com/pyenv/pyenv-installer). A virtualenvironment with Python version 3.10.13 was used for this project.

The libraries needed to run the notebooks are stored in the following textfile ['requirements.txt'](./requirements.txt). (it is the same for both notebooks)

To install the dependencies:

```bash
pip install -r requirements.txt
```
Additionally, the 'en_core_web_sm' pipeline from 'spacy' must also be downloaded using:

```bash
python -m spacy download en_core_web_sm
```

and download the following via the 'nltk' package is as well helpful.

```bash
nltk.download('punkt')
nltk.download('stopwords')
```

Be careful to install it in the correct folder.

## Additional information
This notebook is developed on Github codespaces and tested on a desktop computer:
- Github codespace: 4-core, 16GB RAM, 32GB, European Western Area
- Desktop Computer: 8 GB 1867 MHz DDR3; 3,1 GHz Quad-Core Intel Core i5

