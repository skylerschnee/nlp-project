## Github README Natural Language Processing/ Web-Scraping 

Josh Burch, Mack McGlenn, Skyler Schnee

Codeup: O'neil Cohort

### Project Overview

The goal will be to build a model that can predict the main programming language of a repository, given the text of the README file.

### Objectives

- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter notebook final report.

- Create modules (acquire.py, prepare.py) that make our process repeateable and our report (notebook) easier to read and follow.

### Deliverables

- A GitHub repository containing our work
  - a well-documented jupyter notebook that contains our analysis
- a README file that contains a description of our project, as well as instructions on how to run it
- 2-5 google slides suitable for a general audience:
        - our slides should summarize our findings in exploration and the results of our modeling
        - include well-labeled visualizations in our slides
        - our google slide deck should be linked in the README of our repository

### Project Planning
Outline:

- Acquire & Prepare Data
- Exploration
- Modeling
- Conclusion

### 1. Acquire & Prepare
Acquire Actions

- Data acquired from web-scraping 30 pages of README.md files from popular github repositories on 15 May 2023.
    - Filters: Java, Javascript, Python, most starred

Prepare Actions:
    - Convert text to all lower case for normalcy.
    - Remove any accented characters, non-ASCII characters.
    - Remove special characters.
    - Stem or lemmatize the words.
    - Remove stopwords.
    - Store the clean text and the original text for use in future notebooks.    
   
  - Remove "giveaways"
      - Remove 'python', 'py', 'java', 'javascript', 'js', 'script'
      - Removed because these words showed to be dead giveaways of the language we are predicting.

   - Calculate README word counts
      - Define word counts for analysis, exploration, and determining a proper statistical test.  
   
Functions called from:
   - acquire.py
   - prepare.py
   - functions.py
    
### 2. Exploration/ Correlation Analysis

Here are the steps we took during the exploration phase:
  - performed statistical test
  - created bigrams
  - create word clouds
  - visualize top 20 words with frequencies


### 3. Modeling

Here are the steps we took during the modeling phase:
  - define the basline (33% accuracy)
  - created 4 ML models
  - judged model performace on train and validate subset accuracy scores
  - ran our best model on test subset

In conclusion, we are successful in our efforts to out-perform baseline accuracy predicting the programing language by analyzing the README of a github repository between python, java, and javascript. Our K-Nearest Neighbor Classifier was our best performing model with solid accuracy scores on train and validate sets, with the most most relatively optimal score delta.


### Conclusion & Reccomendations
- To understand the limitations we purposely placed on ourselves in the effort of using alternate methods to increase our model performance, re-review the "giveaway" words we removed from the dataset. Words like "python" and "js" among others proved to be giveaways of the programming language used.
- To make futher advancements with this project, we would like to scrape more github README content and re-run through this notebook to further solidify or improve our resluts. We would also like to experiment with additional models and parameters.

### Steps to Reproduce
- create env.py file 
  - needs username varible with github username 
  - needs github token for api use
- clone repo and run the notebook
