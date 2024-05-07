#!/usr/bin/env python
# coding: utf-8

# ### Importing Necessary Libraries

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string 
import re
import nltk
nltk.download(['stopwords','wordnet', 'punkt'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import warnings 
warnings.filterwarnings('ignore')
import subprocess

BOLD = '\033[1m'
RESET = '\033[0m'


# In[33]:


# for other theme, please run: mpl.pyplot.style.available
PLOT_PALETTE = 'tableau-colorblind10'
# for other color map, please run: mpl.pyplot.colormaps()
WORDCLOUD_COLOR_MAP = 'tab10_r'

# set palette color
plt.style.use(PLOT_PALETTE)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Data

# Data Set Link - 

# In[12]:


df = pd.read_csv('Resumes.csv')


# ### Basic EDA

# In[13]:


print(df.head())
print()
print(BOLD + "Information about the data frame:" + RESET)
print()
df.info()


# ##### Dropped Unused columns like ID, Resume_html

# In[15]:


# drop unused columns
df.pop('ID')
df.pop('Resume_html')
df


# ##### Before Cleaning the resume

# In[16]:


df["Resume_str"][1]


# #### Cleaning Resume Text

# In[17]:


stemmer = nltk.stem.porter.PorterStemmer()


# In[19]:


def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove non-english characters, punctuation, and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Tokenize words
    words = word_tokenize(text)
    
    # Remove stop words and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    
    # Additional cleaning steps
    text = ' '.join(words)
    
     # Remove punctuations, mentions, hashtags, RT and cc,  Remove URLs
    text = re.sub('http\S+\s*|RT|cc|#\S+|@\S+|[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    
     # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7f]', ' ', text) 
    
    # Remove extra whitespace
    text = re.sub('\s+', ' ', text)  
    
    return text.strip()


# In[20]:


# preprocessing text
df['Resume'] = df['Resume_str'].apply(lambda w: preprocess(w))

# drop original text column
df.pop('Resume_str')

df


# In[21]:


#After Cleaning the resume text
df["Resume"][1]


# # Advanced EDA & Visualizations

# In[27]:


# create list of all categories
categories = np.sort(df['Category'].unique())
categories


# ##### Distribution of Resume Categories

# In[25]:


# Define a custom color palette
custom_palette = sns.color_palette("pastel")

# Plot the pie chart with the custom color palette
plt.figure(figsize=(8, 8))
df['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=custom_palette)
plt.xticks(rotation=90)
plt.show()


# In[26]:


df['Category'].value_counts()


# In[48]:



# Set the style and background color
sns.set_style("whitegrid", {"axes.facecolor": "#E0ECFF"})  # Light blue background color

# Create a figure and axis objects
plt.figure(figsize=(10, 8))

# Define a standard blue color for the bars
standard_blue = "#4C72B0"

# Plot a vertical bar chart with the standard blue color
sns.countplot(x="Category", data=df, color=standard_blue, order=df['Category'].value_counts().index)

# Add labels and title
plt.xlabel("Category", fontsize=14)
plt.ylabel("Count of Resumes", fontsize=14)
plt.title("Count of Resumes by Category", fontsize=16)

# Add annotations
for index, value in enumerate(df['Category'].value_counts()):
    plt.text(index, value, str(value), fontsize=12, ha='center')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Show plot
plt.show()


# ### Word Map for Job categories

# In[31]:


def wordcloud(df):
    txt = ' '.join(txt for txt in df['Resume'])
    wordcloud = WordCloud(
        height=2000,
        width=4000,
        colormap=WORDCLOUD_COLOR_MAP
    ).generate(txt)

    return wordcloud


# In[34]:


plt.figure(figsize=(32, 28))

for i, category in enumerate(categories):
    wc = wordcloud(df_categories[i])

    plt.subplot(6, 4, i + 1).set_title(category)
    plt.imshow(wc)
    plt.axis('off')
    plt.plot()

plt.show()
plt.close()


#  ##### Removing most commonly used words for model building

# In[50]:


del_words = ['name', 'city', 'state', 'country', 'fullname', 'company', 'resume', 'curriculum vitae', 'address', 'phone',
             'email', 'linkedin', 'profile', 'summary', 'objective', 'experience', 'education', 'skill', 'skills',
             'reference', 'references', 'contact', 'detail', 'details', 'mail', 'gmail', 'yahoo', 'hotmail', 'mailing',
             'linkedin', 'twitter', 'facebook', 'instagram', 'website', 'web', 'url', 'www', 'year']


# ##### Term Frequency

# In[52]:


word_freq_fist = nltk.FreqDist(total_words)
most_freq = word_freq_fist.most_common(50)
print(most_freq)


# In[51]:


stop_words = set(stopwords.words('english')+['``',"''"]+del_words)
total_words = []
sentences = df['Resume'].values
cleaned_sentences = ""
for sentence in sentences:
    cleaned_sentences += sentence
    required_words = nltk.word_tokenize(sentence)
    for word in required_words:
        if word not in stop_words and word not in string.punctuation:
            total_words.append(word)


# In[53]:


wc = WordCloud(collocations=False, width=800, height=400, background_color='white').generate(cleaned_sentences)
plt.figure(figsize=(12,6), dpi=300)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
plt.show()


# ### Term frequency for Job categories 

# In[54]:


def wordfreq(df):
    count = df['Resume'].str.split(expand=True).stack().value_counts().reset_index()
    count.columns = ['Word', 'Frequency']
    return count.head(10)


# In[55]:


fig = plt.figure(figsize=(32, 64))

for i, category in enumerate(categories):
    wf = wordfreq(df_categories[i])

    fig.add_subplot(12, 2, i + 1).set_title(category)
    plt.bar(wf['Word'], wf['Frequency'])
    plt.ylim(0, 3500)

plt.show()
plt.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Models:   (Kowshik)


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Assuming you have a DataFrame 'resume_df' with extracted resume data
# and 'job_description' as the input job description

job_description="""Position type: Full-time -- Contract

Location: Hybrid - Midtown, NY

No Corp to Corp. W2 contractors only please.

 
As a Python Developer, you will:
 
This role is responsible for developing and delivering complex requirements to accomplish business goals. Key responsibilities of
the role include ensuring that software is developed to meet functional, non functional, and compliance requirements. This role
codes solutions, unit tests, and ensures the solution can be integrated successfully into the overall application/system with clear,
robust and well tested interfaces. They are familiar with development and testing practices of the bank.
¿Contribute to story refinement/defining requirements.
¿Participate and guide team in estimating work necessary to realize a story/requirement through
the delivery lifecycle.
¿Perform spike/proof of concept as necessary to mitigate risk or implement new ideas.
¿Code solutions and unit test to deliver a requirement/story per the defined acceptance criteria
and compliance requirements.
¿Utilize multiple architectural components (across data, application, business) in design and
development of client requirements.
¿Assist team with resolving technical complexities involved in realizing story work.
¿Contribute to existing test suites (integration, regression, performance), a nalyze test reports, identify
any test issues/errors, and triage the underlying cause.
¿Document and communicate required information for deployment, maintenance, support, and
business functionality.
¿Participate, contribute and can coach team members in the delivery/release (CI CD) events. e.g.
branching timelines, pull requests, issue triage, merge/conflict resolution, release notes.
 
Qualified candidates should APPLY NOW for immediate consideration! Please hit APPLY to provide the required information, and we will be back in touch as soon as possible.
 
We are currently interviewing to fill this and other similar positions. If this role is not a fit for you, we do offer a referral bonus program for referrals that we successfully place with our clients, subject to program guidelines. ASK ME HOW.

 
PAY RANGE AND BENEFITS:

Pay Range*: $70- $73 per hour on W2

 
*Pay range offered to a successful candidate will be based on several factors, including the candidate's education, work experience, work location, specific job duties, certifications, etc.
 
Benefits: Innova Solutions offers benefits(based on eligibility) that include the following: Medical & pharmacy coverage, Dental/vision insurance, 401(k), Health saving account (HSA) and Flexible spending account (FSA), Life Insurance, Pet Insurance, Short term and Long term Disability, Accident & Critical illness coverage, Pre-paid legal & ID theft protection, Sick time, and other types of paid leaves (as required by law), Employee Assistance Program (EAP).
 
ABOUT INNOVA SOLUTIONS: Founded in 1998 and headquartered in Atlanta, Georgia, Innova Solutions employs approximately 50,000 professionals worldwide and reports an annual revenue approaching $3 Billion. Through our global delivery centers across North America, Asia, and Europe, we deliver strategic technology and business transformation solutions to our clients, enabling them to operate as leaders within their fields.
 
Recent Recognitions:

One of Largest IT Consulting Staffing firms in the USA - Recognized as #4 by Staffing Industry Analysts (SIA 2022)
ClearlyRated® Client Diamond Award Winner (2020)
One of the Largest Certified MBE Companies in the NMSDC Network (2022)
Advanced Tier Services partner with AWS and Gold with MS"""


# Tokenize and preprocess the job description
vectorizer = TfidfVectorizer(stop_words='english')
job_desc_vector = vectorizer.fit_transform([job_description])

# Calculate cosine similarity between job description and resumes
resume_scores = []
for resume_text in df['Resume']:
    resume_vector = vectorizer.transform([resume_text])
    similarity_score = cosine_similarity(job_desc_vector, resume_vector)[0][0]
    resume_scores.append(similarity_score)

# Add matching scores to the resume DataFrame
df['matching_score'] = resume_scores

# Sort resumes by matching score
sorted_resumes = df.sort_values(by='matching_score', ascending=False)

# Display top matching resumes
print(sorted_resumes.head())


# In[ ]:


sorted_resumes[sorted_resumes['matching_score']>0.30]


# In[ ]:


get_ipython().system('pip install PyPDF2')


# In[ ]:


resume_df=df


# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from PyPDF2 import PdfReader


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        resume_text = ''
        for page in reader.pages:
            resume_text += page.extract_text()
    return resume_text

# Assuming you have a DataFrame 'resume_df' with extracted resume data
# and 'job_description' as the text of the job description

# Receive category input
selected_category = input("Enter the category: ")

# Receive job description input
job_description = input("Enter the job description: ")

# Receive resume upload
resume_path = input("Upload your resume (PDF): ")

# Extract text from the uploaded PDF resume
uploaded_resume_text = extract_text_from_pdf(resume_path)
#print(uploaded_resume_text)
# Extract main keywords from DataFrame for the specified category
# Initialize NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')

# Tokenize, remove stop words, and perform stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Selected category
#selected_category = "HR"  # Change this to your desired category

# Filter resumes for the selected category
category_resumes = resume_df[resume_df['Category'] == selected_category]
# Tokenize and process each resume in the selected category
all_words = []
for resume_text in category_resumes['Resume']:
    # Tokenize the text
    words = word_tokenize(resume_text.lower())
    # Remove stop words and perform stemming
    words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
    all_words.extend(words)

# Count the frequency of each word
word_freq = Counter(all_words)

# Get the top 10 keywords based on frequency
top_keywords = [keyword for keyword, _ in word_freq.most_common(100000)]  # Change 10 to the desired number of keywords

#print("Top keywords in the HR category:")
#for keyword in top_keywords:
    #print(keyword)
'''   
#Tokenize the uploaded resume
vectorizer = CountVectorizer(stop_words='english')
#uploaded_resume_vector = vectorizer.fit_transform([job_description])

# Tokenize the job description
job_description_vector = vectorizer.transform([job_description])


# Get feature names from CountVectorizer
job_description_keywords = list(job_description_vector.vocabulary_.keys())
print(job_description_keywords)
'''

#job_description_keywords = word_tokenize(job_description.lower())
job_description=preprocess(job_description)
resume_data=preprocess(uploaded_resume_text)
vectorizer = CountVectorizer(stop_words='english')





uploaded_resume_vector = vectorizer.fit_transform([resume_data])

# Tokenize the job description
job_description_vector = vectorizer.fit_transform([job_description])


# Get feature names from CountVectorizer
job_description_keywords = job_description_vector.get_feature_names_out().tolist()
resume_keywords = uploaded_resume_vector.get_feature_names_out().tolist()
#print(job_description_keywords)
# Remove stop words and perform stemming
words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]


# Check presence of main keywords in job description
matching_keywords = [keyword for keyword in resume_keywords if keyword in job_description_keywords]
matching_score = len(matching_keywords) / len(resume_keywords)

# Display matching score
print("Matching Score:", matching_score)


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess(text):
    # Tokenize text
    tokens = nltk.word_tokenize(text.lower())
    # Remove punctuation and stop words, and perform stemming
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    processed_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words and token not in punctuation]
    return ' '.join(processed_tokens)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        resume_text = ''
        for page in reader.pages:
            resume_text += page.extract_text()
    return resume_text

# Receive category input
selected_category = input("Enter the category: ")

# Receive job description input
job_description = input("Enter the job description: ")

# Receive resume upload
resume_path = input("Upload your resume (PDF): ")

# Extract text from the uploaded PDF resume
uploaded_resume_text = extract_text_from_pdf(resume_path)

# Preprocess job description and uploaded resume text
job_description = preprocess(job_description)
uploaded_resume_text = preprocess(uploaded_resume_text)

# Tokenize and vectorize the job description and uploaded resume text
vectorizer = CountVectorizer(stop_words='english')
job_description_vector = vectorizer.fit_transform([job_description])
uploaded_resume_vector = vectorizer.transform([uploaded_resume_text])

# Get feature names from CountVectorizer
job_description_keywords = vectorizer.get_feature_names_out()
resume_keywords = vectorizer.get_feature_names_out()

# Convert feature names to lists
job_description_keywords = job_description_keywords.tolist()
resume_keywords = resume_keywords.tolist()

# Check presence of main keywords in both job description and uploaded resume
matching_keywords = [keyword for keyword in resume_keywords if keyword in job_description_keywords]
matching_score = len(matching_keywords) / len(resume_keywords)

# Display matching score
print("Matching Score:", matching_score)


# In[ ]:


**HR Specialist Position:**
Seeking an experienced HR Specialist proficient in HRIS management, performance evaluation, and analytics. Must excel in project management, possess strong organizational development skills, and be adept with tools like 15Five, Paychex, Gusto, Glint, and Power BI.


# In[ ]:





# In[ ]:





# In[ ]:





# # Spacy

# In[61]:


import spacy

# Check if spaCy model is available
try:
    # Attempt to load the spaCy model
    nlp = spacy.load("en_core_web_lg")
    print("spaCy model 'en_core_web_lg' is already available.")
except OSError:
    # If spaCy model is not available, download it
    print("Downloading spaCy model 'en_core_web_lg'...")
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_lg"])
    print("spaCy model 'en_core_web_lg' downloaded successfully.")

    # Load the downloaded model
    nlp = spacy.load("en_core_web_lg")

# Now, you can proceed with your code
skill_pattern_path = "jz_skill_patterns.jsonl"


# In[62]:


ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)
nlp.pipe_names


# In[63]:


def get_skills(text):
    doc = nlp(text)
    myset = []
    subset = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            subset.append(ent.text)
    myset.append(subset)
    return subset


def unique_skills(x):
    return list(set(x))


def get_skills(text):
    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize a list to store skills
    skills = []
    
    # Loop through named entities in the document
    for ent in doc.ents:
        # Check if the entity is labeled as a skill
        if ent.label_ == "SKILL":
            # Add the skill text to the list
            skills.append(ent.text)
    
    # Return the list of skills
    return skills


def unique_skills(skills):
    unique_skills = list(set(skills))
    return unique_skills


# In[66]:


df["skills"] = df["Resume"].str.lower().apply(get_skills)
df["skills"] = df["skills"].apply(unique_skills)


# In[70]:


Job_cat = df["Category"].unique()
Job_cat = np.append(Job_cat, "ALL")


# In[71]:


Total_skills = []
if Job_Category != "ALL":
    fltr = data[data["Category"] == Job_Category]["skills"]
    for x in fltr:
        for i in x:
            Total_skills.append(i)
else:
    fltr = data["skills"]
    for x in fltr:
        for i in x:
            Total_skills.append(i)

fig = px.histogram(
    x=Total_skills,
    labels={"x": "Skills"},
    title=f"{Job_Category} Distribution of Skills",
).update_xaxes(categoryorder="total descending")
fig.show()

