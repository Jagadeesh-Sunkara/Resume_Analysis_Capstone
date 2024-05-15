# Automated Resume Evaluation with NLP, SpaCy, and Topic Modeling

## Project Overview:
This repository contains the codebase for an automated resume evaluation system. The system leverages natural language processing (NLP) techniques, specifically SpaCy, along with topic modeling, to streamline the resume screening process.

## Introduction
In today's competitive job market, resume screening poses challenges in terms of time consumption and fairness. This project aims to address these issues by automating the resume evaluation process using NLP, SpaCy, and topic modeling techniques.

## Hypothesis / Business Use:
Our hypothesis is that by employing machine learning to automate the screening and shortlisting of resumes, we can expedite the candidate selection process, making it more efficient and streamlined. This hypothesis guides our data analysis and modeling approach, aimed at simplifying recruitment procedures and improving the quality of hiring decisions.

## Approach:
Our approach to automated resume evaluation involves a systematic process aimed at streamlining the screening and shortlisting of resumes. We began by acquiring a dataset of over 2400 resume examples, each labeled with its corresponding job category. After cleaning the data to ensure its suitability for analysis, we conducted Exploratory Data Analysis (EDA) to uncover patterns and insights. Next, we built a model that compares job descriptions with uploaded resumes, utilizing natural language processing (NLP) techniques and SpaCy for entity recognition. Additionally, we employed topic modeling, specifically Latent Dirichlet Allocation (LDA), to identify hidden themes within the dataset. Through this approach, we aim to simplify recruitment procedures, enhance the efficiency of hiring decisions, and ultimately revolutionize the resume screening process.

## Dependencies
- Python 3.x
- SpaCy
- scikit-learn
- pandas
- numpy

## Data Source:
The dataset used in this project was obtained from Kaggle, available at [this link](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). It comprises over 2400 resume examples, each labeled with its corresponding job category. While initially planned to be scraped from the LiveCareer website, we encountered challenges due to dynamic content and changes in website structure. Therefore, we opted to utilize pre-existing scraped data instead. This dataset underwent thorough cleaning to ensure its suitability for analysis and modeling. Through this dataset, we aim to automate the resume evaluation process, streamline recruitment procedures, and improve the quality of hiring decisions.

## Data Cleaning:
The dataset underwent thorough cleaning to ensure its suitability for analysis:
- Unused columns like ID and Resume_html were dropped.
- Text data was preprocessed to remove symbols, punctuation, numbers, and irrelevant words.
- Commonly occurring but irrelevant words like 'city' and 'company' were removed to enhance model accuracy.

## Features:
- Automated Resume Evaluation: The system automates the process of screening resumes by matching them with job descriptions.
- NLP and SpaCy: Utilizes NLP techniques and the SpaCy library for text analysis and entity recognition.
- Topic Modeling: Implements Latent Dirichlet Allocation (LDA) for identifying common themes and skills within resumes.
- Feedback Mechanism: Provides feedback to users based on the compatibility between the resume and the job description.

## Key Implementations:
- Automated Resume Evaluation System: Developed a system using NLP techniques and SpaCy for automated resume screening and shortlisting.
- Data Preprocessing: Conducted thorough data cleaning to remove noise and irrelevant information from the dataset.
- Exploratory Data Analysis (EDA): Analyzed job category distributions and identified common keywords to inform model development.
- Model Building: Built a model to compare job descriptions with uploaded resumes, utilizing vectorization and keyword matching techniques.
- SpaCy Modeling: Constructed a Resume Analyzer using SpaCy for entity recognition and skill matching.
- Topic Modeling (LDA): Employed Latent Dirichlet Allocation (LDA) for identifying hidden themes within the resume dataset.

## Exploratory Data Analysis (EDA):
1. **Job Category Distribution**: Analyzed resume distribution across job sectors to understand diversity.
2. **Common Keywords Identification**: Identified prevalent skills and qualifications across resumes.
3. **Visualization Techniques**: Used histograms, pie charts, and word clouds to visualize data distribution and keyword prevalence, guiding analysis.

## Model Building and Results:
1. **Model Development**: Constructed a model to compare job descriptions with uploaded resumes, leveraging natural language processing (NLP) techniques and SpaCy for text analysis.
2. **SpaCy Modeling**: Integrated SpaCy for entity recognition and skill matching within resumes, enhancing the accuracy of the evaluation process.
3. **Keyword Matching**: Implemented a mechanism to calculate a compatibility score between job descriptions and resumes based on the presence of common keywords.
4. **Result Interpretation**: Provided feedback to users based on the compatibility score:
   - Low Compatibility: Indicated the need to update the resume with a sad emoji.
   - Moderate Compatibility: Notified users of a good fit with a neutral emoji.
   - High Compatibility: Recognized strong candidacy with a happy emoji.
5. **Impact**: The automated system streamlined the resume evaluation process, making it more efficient and fair, thus enhancing recruitment procedures and improving the quality of hiring decisions.

## Source Code:
- GitHub: [Resume_Analysis_Capstone](https://github.com/Jagadeesh-Sunkara/Resume_Analysis_Capstone)
- Deepnote: [Capstone Resume Analysis](https://deepnote.com/app/capstone-resume-analysis/Resume-Analysis-b8d3f3f0-fa5d-48c0-8f8e-c97ad3726e67)

## References:
1. Deepnote. (n.d.). spaCy Resume Analysis. Retrieved from [https://deepnote.com/app/abid/spaCy-Resume-Analysis-81ba1e4b-7fa8-45fe-ac7a-0b7bf3da7826](https://deepnote.com/app/abid/spaCy-Resume-Analysis-81ba1e4b-7fa8-45fe-ac7a-0b7bf3da7826)
2. Dutta, G. (n.d.). Resume screening using machine learning

