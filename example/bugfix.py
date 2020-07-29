#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import git
from git import Repo
import gensim
from gensim import corpora
from gensim import similarities
from gensim.utils import simple_preprocess
import nltk
from nltk.stem import WordNetLemmatizer 
import jira
import pandas as pd
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import spacy
import pyLDAvis
import pyLDAvis.gensim 
import seaborn as sns
import matplotlib.pyplot as plt


def preprocessing(path_commit_bug,path_commit_function,path_commit,project,verbose=False,head=10):
    if verbose==True:
        print('Step 1. We need to open all paths and to load them to DataFrmame')
    with open(path_commit_bug) as f:
        commit_bug = json.load(f)
    df_commit_bug=pd.DataFrame([(a,b) for i in commit_bug for a,b in i.items()],columns=['Commit','Bug_number'])
    if verbose==True:
        print('\nFile with commit/bug:')
        display(df_commit_bug.head(head))   
    with open(path_commit_function) as f:
        commit_function = json.load(f)
    commit = [k for k in commit_function.keys() for v in commit_function[k]]
    function = [v for k in commit_function.keys() for v in commit_function[k]]
    df_commit_func=pd.DataFrame.from_dict({'Commit': commit, 'Function': function})
    if verbose==True:
        print('\nFile with commit/functions:')
        display(df_commit_func.head(head)) 
    with open(path_commit) as f:
        commit = json.load(f)
    df_commit_desc=pd.DataFrame([[a,b] for a,b in commit.items()],columns=['Commit','Describe'])
    if verbose==True:
        print('\nFile with commit/describe:')
        display(df_commit_desc.head(head)) 
    df_commit_func_describe_bug=df_commit_bug.merge(df_commit_func, on='Commit').merge(df_commit_desc, on='Commit')
    if verbose==True:
        print('\nStep 2. Lets merge three dataframes in one')
        display(df_commit_func_describe_bug.head(head)) 
    def split_func(func):
        split=func.split('.')
        return '.'.join([split[-2],split[-1]])   
    def split_desc(func):
        return func.split('\n')[0]    
    
    df_commit_func_describe_bug['Function']=df_commit_func_describe_bug['Function'].apply(lambda x:split_func(x)) 
    df_commit_func_describe_bug['Describe']=df_commit_func_describe_bug['Describe'].apply(lambda x:split_desc(x)) 
    df_commit_func_describe_bug=df_commit_func_describe_bug[df_commit_func_describe_bug['Bug_number']!='0']
    if verbose==True:
        print('\nStep 3. Lets do small preprocessing to column "Function" and "Describe"')
        display(df_commit_func_describe_bug.head(head))
        print('\nNumber of unique functions: {}'.format(len(df_commit_func_describe_bug.Function.unique())))
        print('Number of unique bugs: {}'.format(len(df_commit_func_describe_bug.Bug_number.unique())))
    print('\nNumber of unique functions: {}'.format(len(df_commit_func_describe_bug.Function.unique())))
    print('Number of unique bugs: {}'.format(len(df_commit_func_describe_bug.Bug_number.unique())))
    
    list_functions=df_commit_func_describe_bug['Function'].unique().tolist()
    dict_func_desc={}
    for function in list_functions:
        all_desription=set(df_commit_func_describe_bug['Describe'][df_commit_func_describe_bug['Function']==function].tolist())
        all_desription=' '.join(all_desription)
        dict_func_desc[function]=all_desription
    df_function_description=pd.DataFrame([(a,b) for a,b in dict_func_desc.items()],columns=['Function','Decription'])    
    if verbose==True:
        print('\nStep 4. Lets get a document describing each function')
        display(df_function_description.head(head))    
    
    def clean_str(string):
        en_stop = set(nltk.corpus.stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        ch=['%','=','\n','_','\r','.','$','/','!','?','*','(',')','{','}',';',',','<','>',"'","'",'"','"','[',']',':',';','&','#','-','?','@','+']
        for i in ch:
            string=string.replace(i,' ')
        new_string=string.split(' ') 
        new_new_string=[]
        for i in new_string:
            if len(re.findall(r'[A-Z][^A-Z]*', i))==0:
                new_new_string.append(i)
            else:
                new_new_string.extend(re.findall(r'[A-Z][^A-Z]*', i))
        new_new_string=[lemmatizer.lemmatize(i.lower().strip(),'v') for i in new_new_string if len(i)>1 and i.strip() not in en_stop and not i.isdigit()]    
        new_new_string=[i for i in new_new_string if i!='to']
        return new_new_string
    df_function_description['Description_clear']=df_function_description['Decription'].apply(lambda x:clean_str(x))
    if verbose==True:
        print('\nStep 5. Lets do preprocessing for function description')
        display(df_function_description.head(head)) 
        print('Our description is now ready for apply topic modelling')
        print('\nStep 6. We get a description of bugs directly from the site "Jira", so an internet connection is required\nIt takes time')
    
    bug_names=df_commit_func_describe_bug['Bug_number'].unique().tolist()
    def get_jira_issues(project_name, url=r"http://issues.apache.org/jira", bunch=100):
        jira_conn = jira.JIRA(url)
        all_issues=[]
        extracted_issues = 0
        while True:
            issues = jira_conn.search_issues("project={0}".format(project_name), maxResults=bunch, startAt=extracted_issues)
            all_issues.extend(issues)
            extracted_issues=extracted_issues+bunch
            if len(issues) < bunch:
                break
        return all_issues
    list_issues=get_jira_issues(project)   
    dict_bug_desc={}
    for bug in bug_names:
        for issue in list_issues:
            if bug==issue.key.split('-')[1]:
                dict_bug_desc[bug]=issue.raw['fields']['description']  
    df_bug_desc=pd.DataFrame([(a,b) for a,b in dict_bug_desc.items()],columns=['Bug','Description'])
    if verbose==True:
        display(df_bug_desc.head(head)) 
    df_bug_desc['Description_clear']=df_bug_desc['Description'].apply(lambda x:clean_str(x))
    if verbose==True:
        print('\nStep 7. Do preprocessing for "Description"')
        display(df_bug_desc.head(head)) 
        print('\nStep 8. Now we have prepared all data for applying topic modelling')
        
#     For topic modelling
    text_data=df_function_description['Description_clear'].apply(lambda x:list(x)).tolist()
    function_names=df_function_description['Function'].tolist()
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]    
    return (corpus,dictionary,df_function_description,df_commit_func_describe_bug,df_bug_desc,function_names)   

def topic_modeling(corpus,dictionary,df_function_description,df_commit_func_describe_bug,df_bug_desc,function_names,NUM_TOPICS=15,mod='LDA',verbose=False,num_words=4,only_vis=False):
    if mod=='LDA':
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    else:
        model = gensim.models.LsiModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary)
   
    if verbose==True:
        print('Number of topics:{}\nModel:{} '.format(NUM_TOPICS,mod))
        
    df_function_description['Function_topics']=df_function_description['Description_clear'].apply(lambda x:model.get_document_topics(dictionary.doc2bow(x)))
    result1=df_function_description[['Function','Description_clear','Function_topics']]
    
    if verbose==True:
        print('Topics from function description')
        display(result1.head(30))
        print('\nTopics words')
        topics = model.print_topics(num_words=num_words)
        for topic in topics:
            print(topic)
        print('\nTopic visualization')
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        display(vis)
    if only_vis==True:
        topics = model.print_topics(num_words=num_words)
        for topic in topics:
            print(topic)
        print('\nTopic visualization')
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        display(vis)
        
    dict_bug_topic={}
    for index, row in df_bug_desc.iterrows():
        dict_bug_topic[row['Bug']]=model[dictionary.doc2bow(row['Description_clear'])] 
    result2=pd.DataFrame([(a,b) for a,b in dict_bug_topic.items()],columns=['Bug_number','Topic'])
    if verbose==True:
        print('Topics from bugs description')
        display(result2.head(30))

    def recomendation(top=(len(df_function_description)+1)):
        dict_recomendations={}
        index = similarities.MatrixSimilarity(model[corpus],num_features=len(dictionary))
        for bug_number,topic_vector in dict_bug_topic.items():
            similarity=enumerate(index[topic_vector])
            similarity = sorted(similarity, key=lambda item: -item[1])
            list_func=[]
            k=1
            for function_row, similar in similarity:
                if k<=top:
                    list_func.append([function_names[function_row],similar])
                    k+=1
                else:
                    continue
            dict_recomendations[bug_number]=list_func 
        return dict_recomendations   
    result3=recomendation()
    if verbose==True:
        print('Function recomendarion according to similarity')
        print('Example for bug {}'.format(list(result3.keys())[0]))
        display(pd.DataFrame([(a[0],a[1]) for a in result3[list(result3.keys())[0]]],columns=['Fuction','Similarity']).head(30))

    list_bugs=df_commit_func_describe_bug['Bug_number'].unique().tolist()
    dict_bug_fun={}
    for bug in list_bugs:
        all_func=df_commit_func_describe_bug['Function'][df_commit_func_describe_bug['Bug_number']==bug].tolist()
        dict_bug_fun[bug]=all_func
    result4=dict_bug_fun 
    if verbose==True:
        print('Actual functions for the same bug {}'.format(list(result3.keys())[0]))
        display(pd.DataFrame([a for a in result4[list(result3.keys())[0]]],columns=['Fuction']).head(30))
        print('\nEVALUATION')
        print('Top K')
        
    def top_k():
        result={}
        for key in result4.keys():
            test=result4[key]
            pred=[i[0] for i in result3[key]]
            topk=[]
            for i in test:
                place=pred.index(i)
                topk.append(place)
            result[key]=[max(topk)+1,len(test),len(df_function_description),((max(topk)+1)*100)/len(df_function_description)]  
           
        return pd.DataFrame([(a,b[0],b[1],b[2],b[3]) for a,b in result.items()],columns=['Bag','Top_k','Functions_in_bag','Count_all_functions','%'])               
    result5=top_k()
    av5=result5['%'].mean()
    if verbose==True:
        display(result5.head(30))
        print('Average Top K %:',av5)
        print('\nPrecision at K')
    
    def top_k2(n=100):
        y_pred=recomendation(top=n)
        result={}
        for bug in result4.keys():
            s=0
            pred=[i[0] for i in y_pred[bug]]
            for function in result4[bug]:
                if function in pred:
                    s+=1
            result[bug]=[(s/len(result4[bug]))*100,len(result4[bug]),n,s]
        return pd.DataFrame([(a,b[0],b[1],b[2],b[3]) for a,b in result.items()],columns=['Bug','%','Functions_in_bug','Number_recomendations','Correctly_suggested'])  
    result6=top_k2()
    av6=result6['%'].mean()
    if verbose==True:
        display(result6.head(30))
        print('Average Precision at K %:',av6)
        
    def top_k3():
        y_pred=recomendation()
        result={}
        for bug in result4.keys():
            size=len(result4[bug])
            s=0  
            pred=y_pred[bug][:size]
            pred=[i[0] for i in pred]
            for function in result4[bug]:
                if function in pred:
                    s+=1
            result[bug]=[(s/len(result4[bug]))*100,len(result4[bug]),s]
        return pd.DataFrame([(a,b[0],b[1],b[2]) for a,b in result.items()],columns=['Bug','%','Functions_in_bug','Correctly_suggested'])  
    result7=top_k3()
    av7=result7['%'].mean()
    if verbose==True:
        print('\nK3')
        display(result7.head(30))
        print('Average K3 %:',av7)    
    return (NUM_TOPICS,av5,av6,av7)  

def search(project,num_topics=50):
    """ Search the best number topics
    
    Paramets:
    project:tupple, first element is path to folder with files;second element is project name
    Note: project name is important for correct search in Jira
    num_topics: searching diapasone 
    
    Files in folder: must have same names:
    - "bugs_data.json"
    - "methods_per_commit.json"
    - "commits_desc.json"
    
    """
    list_topic=range(2,num_topics,1)
    num=[]
    typ=[]
    k=[]
    path_commit_bug=project[0]+'/bugs_data.json'
    path_commit_function=project[0]+'/methods_per_commit.json'
    path_commit=project[0]+'/commits_desc.json'
    proj=project[1]
    print('Project:',proj)
    if True:
        corpus,dictionary,df_function_description,df_commit_func_describe_bug,df_bug_desc,function_names=preprocessing(path_commit_bug,path_commit_function,path_commit,project=proj,verbose=False)
        for i in list_topic:
            try:
                res=topic_modeling(corpus,dictionary,df_function_description,df_commit_func_describe_bug,df_bug_desc,function_names,NUM_TOPICS=i,verbose=False)
                num.extend([res[0],res[0]])
                typ.extend(['Top K','Precision at K'])
                k.extend([res[1],res[2]])
            except:
                continue
        result=pd.DataFrame.from_dict({'Num_topic': num, 'Metric': typ,'Result': k})  
        best_num_topic=result[result['Metric']=='Top K'].sort_values(by='Result',ascending=True)['Num_topic'].tolist()[0]
        plt.figure(figsize = (10, 8), facecolor = None) 
        ax = sns.lineplot(x='Num_topic', y='Result',hue='Metric', data=result)
        plt.xlabel("Num_topics")
        plt.ylabel("%")
        plt.show()  
    #     ______________visualization_____________
        print('Best Top K when number topics=',best_num_topic)
        topic_modeling(corpus,dictionary,df_function_description,df_commit_func_describe_bug,df_bug_desc,function_names,NUM_TOPICS=best_num_topic,verbose=False,only_vis=True)
