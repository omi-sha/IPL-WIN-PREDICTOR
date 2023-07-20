#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


match=pd.read_csv('matches.csv')
delivery=pd.read_csv('deliveries.csv')


# In[ ]:


match.head(10)


# In[ ]:


match.isnull().sum()


# In[ ]:


match.info()


# In[ ]:


total_score_df=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[ ]:


total_score_df


# In[ ]:


total_score_df = total_score_df[total_score_df['inning'] == 1]


# In[ ]:


match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[ ]:


match_df


# In[ ]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[ ]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[ ]:


match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]


# In[ ]:


match_df=match_df[match_df['dl_applied']==0]


# In[ ]:


match_df


# In[ ]:


delivery


# In[ ]:


match_df=match_df[['match_id','city','winner','total_runs']]


# In[ ]:


delivery_df=match_df.merge(delivery,on='match_id')


# In[ ]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[ ]:


delivery_df.shape


# In[ ]:


delivery_df


# In[ ]:


#cumsum is cumulative sum
#we are using cumsum to know runs after each ball that is bowled
delivery_df['current_score']=delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[ ]:


delivery_df


# In[ ]:


delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']


# In[ ]:


delivery_df


# In[ ]:


delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])


# In[ ]:


delivery_df


# In[ ]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets'] = 10 - wickets
delivery_df


# In[ ]:


delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])


# In[ ]:


delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[ ]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[ ]:


delivery_df['result'] = delivery_df.apply(result,axis=1)


# In[ ]:


delivery_df


# In[ ]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[ ]:


final_df.head(10)


# In[ ]:


final_df=final_df.sample(final_df.shape[0])


# In[ ]:


final_df.sample()


# In[ ]:


final_df.dropna(inplace=True)


# In[ ]:


final_df = final_df[final_df['balls_left'] != 0]


# In[ ]:


from sklearn.model_selection import train_test_split
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[ ]:


X_train


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[ ]:


final_df.isnull().sum()


# In[ ]:


print(final_df.dtypes)


# In[ ]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[ ]:


try:
    pipe.fit(X_train, y_train)
    print("Model successfully fitted!")
except Exception as e:
    print("Error occurred during model fitting:")
    print(e)


# In[ ]:


pipe.fit(X_train, y_train)


# In[ ]:


y_pred = pipe.predict(X_test)


# In[ ]:


print("NaN in y_train:", y_train.isnull().sum())
print("Infinity in y_train:", y_train.isin([np.inf, -np.inf]).sum())


# In[ ]:


X_train.describe()


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score=(y_test,y_pred)


# In[ ]:


pipe.predict_proba(X_test)[10]


# In[ ]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
    


# In[ ]:


temp_df,target = match_progression(delivery_df,74,pipe)
temp_df


# In[ ]:


import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:




