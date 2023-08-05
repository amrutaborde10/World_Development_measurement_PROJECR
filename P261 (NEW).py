#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading required libraries
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.cluster import k_means
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# # Reading Dataset

# In[2]:



world= pd.read_excel(r"C:\Users\Hp\Desktop\World_development_mesurement.xlsx")
world


# # Purpose
# 1. Analyzing the impact of economic indicators on a country's development status.
# 2.Analyzing renewable energy adoption and its influence on carbon emissions.
# 3. Predecting GDP grwoth rate based on various economic indicators.
# 4.Predicting poverty levels using demographic and economic variables.

# # Limitation 
# 1.The dataset may contain missing or incomplete data for certain countries or indicators. Inaccuracies or errors in data collection, reporting, or entry could also be present, affecting the overall quality of the analysis
# 
# 2.The dataset may not cover all countries or regions worldwide, which could lead to biases or limited generalizability of the findings, particularly for smaller or less well-known countries.
# 
# 
# 4.World development is an ongoing process with factors continuously changing over time. The dataset might not capture the dynamism and evolution of various development indicators adequately.
# 
# 5.Countries have diverse characteristics and challenges. Treating all countries as a homogeneous group could overlook specific challenges faced by individual nations.
# 
# 

# # Exploratory Data Analysis

# In[3]:


# information about dataset
info = pd.DataFrame(world.info())


# In[4]:


# Renaming columns
world.rename(columns={'Birth Rate':'birth_rate', 'Business Tax Rate':'business_tax_rate','CO2 Emissions':'co2_emission','Country':'country',
                      'Days to Start Business':'days_to_start_business','Ease of Business':'ease_of_business','Energy Usage':'energy_usage',
                      'Gdp':'gdp','Health Exp % GDP':'health_exp_percent_GDP','Health Exp/Capita' :'health_exp_percapita',
                      'Hours to do Tax':'hours_to_do_tax','Infant Mortality Rate':'infant_mortality_rate','Internet Usage':'internet_usage',
                      'Lending Interest':'lending_interest','Life Expectancy Female':'life_expectancy_female','Life Expectancy Male':'life_expectancy_male',
                      'Mobile Phone Usage':'mobile_phone_usage','Number of Records':'no_of_records','Population 0-14':'population_0_14',
                      'Population 15-64':'population_15_64', 'Population 65+':'population_65_plus','Population Total':'population_total','Population Urban':'population_urban',
                      'Tourism Inbound':'tourism_inbound','Tourism Outbound':'tourism_outbound'}, inplace=True)


# In[5]:


world.head()


# In[6]:


world.isnull().sum()


# In[7]:


# Dropping columns "Ease of Business & No. of Records"
'''here we are deleting Ease of business because there are 2519 null values out of 2704 rows ,
 and we also delete  Number of Records having only 1 as a value '''

world_2 = world.drop(columns=['ease_of_business','no_of_records'] , axis=1)


# In[8]:


# Removing Special Characters ($, %)
''' Here we are replacing  "$", "%" and "," symbol with space'''

colstocheck = world_2.columns
world_2[colstocheck] = world_2[colstocheck].replace({'\$|\%|\,':""}, regex = True)

world_2.head()


# In[9]:


# Calculating percentage of missing/null values
percent_missing = round(world_2.isnull().sum() * 100 / len(world_2),3)
percent_missing.sort_values(ascending=False)


# In[10]:


# Heatmap for null values
plt.figure(figsize=(16,8))
sns.heatmap(world_2.isnull())
plt.show()


# In[11]:


# Converting Object datatypes to Numeric Datatypes
world_2["GDP"] = pd.to_numeric(world_2["GDP"], errors='coerce')
world_2["tourism_inbound"] = pd.to_numeric(world_2["tourism_inbound"], errors='coerce')
world_2["tourism_outbound"] = pd.to_numeric(world_2["tourism_outbound"], errors='coerce')
world_2['business_tax_rate'] = pd.to_numeric(world_2['business_tax_rate'], errors='coerce')
world_2['health_exp_percapita'] = pd.to_numeric(world_2['health_exp_percapita'], errors='coerce')


# In[12]:


# Checking dataset info after converting the datatypes
world_2.info()


# In[13]:


# Describing the data
world_2.describe().T


# ## Interpretations on Data Description
# 
# * co2_emission, energy_usage, GDP, health_exp_percapita, population_total, tourism_inbound, tourism_outbound mean values are far away from the 50% of the data.
# 
# * Remaining features are somewhat nearby the 50% of data.

# In[14]:


features = world_2.drop(columns='country',axis = 1)


# In[15]:


features.skew().sort_values(ascending=False)


# In[16]:


for i in features:

    world_2[i].hist(bins=25)
    plt.ylabel('Count')
    plt.title(i)
    plt.show()


# ## Interpretations on Skewness of Features
# 
# * lending_interest, GDP, population_total, co2_emission, days_to_start_business, tourism_inbound, energy_usage, tourism_outbound features are highly skewed
# 
# * Need to find a way to reduce the skewness for the above mentioned features

# # Before imputation let us check for outliers using boxplot and histograms

# In[17]:


for i in features:
    plt.figure(figsize=(12,2))
    plt.boxplot(world_2[i].dropna(),vert=False,)
    plt.title(i)
    plt.show()


# ## Interpretations on Boxplots
# * business_tax_rate, co2_emission, days_to_start_business, energy_usage, GDP, health_exp%_GDP, health_exp_percapita, hours_to_do tax, population_total, tourism_inbound, tourism_outbound are having more number of Outliers
# 
# * infant_mortality_rate, life_expectancy_female, life_expectancy_male, moble_phone_usage, population_15_64, population_65+ are having few number of Outliers
# 
# * The boxplots of Birth Rate,Ease of Business ,Mobile Phone Usage,Internet Usage,Infant Mortality Rate,Life Expectancy Female,Life Expectancy Male,,Population 0-14,Population 15-64,Population 65+, Population Urban looks fine.
# 
# *  few outliers are detected but it make sense as it is global data and not much deviated from the actual values.
# 
# *  max(Business Tax Rate) is around 340% , which means paying 340 rupees as tax for every 100 rupees profit. The global highest Business Tax Rate is around 55% , so assuming the max value to be 60% and replacing all the ouliers (i.e., above 60%) with np.nan and will fill them later using imputation techniques
# 
# *  max(Days to Start Business) is 694 days. Accounting 18-20 business days a month it takes like around 3 years , comparing it to real time global values the max time required to start a business is around 50 days.Based on the boxplot assuming the max days to start a business is 80 days and replacing all the outliers with np.nan and will figure a way to fill them up with sensible number later based on all other parameters
# 
# *  Based on the boxplot assuming 600 hours as max(hours to do tax ) and replacing all the outliers with np.nan and will figure a way to fill them up with sensible number later based on all other parameters.

# In[18]:


world_2['business_tax_rate'] =np.where(world_2['business_tax_rate']>60 ,np.nan,world_2['business_tax_rate'])
world_2['days_to_start_business'] =np.where(world_2['days_to_start_business']>80 ,np.nan,world_2['days_to_start_business'])
world_2['hours_to_do_tax'] =np.where(world_2['hours_to_do_tax']>600 ,np.nan,world_2['hours_to_do_tax'])


# ###  Note:  Removing outliers of the business_tax_rate, days_to_start_business, hours_to_do_tax. there is no sense as the feature values are practically taken from the real world and depends on the population and gdp of the countries which varey significantly from one another
#   

# In[19]:


# Calculating percentage of missing/null values
percent_missing =  world_2.isnull().sum() * 100 / len(world_2)
percent_missing.sort_values(ascending=False)


# ## Interpretations on Missing Values
# * Most of the features are having missing values.
# 
# * hours_to_do tax, business_tax_rate, days_to_start_business, energy_usage, lending_interest are the top 5 features with highest percentage of missing values.

# In[20]:


# Describing the data
world_2.describe().T


# # Country & Feature wise Mean Imputation
# * Imputing missing values with country and feature wise mean, as if the imputation is done through the mean of feature may mislead the data.

# In[21]:


for i in world_2.columns:
    null_df = world_2[world_2[i].isnull()]
    print(i, " is having missing values from ", null_df['country'].nunique(), " countries.")
    print("--------------------------------------------------------------------------------")


# In[22]:


world_fill_mean_na = world_2.copy()


# In[23]:


mean_fill_cols = world_fill_mean_na.drop(columns=['country', 'population_total']).columns


# In[24]:


final_df_after_mean_imputation = pd.DataFrame()
for col in mean_fill_cols:
    fill_na_df = world_fill_mean_na[['country', col]]
    fill_na_df[col] = fill_na_df.groupby("country")[col].transform(lambda x: x.fillna(x.mean()))
    final_df_after_mean_imputation = pd.concat([final_df_after_mean_imputation, fill_na_df], axis=1)


# In[25]:


final_df_after_mean_imputation.shape  


# In[26]:


final_df_after_mean_imputation = final_df_after_mean_imputation.drop(columns=['country'])


# In[27]:


final_df_after_mean_imputation['country'] = world_fill_mean_na['country']
final_df_after_mean_imputation['population_total'] = world_fill_mean_na['population_total']


# In[28]:


final_df_after_mean_imputation.head()


# In[29]:


# Example of Country which is having null values for the entire feature
example_df = final_df_after_mean_imputation[['country', 'energy_usage']]
example_df[example_df['country'] == 'Afghanistan']


# In[30]:


# Calculating percentage of missing/null values after imputing with country and feature wise mean
percent_missing =  final_df_after_mean_imputation.isnull().sum() * 100 / len(world_2)
percent_missing.sort_values(ascending=False)


# In[31]:


final_df_after_mean_imputation.describe().T


# # Median Imputation

# In[32]:


median_data = final_df_after_mean_imputation.drop(columns = ["country"],axis=1)
median_data.fillna(median_data.median(), inplace=True)


# ### Whichever countries having null values for the entire feature, imputing those countries with median value of that feature from all the countries.

# In[33]:


median_data.describe().T


# In[34]:


median_data['country'] = final_df_after_mean_imputation['country']


# In[35]:


median_data.head()


# In[36]:


median_data.isnull().sum()


# median imputation is giving better results, hence going ahead with median imputation

# In[37]:


data_after_imputation = median_data[['birth_rate', 'business_tax_rate', 'co2_emission', 'country',
       'days_to_start_business', 'energy_usage', 'GDP', 'health_exp_percent_GDP',
       'health_exp_percapita', 'hours_to_do_tax', 'infant_mortality_rate',
       'internet_usage', 'lending_interest', 'life_expectancy_female',
       'life_expectancy_male', 'mobile_phone_usage', 'population_0_14',
       'population_15_64', 'population_65_plus', 'population_total',
       'population_urban', 'tourism_inbound', 'tourism_outbound']]


# ### Data Sanity Check

# In[38]:


pd.set_option('display.max_columns', None)
world_2.head()


# In[39]:


data_after_imputation.duplicated().sum()


# In[40]:


world_development_final_data =  data_after_imputation.copy()


# In[41]:


world_development_final_data.skew().sort_values(ascending=False)


# # Boxplots after Imputation

# In[42]:


features = world_development_final_data.drop(columns="country",axis=1)
for i in features:
    plt.figure(figsize=(12,2))
    plt.boxplot(world_development_final_data[i].dropna(),vert=False,)
    plt.title(i)
    plt.show()


# # Data Transformations

# In[43]:


# Data Transformtions
world_development_final_data['lending_interest'] = np.log(world_development_final_data['lending_interest'])
world_development_final_data[['GDP', 'co2_emission', 'population_total', 'energy_usage', 'tourism_inbound', 'tourism_outbound']] = np.sqrt(world_development_final_data[['GDP', 'co2_emission', 'population_total', 'energy_usage', 'tourism_inbound', 'tourism_outbound']])


# * Applying log transformation for lending_interest feature
# * Applying Square root transformation for 'GDP', 'co2_emission', 'population_total', 'energy_usage', 'tourism_inbound', 'tourism_outbound' features

# In[44]:


world_development_final_data.skew().sort_values(ascending=False)


# In[45]:


# Correaltion matrix
world_development_final_data.corr()


# In[ ]:





# # Boxplots After Data Transformation

# In[46]:


features_1 = world_development_final_data.drop(columns="country",axis=1)
# Histograms for numerical features
for i in features_1:
    world_development_final_data[i].hist(bins=25)
    plt.ylabel('Count')
    plt.title(i)
    plt.show()


# In[47]:


#Top 30 countries with highest and lowest GDP

df_gdp_country = world_development_final_data.groupby('country', group_keys=False).apply(lambda x: x.loc[x.GDP.idxmax()])

# df_gdp_country
top40 = df_gdp_country['GDP'].sort_values(ascending=False)[:30]
bot40 = df_gdp_country['GDP'].sort_values()[:30]

plt.figure(figsize=(20,8), dpi=80)
top = sns.barplot(x=top40, y=top40.index, log=True)
plt.title('Top 30 countries with Highest GDP')
plt.show()

plt.figure(figsize=(20,8), dpi=80)
bot = sns.barplot(x=bot40, y=bot40.index, log=True)
plt.title('Top 30 countries with Lowest GDP')
plt.show()


# In[48]:


#Top 30 countries highest and lowest Tourism Inbound

df_ti_country = world_development_final_data.groupby('country', group_keys=False).apply(lambda x: x.loc[x['tourism_inbound'].idxmax()])

# df_gdp_country
top30 = df_ti_country['tourism_inbound'].sort_values(ascending=False)[:30]
bot30 = df_ti_country['tourism_inbound'].sort_values()[:30]

plt.figure(figsize=(20,8), dpi=80)
sns.barplot(x=top30, y=top30.index, log=True)
plt.title('Top 30 countries with Highest Tourism Inbound')
plt.show()

plt.figure(figsize=(20,8), dpi=80)
sns.barplot(x=bot30, y=bot30.index, log=True)
plt.title('Top 30 countries with lowest Tourism Inbound')
plt.show()
    


# In[49]:


#Top 30 countries highest and lowest Tourism Outbound¶

df_to_country = world_development_final_data.groupby('country', group_keys=False).apply(lambda x: x.loc[x['tourism_outbound'].idxmax()])

# df_gdp_country
top30 = df_to_country['tourism_outbound'].sort_values(ascending=False)[:30]
bot30 = df_to_country['tourism_outbound'].sort_values()[:30]

plt.figure(figsize=(20,8), dpi=80)
top_bp = sns.barplot(x=top30, y=top30.index, log=True)
plt.title("Top 30 countries with highest Tourism Outbound")
plt.show()

plt.figure(figsize=(20,8), dpi=80)
bot_bp = sns.barplot(x=bot30, y=bot30.index, log=True)
plt.title("Top 30 countries witth lowest Tourism Outbound")
plt.show()


# In[50]:


#Top 30 countries highest and lowest Energy Usage

df_eu_country = world_development_final_data.groupby('country', group_keys=False).apply(lambda x: x.loc[x['energy_usage'].idxmax()])

# df_gdp_country
top30 = df_eu_country['energy_usage'].sort_values(ascending=False)[:30]
bot30 = df_eu_country['energy_usage'].sort_values()[:30]

plt.figure(figsize=(20,8), dpi=80)
top_bp = sns.barplot(x=top30, y=top30.index, log=True)
plt.title("Top 30 countries highest Energy Usage")
plt.show()

plt.figure(figsize=(20,8), dpi=80)
bot_bp = sns.barplot(x=bot30, y=bot30.index, log=True)
plt.title("Top 30 countries lowest Energy Usage")
plt.show()


# In[51]:


#Top 30 countries highest and lowest CO2 Emissions

df_ce_country = world_development_final_data.groupby('country', group_keys=False).apply(lambda x: x.loc[x['co2_emission'].idxmax()])

# df_gdp_country
top30 = df_ce_country['co2_emission'].sort_values(ascending=False)[:30]
bot30 = df_ce_country['co2_emission'].sort_values()[:30]

plt.figure(figsize=(20,8), dpi=80)
top_bp = sns.barplot(x=top30, y=top30.index, log=True)
plt.title("Top 30 countries highest CO2 Emission")
plt.show()

plt.figure(figsize=(20,8), dpi=80)
bot_bp = sns.barplot(x=bot30, y=bot30.index, log=True)
plt.title("Top 30 countries lowest CO2 Emission")
plt.show()


# ## Hopkins test
# *The Hopkins statistic, commonly known as the Hopkins test, is a measure used to assess the clustering tendency of a dataset. It helps determine whether a dataset is suitable for clustering analysis. The test is based on the comparison of the distribution of distances between data points and the distribution of distances between data points and randomly generated points.

# In[52]:


#Calculating the Hopkins statistic 
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan

#Function to calculate Hopkins test score
def hopkins(X):
    d = X.shape[1]
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
        
    return H


# In[53]:


features_1.columns


# In[54]:


hopkins(features_1)


# Hopkins test results will vary as it picks a set of samples each time. On running it multiple times, it can be seen that this data set gives Hopkins statistic value in the range of 0.88 to 0.97 and hence our dataset is good for clustering and lets proceed our analysis

# # Standardization

# In[55]:


# Scaling on numerical features
scaler = MinMaxScaler() # instantiate scaler
scaled_info = scaler.fit_transform(features_1)# fit and transform numerical data of given dataset
scaled_df = pd.DataFrame(scaled_info, columns = features_1.columns) # convert to dataframe
scaled_df.head()


# # Agglomeritive or Hierarchical Clustering

# In[56]:


# create dendrogram
world_development1=world_development_final_data.copy()
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(20, 15))  
dendograms=sch.dendrogram(sch.linkage(scaled_info,"complete"))


# In[57]:


model=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="complete")
cluster_numbers=model.fit_predict(scaled_info)


# In[58]:


world_development1['Hierarchical_Cluster_tag']=cluster_numbers


# In[59]:


world_development1.Hierarchical_Cluster_tag.unique()


# In[60]:


world_development1['Hierarchical_Cluster_tag'].value_counts()


# In[61]:


# silhouette score
sil_score= silhouette_score(scaled_info, model.labels_)
print('silhouette score: ',sil_score)


# # K-Means Clustering

# In[62]:


world_development1 = world_development1.drop(columns=['Hierarchical_Cluster_tag'], axis=1)


# In[63]:


from sklearn.cluster import KMeans
WCSS = []
for i in range(1,10):
    k = KMeans(n_clusters=i).fit(scaled_info)
    WCSS.append(k.inertia_)
plt.plot(range(1,10),WCSS)
plt.title('Elbow Method')
plt.xlabel('no.of clusters')
plt.ylabel('WCSS')
plt.show()


# In[64]:


pip install kneed


# In[65]:


## Getting Optimal K value
from kneed import KneeLocator

y = WCSS
x = range(1, len(y)+1)

kn = KneeLocator(x, y, curve= 'convex', direction='decreasing')
print("Optimal Number of Clusters is ", kn.knee)

plt.plot(x, y, 'bx--')
plt.xlabel('Number of Clusters')
plt.ylabel('Distances')
plt.vlines(kn.knee, plt.ylim()[1], plt.xlim()[1], linestyles='dotted')
plt.show()


# #### From the above elbow method diagram we can say that no.of clusters = 3.

# In[66]:


model1=KMeans(n_clusters=3, random_state=5, init='k-means++', n_init=15,
               max_iter=500,)
cluster_numbers=model1.fit_predict(scaled_info)


# In[67]:


world_development1['Kmeans_Cluster_tag']=cluster_numbers


# In[68]:


world_development1['Kmeans_Cluster_tag'].value_counts()


# In[69]:


import matplotlib.pyplot as plt

world_development1['Kmeans_Cluster_tag'].value_counts().plot(kind='bar',figsize = (8,6))
plt.xlabel("clusters",loc="center",fontsize= 20,fontweight= "bold")
plt.ylabel("ID Counts",loc="center",fontsize=20,fontweight= "bold")
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[70]:


# silhouette score
sil_score_kmeans= silhouette_score(scaled_info, model1.labels_)
print('silhouette score: ',sil_score_kmeans)


# In[71]:


world_development1.groupby('Kmeans_Cluster_tag').mean().round(1).reset_index()


# # Principal Component Analysis

# In[72]:


from sklearn.decomposition import PCA
pca = PCA()
pca_values = pca.fit_transform(scaled_df)
variance = pca.explained_variance_ratio_ # it gives importance of each and every PCA
variance


# In[73]:


var_cumulative = np.cumsum(np.round(variance, decimals= 3)*100)
var_cumulative


# In[74]:


plt.figure(figsize=(10,4))
plt.scatter(x=[i+1 for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_,
           s=200, alpha=0.75,c='blue',edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
plt.xlabel("Principal components",fontsize=10)
plt.xticks([i+1 for i in range(len(pca.explained_variance_ratio_))],fontsize=15)
plt.yticks(fontsize=10)
plt.ylabel("Explained variance ratio",fontsize=10)
plt.show()


# In[75]:


# PCA for 3 components
pca = PCA(n_components=3)
pca_values = pca.fit_transform(scaled_df)
variance = pca.explained_variance_ratio_
variance


# In[76]:


var_cumulative = np.cumsum(np.round(variance, decimals= 3)*100)
var_cumulative


# In[77]:


## Creating Dataframe for top 7 PCA values
# pca_df = pd.DataFrame(pca_values ,columns=["PCA1","PCA2","PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8", "PCA9"])
pca_df = pd.DataFrame(pca_values ,columns=["PCA1","PCA2","PCA3"])
pca_df.head(10)


# # Agglomeritive or Hierarchical Clustering using PCA values

# In[78]:


# create dendrogram
world_development2=world_development_final_data.copy()
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(20, 15))  
dendograms=sch.dendrogram(sch.linkage(pca_values,"complete"))


# In[79]:


model=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="complete")
cluster_numbers=model.fit_predict(pca_values)


# In[80]:


world_development2['Hierarchical_Cluster_tag']=cluster_numbers


# In[81]:


world_development2.Hierarchical_Cluster_tag.unique()


# In[82]:


world_development2['Hierarchical_Cluster_tag'].value_counts()


# In[83]:


# silhouette score
sil_score= silhouette_score(pca_values, model.labels_)
print('silhouette score: ',sil_score)


# # K-means clustering using PCA values

# In[84]:


world_development2 = world_development2.drop(columns=['Hierarchical_Cluster_tag'], axis=1)


# In[85]:


from sklearn.cluster import KMeans
WCSS = []
for i in range(1,10):
    k = KMeans(n_clusters=i, init='k-means++',
               n_init=15,
               max_iter=500,
               random_state=5).fit(pca_values)
    WCSS.append(k.inertia_)
plt.plot(range(1,10),WCSS)
plt.title('Elbow Method')
plt.xlabel('no.of clusters')
plt.ylabel('WCSS')
plt.show() 


# In[86]:


# We will also use the Silhouette score to determine an optimal number.

k = [2,3,4,5,6,7,8,9]

#  Silhouette score for MinMaxScaler Applied on data .

for n_clusters in k:
    clusterer1 = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels1 = clusterer1.fit_predict(pca_values)
    sil_score1= silhouette_score(pca_values, cluster_labels1)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", sil_score1)


# In[87]:


## Getting Optimal K value
from kneed import KneeLocator

y = WCSS
x = range(1, len(y)+1)

kn = KneeLocator(x, y, curve= 'convex', direction='decreasing')
print("No. of Optimal Clusters is ", kn.knee)

plt.plot(x, y, 'bx--')
plt.xlabel('Number of Clusters')
plt.ylabel('Distances')
plt.vlines(kn.knee, plt.ylim()[1], plt.xlim()[1], linestyles='dotted')
plt.show()


# In[88]:


kmeans_model=KMeans(n_clusters=3, init='k-means++',
               n_init=15,
               max_iter=500,
               random_state=5)
cluster_numbers = kmeans_model.fit_predict(pca_values)


# In[89]:


data_after_imputation['Cluster']=cluster_numbers


# In[90]:


data_after_imputation['Cluster'].value_counts()


# In[91]:


import matplotlib.pyplot as plt

data_after_imputation['Cluster'].value_counts().plot(kind='bar',figsize = (8,6))
plt.xlabel("clusters",loc="center",fontsize= 20,fontweight= "bold")
plt.ylabel("ID Counts",loc="center",fontsize=20,fontweight= "bold")
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()


# In[92]:


# silhouette score
sil_score_kmeans= silhouette_score(pca_values, kmeans_model.labels_)
print('silhouette score: ',sil_score_kmeans)


# In[93]:


data_after_imputation.groupby('Cluster').mean().round(1).reset_index()


# In[94]:


cluster_df =data_after_imputation.copy()
cluster_df


# In[95]:


cluster_1_df = cluster_df[cluster_df["Cluster"]==0]
cluster_1_df


# In[96]:


cluster_2_df = cluster_df[cluster_df["Cluster"]==1]
cluster_2_df


# In[97]:


cluster_3_df = cluster_df[cluster_df["Cluster"]==2]
cluster_3_df


# In[98]:


#Visualization
sns.countplot(x='Cluster', data=cluster_df)


# In[99]:


for c in cluster_df.drop(['Cluster'],axis=1):
    grid= sns.FacetGrid(cluster_df, col='Cluster')
    grid= grid.map(plt.hist, c)
plt.show()


# # Saving the kmeans clustering model and the data with cluster label

# In[122]:


#Saving Scikitlearn models
import joblib
from pickle import load

joblib.dump(kmeans_model, "Model_Kmeans_new")


# In[123]:


cluster_df.to_csv("Clustered_ World_Development_Data.csv")


# # Training and testing the model and the data withcluster label

# In[102]:


X = cluster_df.drop(['Cluster','country'],axis=1)

y= cluster_df['Cluster']
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=5)


# In[103]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[104]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[105]:


clf=SVC()
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[106]:


clf = SVC(kernel= "poly")
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[107]:


clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[108]:


#Confusion_Matrix
from sklearn.metrics import classification_report,confusion_matrix
import sklearn.metrics as metrics
print("Confusion Matrix")
print("________")
print(metrics.confusion_matrix(y_test, y_pred))
print("-----------------------------------------------------------------")
print("Classification Report")
print("________")
print(classification_report(y_test, y_pred))


# In[109]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[110]:


#Decision_Tree
from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=15, max_depth=25, random_state=11)
model3.fit(X_train, y_train)
y_pred = model.fit_predict(X_test)


# In[111]:


from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics
from sklearn.svm import SVC
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# Import the classifier of your choice (e.g., Support Vector Machine)

# Assuming you have your training and test data in 'X_train', 'y_train', 'X_test', and 'y_test', respectively

# Create and train your classifier (SVM as an example)
classifier = SVC()  # You can replace 'SVC()' with the classifier of your choice
classifier.fit(X_train, y_train)

# Use the trained classifier to make predictions on the test data
y_pred = classifier.predict(X_test)

# Generate the confusion matrix
print("Confusion Matrix")
print("________")
print(confusion_matrix(y_test, y_pred))
print("-----------------------------------------------------------------")

# Generate the classification report
print("Classification Report")
print("________")
print(classification_report(y_test, y_pred))
print(f"training accuracy : {rf.score(X_train,y_train )}\ntesting accuracy : {rf.score(X_test, y_test)}")


# In[112]:


model.fit(X_train, y_train)


# # Model Deployment

# In[116]:


import pickle


# In[121]:



with open('Model_Kmeans_new', 'wb') as f:
    pickle.dump('Model_Kmeans_new', f)


# In[120]:


pickle.dump(model,open('Model_Kmeans_new','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




