#!/usr/bin/env python
# coding: utf-8

# # coffee shop analysis

# An analysis on coffee shop ratings to determine the bet place to find good coffee. 

# ### import city data

# To start we find find US cities to search for coffee shops. Below we loaded a dataset of US cities from https://simplemaps.com/data/us-cities and sorted it by population. 

# In[1]:


# import dataset
from csv import reader
opened_file = open('uscities.csv')
read_file = reader(opened_file)
uscities = list(read_file)
### dataset provided by https://simplemaps.com/data/us-cities


# In[2]:


# view data 
uscities 


# In[3]:


# convert data to data frame
import pandas as pd
uscities = pd.DataFrame(uscities[1:], columns = ['city',
  'city_ascii',
  'state_id',
  'state_name',
  'county_fips',
  'county_name',
  'lat',
  'lng',
  'population',
  'density',
  'source',
  'military',
  'incorporated',
  'timezone',
  'ranking',
  'zips',
  'id',])


# In[4]:


###clean city data
uscities = uscities.drop(["city_ascii", "county_fips", "county_name", "source", "military", "incorporated", "timezone", "ranking", "zips", "id"], axis = 1)


# In[5]:


uscities.head(5)


# In[6]:


### add latitide & Longitude column
uscities["ll"] =  uscities['lat'].str.cat(uscities['lng'],sep=", ")


# In[7]:


### reformat data
uscities = pd.DataFrame(uscities, columns = ['city',
  'state_id',
  'state_name',
  'lat',
  'lng',
  'll',                                              
  'population',
  'density',
   ])


# In[8]:


uscities.shape


# In[9]:


# convert datatypes in uscities
uscities['lat'] = uscities['lat'].astype(float)
uscities['lng'] = uscities['lng'].astype(float)
uscities['ll'] = uscities['ll'].astype(str)
uscities['population'] = uscities['population'].astype(float)
uscities['density'] = uscities['density'].astype(float)
print(uscities.dtypes)


# In[28]:


# view final city dataset
uscities


# Once we have an established set of US Cities we loop throught these cities to find coffee shops and build a coffee shop dataset

# In[29]:


# load programs
import json, requests 
from pandas import json_normalize


# In[12]:


# create function to generate coffee shops
def coffee_find(lat_lng):
    url = 'https://api.foursquare.com/v2/venues/explore'
    params = dict(
        client_id='3Q530RJEKQDDEZHANP4BXSEK5BA2JJ1WFTA4L1IR0YNT1I1Q',
        client_secret='TRBK4GRSWRTDLW35X3IN2S2AFO5HHNJCDNARRFEZRLICGRKQ',
        v='20210115',
        ll= lat_lng,
        radius = '1700',
        query= 'coffee',
        limit= '1500')
        
    results = requests.get(url=url, params=params).json()

    venues = results['response']['groups'][0]['items']
    nearby_venues = json_normalize(venues)
    nearby_venues  = pd.DataFrame(nearby_venues, columns = ['venue.id',
        'venue.name','venue.location.address', 'venue.location.lat', 'venue.location.lng'])
    return nearby_venues


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


### find coffee shops for 10 most populated cities

coffee_list = coffee_find(uscities.loc[0]['ll'])
coffee_list = coffee_list.append(coffee_find(uscities.loc[1]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[2]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[3]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[4]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[5]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[6]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[7]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[8]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[9]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[10]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[11]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[12]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[13]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[14]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[15]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[16]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[17]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[18]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[19]['ll']))
coffee_list = coffee_list.append(coffee_find(uscities.loc[20]['ll']))

coffee_list = coffee_list.reset_index()
coffee_list = coffee_list.rename(columns={"venue.id": 
                                          "id", "venue.name": "coffee_shop", 
                                          "venue.location.address":"coffee_shop_address", 
                                          "venue.location.lat":"cs_lat", 
                                          "venue.location.lng":"cs_lng"})
coffee_list = coffee_list.drop(columns="index")


# In[40]:


coffee_list = coffee_list[0:500]
coffee_list


# In[51]:


def coffee_details(venue_id):
    CLIENT_ID='3Q530RJEKQDDEZHANP4BXSEK5BA2JJ1WFTA4L1IR0YNT1I1Q'
    CLIENT_SECRET='TRBK4GRSWRTDLW35X3IN2S2AFO5HHNJCDNARRFEZRLICGRKQ'
    ACCESS_TOKEN = ''
    v='20210115'
    venue_id = venue_id 
    url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&oauth_token={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, v)
    cafe_result = requests.get(url).json()
    cafe_result['response']
    coffee_venues = cafe_result
    return coffee_venues


# In[59]:


####convert data frame to list of lists
coffee_list_py = coffee_list.values.tolist()
coffee_list_py[0:1]


# In[65]:


coffee_venues = []
for row in coffee_list_py[0:499]:
    c_id = row[0]
    coffee_shop = coffee_details(c_id)
    coffee_venues.append(coffee_shop)


# In[66]:


coffee_venues


# In[67]:


coffee_venues_test = coffee_venues
coffee_venues_test = json_normalize(coffee_venues_test)


# In[68]:


coffee_venues_test


# In[69]:


coffee_venues_test = pd.DataFrame(coffee_venues_test, columns = ['response.venue.id','response.venue.name','response.venue.price.tier','response.venue.rating','response.venue.likes.count'])
coffee_venues_test = coffee_venues_test.rename(columns={'response.venue.id':'id', 'response.venue.name':'name', 'response.venue.price.tier':'price_tier', 'response.venue.rating':'rating', 'response.venue.likes.count':'likes_count'})


# In[70]:


coffee_venues_test


# In[71]:


coffee_shop_list = coffee_list.merge(coffee_venues_test, how= 'outer', on= 'id')
coffee_shop_list = coffee_shop_list.drop(columns='name')
coffee_shop_list


# In[72]:


u_rating = coffee_shop_list.rating.unique()
print(u_rating)
u_rating_counts = coffee_shop_list.rating.value_counts(dropna=False)
u_rating_counts


# In[73]:


coffee_shop_list = coffee_shop_list.dropna()
coffee_shop_list = coffee_shop_list.reset_index(drop=True)
coffee_shop_list


# In[ ]:





# In[74]:


import random 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs


# In[75]:


Coffee_cluster_set = coffee_shop_list.drop('coffee_shop_address', axis=1)
Coffee_cluster_set = Coffee_cluster_set.drop('id', axis=1)
Coffee_cluster_set = Coffee_cluster_set.drop('coffee_shop', axis=1)
Coffee_cluster_set.head()


# In[76]:


#Scale price_tier, rating, likes_count
from sklearn.preprocessing import StandardScaler
X = Coffee_cluster_set.values[:,2:]
X = np.nan_to_num(X)

flat_Coffee_cluster_set = StandardScaler().fit_transform(X)
flat_Coffee_cluster_set


# In[77]:


X


# In[78]:


#plot rating, likes_count
plt.scatter(X[:,1],X[:,2])
plt.xlabel("Ratings")
plt.ylabel("Likes Counts")
plt.title("Likes & Ratings", fontdict=None, loc='center')


# In[80]:


#plot rating, likes_count
plt.bar(X[:,0],X[:,1])
plt.xlabel("Price Tier")
plt.ylabel("Rating")
plt.title("Price Tier & Ratings", fontdict=None, loc='center')


# In[213]:


#plot rating, likes_count
plt. bar(X[:,0],X[:,2])
plt.xlabel("Price Tier")
plt.ylabel("Like Counts")
plt.title("Price Tier & Like Counts", fontdict=None, loc='center')


# In[211]:


num_clusters = 4

k_means = KMeans(init="k-means++", n_clusters=num_clusters, n_init=12)
k_means.fit(flat_Coffee_cluster_set)
labels = k_means.labels_

print(labels)


# In[214]:


Coffee_cluster_set["labels"] = labels
Coffee_cluster_set.head(5)


# In[215]:


k_means_labels = Coffee_cluster_set.groupby('labels').mean()


# In[ ]:





# In[216]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.set_xlabel('Price Range')
ax.set_ylabel('Ratings')
ax.set_zlabel('Facebook Like')

ax.scatter(X[:, 0], X[:, 1], X[:, 2])


# In[217]:


print(X) # Price_tier, Ratings, Like Counts


# In[218]:


# plot of price tier, rqtings
plt.figure(figsize=(15, 10))
plt.scatter(X[:, 1], X[:, 2], marker='.')
plt.xlabel("price tier")
plt.ylabel("ratings")


# In[219]:


k_means.fit(X)


# In[220]:


k_means_labels = k_means.labels_
k_means_labels
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


# In[221]:


# initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(15, 10))

# colors uses a color map, which will produce an array of colors based on
# the number of labels. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# create a plot
ax = fig.add_subplot(1, 1, 1)

# loop through the data and plot the datapoints and centroids.
# k will range from 0-3, which will match the number of clusters in the dataset.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # create a list of all datapoints, where the datapoitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # plot the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # plot the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# title of the plot
ax.set_title('KMeans')

# remove x-axis ticks
ax.set_xticks(())

# remove y-axis ticks
ax.set_yticks(())

# show the plot
plt.show()


# In[222]:


coffee_shop_list["labels"] = labels
coffee_shop_list.head()


# In[223]:


l_unique = coffee_shop_list.labels.unique()
l_unique


# In[224]:


label_0 = coffee_shop_list[coffee_shop_list['labels'] == 0]
label_1 = coffee_shop_list[coffee_shop_list['labels'] == 1]
label_2 = coffee_shop_list[coffee_shop_list['labels'] == 2]
label_3 = coffee_shop_list[coffee_shop_list['labels'] == 3]


# In[225]:


get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium


# In[226]:


# create map and display it
map = folium.Map(location=[40, -74], zoom_start=7)

# display the map
map


# In[227]:


# instantiate a feature group for the incidents in the dataframe
coffee_markers0 = folium.map.FeatureGroup()
coffee_markers1 = folium.map.FeatureGroup()
coffee_markers2 = folium.map.FeatureGroup()
coffee_markers3 = folium.map.FeatureGroup()

# label 0 
for lat, lng, in zip(label_0.cs_lat, label_0.cs_lng):
    coffee_markers0.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius= 1, # define how big you want the circle markers to be
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=.5
        )
    )
    
# label 1 
for lat, lng, in zip(label_1.cs_lat, label_1.cs_lng):
    coffee_markers1.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius= 1, # define how big you want the circle markers to be
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=.5
        )
    )

# label 2 
for lat, lng, in zip(label_2.cs_lat, label_2.cs_lng):
    coffee_markers2.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius= 1, # define how big you want the circle markers to be
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=.5
        )
    )
    
# label 3 
for lat, lng, in zip(label_3.cs_lat, label_3.cs_lng):
    coffee_markers3.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius= 1, # define how big you want the circle markers to be
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=.5
        )
    )

# add incidents to map
map_w_markers = map.add_child(coffee_markers0)
map_w_markers = map.add_child(coffee_markers1)
map_w_markers = map.add_child(coffee_markers2)
map_w_markers = map.add_child(coffee_markers3)
map_w_markers


# In[ ]:





# In[228]:


X0 = label_0.drop(['coffee_shop_address','id','coffee_shop','cs_lat','cs_lng','labels'], axis=1)
X1 = label_1.drop(['coffee_shop_address','id','coffee_shop','cs_lat','cs_lng','labels'], axis=1)
X2 = label_2.drop(['coffee_shop_address','id','coffee_shop','cs_lat','cs_lng','labels'], axis=1)
X3 = label_3.drop(['coffee_shop_address','id','coffee_shop','cs_lat','cs_lng','labels'], axis=1)


# In[229]:


X0 = X0.values[:,:]
X0 = np.nan_to_num(X0)
X0 = StandardScaler().fit_transform(X0)

X1 = X1.values[:,:]
X1 = np.nan_to_num(X1)
X1 = StandardScaler().fit_transform(X1)

X2 = X2.values[:,:]
X2 = np.nan_to_num(X2)
X2 = StandardScaler().fit_transform(X2)

X3 = X3.values[:,:]
X3 = np.nan_to_num(X3)
X3 = StandardScaler().fit_transform(X3)



# In[231]:


fig = plt.figure(figsize = (16,8))
AX0 = fig.add_subplot(2,2,1)
AX1 = fig.add_subplot(2,2,2)
AX2 = fig.add_subplot(2,2,3)
AX3 = fig.add_subplot(2,2,4)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.3)


AX0.scatter(X0[:,1],X0[:,2])
AX0.set_xticks(np.arange(-3, 4, step=1))
AX0.set_yticks(np.arange(-1, 6, step=1))
AX0.set_xlabel("Ratings")
AX0.set_ylabel("Likes Counts")
AX0.set_title("Label 0", fontdict=None, loc='left')

AX1.scatter(X1[:,1],X1[:,2])
AX1.set_xticks(np.arange(-3, 4, step=1))
AX1.set_yticks(np.arange(-1, 6, step=1))
AX1.set_xlabel("Ratings")
AX1.set_ylabel("Likes Counts")
AX1.set_title("Label 1", fontdict=None, loc='left')

AX2.scatter(X2[:,1],X2[:,2])
AX2.set_xticks(np.arange(-3, 4, step=1))
AX2.set_yticks(np.arange(-1, 6, step=1))
AX2.set_xlabel("Ratings")
AX2.set_ylabel("Likes Counts")
AX2.set_title("Label 2", fontdict=None, loc='left')

AX3.scatter(X3[:,1],X3[:,2])
AX3.set_xticks(np.arange(-3, 4, step=1))
AX3.set_yticks(np.arange(-1, 6, step=1))
AX3.set_xlabel("Ratings")
AX3.set_ylabel("Likes Counts")
AX3.set_title("Label 3", fontdict=None, loc='left')

plt.show()


# In[232]:


fig = plt.figure(figsize = (16,8))
AX0 = fig.add_subplot(2,2,1)
AX1 = fig.add_subplot(2,2,2)
AX2 = fig.add_subplot(2,2,3)
AX3 = fig.add_subplot(2,2,4)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.3)


AX0.scatter(X0[:,0],X0[:,2])
AX0.set_xticks(np.arange(-1, 5, step=1))
AX0.set_yticks(np.arange(-1, 5, step=1))
AX0.set_xlabel("Price Tier")
AX0.set_ylabel("Likes Counts")
AX0.set_title("Label 0", fontdict=None, loc='left')

AX1.scatter(X1[:,0],X1[:,2])
AX1.set_xticks(np.arange(-1, 5, step=1))
AX1.set_yticks(np.arange(-1, 5, step=1))
AX1.set_xlabel("Price Tier")
AX1.set_ylabel("Likes Counts")
AX1.set_title("Label 1", fontdict=None, loc='left')

AX2.scatter(X2[:,0],X2[:,2])
AX2.set_xticks(np.arange(-1, 5, step=1))
AX2.set_yticks(np.arange(-1, 5, step=1))
AX2.set_xlabel("Price Tier")
AX2.set_ylabel("Likes Counts")
AX2.set_title("Label 2", fontdict=None, loc='left')

AX3.scatter(X3[:,0],X3[:,2])
AX3.set_xticks(np.arange(-1, 5, step=1))
AX3.set_yticks(np.arange(-1, 5, step=1))
AX3.set_xlabel("Price Tier")
AX3.set_ylabel("Likes Counts")
AX3.set_title("Label 3", fontdict=None, loc='left')

plt.show()


# In[233]:


fig = plt.figure(figsize = (16,8))
AX0 = fig.add_subplot(2,2,1)
AX1 = fig.add_subplot(2,2,2)
AX2 = fig.add_subplot(2,2,3)
AX3 = fig.add_subplot(2,2,4)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.3)


AX0.scatter(X0[:,0],X0[:,1])
AX0.set_xticks(np.arange(-1, 5, step=1))
AX0.set_yticks(np.arange(-1, 4, step=1))
AX0.set_xlabel("Price Tier")
AX0.set_ylabel("Ratings")
AX0.set_title("Label 0", fontdict=None, loc='left')

AX1.scatter(X1[:,0],X1[:,1])
AX1.set_xticks(np.arange(-1, 5, step=1))
AX1.set_yticks(np.arange(-1, 4, step=1))
AX1.set_xlabel("Price Tier")
AX1.set_ylabel("Ratings")
AX1.set_title("Label 1", fontdict=None, loc='left')

AX2.scatter(X2[:,0],X2[:,1])
AX2.set_xticks(np.arange(-1, 5, step=1))
AX2.set_yticks(np.arange(-1, 4, step=1))
AX2.set_xlabel("Price Tier")
AX2.set_ylabel("Ratings")
AX2.set_title("Label 2", fontdict=None, loc='left')

AX3.scatter(X3[:,0],X3[:,1])
AX3.set_xticks(np.arange(-1, 5, step=1))
AX3.set_yticks(np.arange(-1, 4, step=1))
AX3.set_xlabel("Price Tier")
AX3.set_ylabel("Ratings")
AX3.set_title("Label 3", fontdict=None, loc='left')

plt.show()


# In[ ]:



LAbel 0 is 


LAbel 2 is medium price, low like counts


# In[234]:


label_1


# LABEL 1 has the bet coffee by like counts and user ratings. The majority of coffee shops in this cluster are in New York. Newy York has the best coffee. 
# 

# In[ ]:




