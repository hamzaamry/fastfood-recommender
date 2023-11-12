import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#loading datasets 

food = pd.read_csv('../Datasets/food.csv')
ratings = pd.read_csv('../Datasets/ratings.csv')

'''

print(' Let us help you with ordering ')

vegan = input('Vegetables or none!\n')
cuisine = input('choose ur favorite cuisine\n')
val = int(input("How well do you want the dish to be?\n"))
 
combined = pd.merge(ratings, food, on='Food_ID')

# filtering user preferences

ans = combined.loc[(combined.C_Type == cuisine) & (combined.Veg_Non == vegan) & (combined.Rating >= val),['Name','C_Type','Veg_Non']]
names = ans['Name'].tolist()
x = np.array(names)


filtred_user_pref = np.unique(x)
'''

##### IMPLEMENTING RECOMMENDER ######

dataset = ratings.pivot_table(index='Food_ID',columns='User_ID',values='Rating')


dataset.fillna(0,inplace=True)

csr_dataset = csr_matrix(dataset.values)

dataset.reset_index(inplace=True)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

def food_recommendation(Food_Name):
    n = 10
    FoodList = food[food['Name'].str.contains(Food_Name)] 

    if len(FoodList):        
        Foodi= FoodList.iloc[0]['Food_ID']

        
        Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
        distances , indices = model.kneighbors(csr_dataset[Foodi],n_neighbors=n+1)  

        Food_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]

        Recommendations = []

        for val in Food_indices:
            Foodi = dataset.iloc[val[0]]['Food_ID']
            i = food[food['Food_ID'] == Foodi].index
            Recommendations.append({'Name':food.iloc[i]['Name'].values[0],'Distance':val[1]})

        df = pd.DataFrame(Recommendations,index=range(1,n+1))
        return df['Name']
    else:
        return "No Similar Foods."
    

if __name__ == "__main__":

    food_name_to_test = input('test --- give a dish :')
    result = food_recommendation(food_name_to_test)

    print(f"Recommended foods for {food_name_to_test}:")
    print(result)