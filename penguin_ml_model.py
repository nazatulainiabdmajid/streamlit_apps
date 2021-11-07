import pandas as pd


import pandas as pd
import pickle

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



penguin_df=pd.read_csv("penguins.csv")
print(penguin_df.head())
penguin_df.dropna(inplace=True)

output=penguin_df['species']#target
features=penguin_df[['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g', 'sex']]
#how to convert categorical kepada numbers

#original after cleaning NaN Values
print(output.tail())
print(features.tail())

#slps encoding
features=pd.get_dummies(features)
print(features.tail())
#encoding untuk output
output, uniques =pd.factorize (output)

#output, uniques = pd.factorize(output)
# train test split
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.2)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)

y_pred =rfc.predict(x_test)
score = accuracy_score (y_pred, y_test)
#print("Our accuracy score for this model is {}".format(score))
print("Our accuracy score for this model is {}".format(round(score, 2)))

#save the pequin RF classifier
# save the penguin RF classifier
rfc_pickle = open("random_forest_penguin.pickle", 'wb')
pickle.dump(rfc, rfc_pickle)
rfc_pickle.close()

output_pickle = open('output_penguin.pickle','wb') #write bytes
pickle.dump(uniques, output_pickle)
output_pickle.close()



print('Successfull!')






#train test split
