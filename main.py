import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def modelo(w):
  # read data
  main_file_path = 'train.csv' 
  data = pd.read_csv(main_file_path)
  # target data
  sPrice = data.SalePrice
  y = sPrice
  # features
  pred = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
  X = data[pred]
  # Machine learing Model
  my_model = DecisionTreeRegressor()
  # Train this model 
  my_model.fit(X, y)
  # predict
  print("The predictions are:")
  return my_model.predict(w)


def main():
  
  l = []
  
  for i in range(7):
    if i == 0:
      l.append(float(input("Enter Size in Square Feet:")))
    elif i == 1:
      l.append(float(input("What Year Constructed in:")))
    elif i == 2:
      l.append(float(input("How Much First Floor Size:")))
    elif i == 3:
      l.append(float(input("How Much Second Floor Size:")))
    elif i == 4:
      l.append(float(input("No. of Bathrooms of House:")))
    elif i == 5:
      l.append(float(input("No. of Bedrooms of House:")))
    elif i == 6:
      l.append(float(input("NO. of Rooms of House:")))
  
  print(modelo(l))

if __name__ == '__main__':
  main()