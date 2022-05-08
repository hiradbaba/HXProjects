import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.linear_model import LinearRegression
import math


df_train = pd.read_csv('files/train.csv')
#test
df_test = pd.read_csv('files/testCopy.csv')


# if you want to use error in competition :
def rmsle(preds, dtrain):
	labels = dtrain.get_label()
	assert len(preds) == len(labels)
	labels = labels.tolist()
	preds = preds.tolist()
	terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0, preds[i]) + 1)) ** 2.0 for i, pred in enumerate(labels)]
	return (sum(terms_to_sum) * (1.0 / len(preds))) ** 0.5

def getAccurcy(preds,ytrain):
	acc = []
	for p,y in zip(preds,ytrain):
		acc.append(round(math.log(p+1) - math.log(y+1),4))

	return (np.array(acc), round(sum(acc)/len(acc),8))

def rmsle2(errors):
	s = 0
	for error in errors:
		s+=error**2
	rms = math.sqrt(s/len(errors))
	return round(rms,4)

def getAccurcy2(preds,ytrain):
	summ = 0
	count = 0
	for p,a in zip(preds,ytrain):
		summ+= (a-p)**2
		count += 1
	savg = math.sqrt(summ/count)
	return savg  
    	
def calPriceError():
    valid = pd.read_csv('files/validation.csv')
    predict = valid['Predicted']
    actual = valid['Actual']
    summ = 0
    count = 0
    for p,a in zip(predict,actual):
        summ+= a-p
        count += 1
    print("The avg difference in price is:{}".format(summ//count))
# We take all float/int columns except for ID, timestamp, and the target value
# train_columns = list(
# 	set(df_train.select_dtypes(include=['float64', 'int64']).columns) - set(['id', 'timestamp', 'price_doc']))
#added
features = pd.read_csv('features/features.csv')
train_columns = list(set(features['name'].values))
'''SET #1'''
# df_train = df_train[df_train.additional_education_km <20]
# df_train = df_train[df_train.big_church_km <20]
# df_train = df_train[df_train.hospice_morgue_km <20]
# df_train = df_train[df_train.green_zone_km <20]
'''SET #2'''
#df_train = df_train[df_train.life_sq < df_train.full_sq]
# df_train = df_train[df_train.life_sq < 2000]
# df_train = df_train[df_train.full_sq < 2000]
# df_train = df_train[df_train.floor < 40]
'''SET #3'''
# df_train = df_train[df_train.life_sq < df_train.full_sq]
# df_train = df_train[df_train.life_sq < 2000]
# df_train = df_train[df_train.full_sq < 2000]
# df_train = df_train[df_train.office_km < 15]
# df_train = df_train[df_train.metro_min_avto < 30]
# df_train = df_train[df_train.metro_min_walk < 30]
'''Set #4'''
# df_train = df_train[df_train.life_sq < df_train.full_sq]
# df_train = df_train[df_train.life_sq < 2000]
# df_train = df_train[df_train.full_sq < 2000]
# df_train = df_train[df_train.office_km < 15]
# df_train = df_train[df_train.metro_min_avto < 30]
# df_train = df_train[df_train.metro_min_walk < 30]
# df_train = df_train[df_train.museum_km < 30]
# df_train = df_train[df_train.fitness_km < 15]
# df_train = df_train[df_train.catering_km < 8]
# df_train = df_train[df_train.bus_terminal_avto_km < 40]
'''Set #5'''
# df_train = df_train[df_train.life_sq < df_train.full_sq]
# df_train = df_train[df_train.life_sq < 2000]
# df_train = df_train[df_train.full_sq < 2000]
# df_train = df_train[df_train.office_km < 15]
# df_train = df_train[df_train.metro_min_avto < 30]
# df_train = df_train[df_train.metro_min_walk < 30]
# df_train = df_train[df_train.museum_km < 30]
# df_train = df_train[df_train.fitness_km < 15]
# df_train = df_train[df_train.market_shop_km < 20]
'''Set #6'''
# df_train = df_train[df_train.life_sq < df_train.full_sq]
# df_train = df_train[df_train.life_sq < 2000]
# df_train = df_train[df_train.full_sq < 2000]
# df_train = df_train[df_train.office_km < 15]
# df_train = df_train[df_train.metro_min_avto < 30]
# df_train = df_train[df_train.metro_min_walk < 30]
# df_train = df_train[df_train.museum_km < 30]
# df_train = df_train[df_train.fitness_km < 15]
# df_train = df_train[df_train.market_shop_km < 20]
'''Set #7'''
# df_train = df_train[df_train.life_sq < df_train.full_sq]
# df_train = df_train[df_train.life_sq < 2000]
# df_train = df_train[df_train.full_sq < 2000]
# df_train = df_train[df_train.office_km < 15]
# df_train = df_train[df_train.metro_min_avto < 30]
# df_train = df_train[df_train.metro_min_walk < 30]
# df_train = df_train[df_train.museum_km < 30]
# df_train = df_train[df_train.fitness_km < 15]
# df_train = df_train[df_train.market_shop_km < 20]
# df_train = df_train[df_train.public_transport_station_km < 7.5]
'''Set #8-9'''
# df_train = df_train[df_train.life_sq < df_train.full_sq]
# df_train = df_train[df_train.life_sq < 2000]
# df_train = df_train[df_train.full_sq < 2000]
# df_train = df_train[df_train.office_km < 15]
# df_train = df_train[df_train.metro_min_avto < 30]
# df_train = df_train[df_train.metro_min_walk < 30]
# df_train = df_train[df_train.museum_km < 30]
# df_train = df_train[df_train.fitness_km < 15]
# df_train = df_train[df_train.market_shop_km < 20]
# df_train = df_train[df_train.public_transport_station_km < 7.5]

'''Set #10+'''
# df_train = df_train[df_train.life_sq < df_train.full_sq]
# df_train = df_train[df_train.life_sq < 2000]
# df_train = df_train[df_train.full_sq < 2000]
# df_train = df_train[df_train.office_km < 15]
# df_train = df_train[df_train.metro_min_avto < 30]
# df_train = df_train[df_train.metro_min_walk < 30]
# df_train = df_train[df_train.museum_km < 30]
# df_train = df_train[df_train.fitness_km < 15]
# df_train = df_train[df_train.market_shop_km < 20]
# df_train = df_train[df_train.public_transport_station_km < 7.5]
# df_train = df_train[df_train.workplaces_km< 40]
# df_train = df_train[df_train.detention_facility_km< 60]
'''NEW MODEL'''
# df_train = df_train[df_train.price_doc<8*(10**7)]
'''new sets'''
# df_train = df_train[df_train.life_sq < df_train.full_sq]
# df_train = df_train[df_train.life_sq < 2000]
# df_train = df_train[df_train.full_sq < 2000]
# df_train = df_train[df_train.num_room < 7]
# df_train = df_train[df_train.build_year <2020]
# df_train = df_train[df_train.build_year >1950]
# df_train = df_train[df_train.kitch_sq < 40]
# df_train = df_train[df_train.public_healthcare_km < 30]

# df_train = df_train[df_train.bus_terminal_avto_km <40]

# df_train = df_train[df_train.big_church_km <30]
# df_train = df_train[df_train.church_synagogue_km <10]
'''Another :/'''
df_train = df_train[df_train.price_doc<8*(10**7)]
df_train = df_train[df_train.life_sq < df_train.full_sq]
df_train = df_train[df_train.life_sq < 2000]
df_train = df_train[df_train.full_sq < 2000]
df_train = df_train[df_train.num_room < 7]
df_train = df_train[df_train.build_year <2020]
df_train = df_train[df_train.build_year >1950]
df_train = df_train[df_train.kitch_sq < 40]
df_train = df_train[df_train.public_healthcare_km < 30]
df_train = df_train[df_train.metro_min_avto < 30]
df_train = df_train[df_train.metro_min_walk < 30]
df_train = df_train[df_train.museum_km < 30]
df_train = df_train[df_train.fitness_km < 15]
df_train = df_train[df_train.market_shop_km < 20]
df_train = df_train[df_train.public_transport_station_km < 7.5]
df_train = df_train[df_train.workplaces_km< 40]
df_train = df_train[df_train.detention_facility_km< 60]
df_train = df_train[df_train.additional_education_km <20]
df_train = df_train[df_train.big_church_km <20]
df_train = df_train[df_train.hospice_morgue_km <20]
df_train = df_train[df_train.green_zone_km <20]
df_train = df_train[df_train.catering_km < 8]
df_train = df_train[df_train.floor < 40]
df_train = df_train[df_train.max_floor<40]
df_train = df_train[df_train.public_transport_station_min_walk<100]
df_train = df_train[df_train.state<5]



y_train = df_train['price_doc'].values
x_train = df_train[train_columns].values
#test
x_test = df_test[train_columns].values

# Train/Valid split, note that you can use validation set 
split = int(0.8 * len(df_train))
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

y_train = y_train[~np.isnan(x_train).any(axis=1)]
x_train = x_train[~np.isnan(x_train).any(axis=1)]

y_valid = y_valid[~np.isnan(x_valid).any(axis=1)]
x_validCpy = x_valid
x_valid = x_valid[~np.isnan(x_valid).any(axis=1)]

'''feauture scoring'''
# xgbm = xgb.XGBRegressor(objective='reg:linear',random_state=42)
# xgbm.fit(x_train,y_train)
# xgbm
# quit()
''''''
# if you want not to remove Nan see this link :
clf = LinearRegression()
clf.fit(x_train,y_train)



# fill Nan data in test
from sklearn.preprocessing import Imputer
imputer = Imputer()
x_test_imputed = imputer.fit_transform(x_test)
x_valid_imputed = imputer.fit_transform(x_valid)

p_valid = clf.predict(x_valid_imputed)
p_valid=p_valid.astype(int)
#test
p_test = clf.predict(x_test_imputed)
p_test=p_test.astype(int)
sub = pd.DataFrame()
sub['id'] = df_test['id'].values
sub['price_doc'] = p_test
sub.to_csv('files/result.csv', index=False)

df_valid = df_train[split:]
df_valid = df_valid[~np.isnan(x_validCpy).any(axis=1)]

sub2 = pd.DataFrame()
sub2['id'] = df_valid['id'].values
sub2['Actual'] = y_valid
sub2['Predicted'] = p_valid
result = getAccurcy(p_valid,y_valid)
savg = getAccurcy2(p_valid,y_valid)
sub2['Accurcy'] = result[0]
sub2['Mean_accuarcy'] = np.array(result[1])
sub2['RMSE'] = savg
re = rmsle2(result[0])
sub2['RMSLE'] = re
sub2.to_csv('files/validation.csv',index=False)
#print("the avg rmsle is: {}".format(result[1]))
print("the rmsle is: {}".format(re))