import pandas as pd
import numpy as np
# features = pd.read_csv('features/features.csv')
# print(features['name'])

df = pd.read_csv('files/trainCopy.csv')
dictionary = pd.DataFrame()

def giveValueToSubArea():
    value = 1
    areas = []
    areaDict = {}
    names = df['sub_area']
    for name in names:
        if name not in areaDict:
            areaDict[name] = value
            areas.append(value)
            value+=1
        else:
            areas.append(areaDict[name])
    return areaDict,areas

def arrangeSubArea(areaDict,ar):
    value = 146
    prices = df['price_doc'][:]
    names = df['sub_area'][:]
    pricesSorted,namesSorted = (list(t) for t in zip(*sorted(zip(prices,names))))
    observed={}
    
    for i in range(len(namesSorted)-1,-1,-1):
        if namesSorted[i] in observed or pricesSorted[i]>80000000:
            continue
        else:
            areaDict[namesSorted[i]] = value
            value-=1
            if value == 0:
                break
        
    ar=[]
    for name in names:
        ar.append(areaDict[name])
    
    return areaDict,ar


ad,ar = giveValueToSubArea()
ad2,ar2 = arrangeSubArea(ad,ar)
areas=[]
vals=[]
for key in ad2:
    areas.append(key)
    vals.append(ad2[key])
dictionary['area'] = np.array(areas)
dictionary['value'] = np.array(vals)
dictionary.to_csv('features/area_val.csv',index=False)
df['sub_area_2'] = np.array(ar2)
df.to_csv('files/trainCopy.csv',index=False)


