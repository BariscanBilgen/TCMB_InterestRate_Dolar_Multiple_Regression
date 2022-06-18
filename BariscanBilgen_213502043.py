import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

#veri yükle
data = pd.read_csv('MerkezBankasi.CSV', sep = ';' ,encoding='latin-1')


#veri floata çevir
#data["tpmkfbilesik"] = pd.to_numeric(data["tpmkfbilesik"], downcast="float")
#print(data.dtypes)


#Regresyon oluştur
#iki bağımsız değişken var
x = data[['borcverme','tpdkusdaytl']] #gösterdefaiz , dolar kur

#bağımlı değişken
y = data['tpmkfbilesik'] #imkb100

#BİST100 - DOALR - REPO GRAFİĞİ

plt.scatter(x['tpdkusdaytl'],y)
plt.xlabel('DOLAR KURU', fontsize=15)
plt.ylabel('BİST100', fontsize=15)
plt.show()

plt.scatter(x['borcverme'],y)
plt.xlabel('GÖSTERGE REPO', fontsize=15)
plt.ylabel('BİST100', fontsize=15)
plt.show()
#saçılım grafiği
plt.plot(x, label='x labeli')
plt.show()
plt.plot(y, label='y labeli')
plt.show()

#Regresyon
reg = LinearRegression()
print("regreston fit")
print(reg.fit(x,y)) #(önce bağımsız sonra bağımlı)

#sabiti kesim noktasını bul
print("regreston intercept_")
print(reg.intercept_)

#katsayı bul
print("regreston coef_")
print(reg.coef_)

#R kare hesapla
print("regreston score")
print(reg.score(x,y))

#düzeltilmiş r kare
print(x.shape)
r2 = reg.score(x,y)

#gözlem sayısı (n), eksen 0 boyunca olan şekildir
n = x.shape[0]

#özellik sayısı p eksen 1 boyunca şeklidir
p = x.shape[1]

#düzeltilmiş r kare aşağıdaki formülü kullanrak bul
duzeltilmis_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print("regreston duzeltilmis_r2")
print(duzeltilmis_r2)



#TAHMİN YAPMA
print(reg.predict([[10.00,2.2168]]))