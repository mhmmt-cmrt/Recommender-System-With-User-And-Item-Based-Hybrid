
#*********************************************************
#Verinin düzenlenmesi ve model için hazırlanması
#*********************************************************

import pandas as pd
import numpy as np
from holoviews.operation import collapse

pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 500)

filmler = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/movies.csv")
filmler.head()
filmler.shape

puanlar = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/ratings.csv")
puanlar.head()
puanlar.shape
puanlar["userId"].nunique()

#************************#
#veri seti bilgilerini yazar
######################################3333
egitim = puanlar[['userId', 'movieId', 'rating']]
print('>> {} tane kullanicidan {} satir verisi vardir'.format(len(egitim.userId.unique()),len(puanlar)))
print('>> toplam yapilan puanlama sayisi: ',egitim.shape[0],' dir')
print('>> sahip oldugumuz benzersiz kullanici sayisi:', len(egitim.userId.unique()))
print('>> sahip oldugumuz benzersiz film sayisi:', len(egitim.movieId.unique()))
print('>> maksimum oy degeri: %d'%egitim.rating.max())
print('>> minumum oy degeri: %d' %egitim.rating.min())
######################################333

birlesikTablo = filmler.merge(puanlar, how="left", on="movieId")
birlesikTablo.head()
birlesikTablo.shape

# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
# kalabalığın bilgeliği kavramı üzerine en çok değerlenen filmleri alırız.
# çünkü ne kadar çok yorum varsa değer o kadar genelleştirilebilir.

degerlendirme_sayisi = pd.DataFrame(birlesikTablo["title"].value_counts())
degerlendirme_sayisi.shape

yetersiz_degerlenmis = degerlendirme_sayisi[degerlendirme_sayisi["title"] <= 5]
yetersiz_degerlenmis.shape
yeterli_degerlenmis = birlesikTablo[~birlesikTablo["title"].isin(yetersiz_degerlenmis)]
yeterli_degerlenmis.shape

#biased için(bilgi amaçlı kısım)
#####################################################################
oylama_sayilari = pd.DataFrame(birlesikTablo.groupby('title')['rating'].mean())
oylama_sayilari['toplam oylama'] = pd.DataFrame(birlesikTablo.groupby('title')['rating'].count())
oylama_sayilari.rename(columns={'rating': 'oy ortalamasi'}, inplace=True)

oylama_sayilari.sort_values('toplam oylama', ascending=False).head(50)

#100 den fazla oylanan filmleri inceleyelim
oylama_sayilari_top = oylama_sayilari[oylama_sayilari['toplam oylama'] >= 100]
oylama_sayilari_top = oylama_sayilari_top.sort_values('toplam oylama', ascending=False)
pd.DataFrame(oylama_sayilari_top).count()

#100 den az oylanan filmleri inceleyelim
oylama_sayilari_down = oylama_sayilari[oylama_sayilari['toplam oylama'] < 50]
oylama_sayilari_down = oylama_sayilari_down.sort_values('toplam oylama', ascending=False)
pd.DataFrame(oylama_sayilari_down).count()
#######################################################################33

#bias-1 için az popüler olanları inceleyelim
yetersiz_degerlilerin_isimleri = degerlendirme_sayisi[degerlendirme_sayisi["title"] <= 5]
yetersiz_degerlenenen_degerlerin_sayisi = degerlendirme_sayisi[degerlendirme_sayisi["title"] <= 5].value_counts()
yetersiz_degerlenenen_degerlerin_sayisi.sum()
#burada örneğin 5 puan verilen fakat değerlendirme genel sayısı az olduğu için yetersiz grubuna düşen 383 adet film var

populer_olmayan_onerilebilecekler = yetersiz_degerlilerin_isimleri[yetersiz_degerlilerin_isimleri["title"] == 5]

#yetersiz olanlardan bir df oluşturuyoruz.
yeterli = degerlendirme_sayisi[degerlendirme_sayisi["title"] > 5]
yetersiz = birlesikTablo[~birlesikTablo["title"].isin(yeterli)]
kullaniciFildDf_WithYetersiz = yetersiz.pivot_table(index=["userId"], columns=["title"], values="rating")
#####################################################################################333

#bias-2 için popülerite katsayılarını bulalım.
popularite_tablosu = pd.DataFrame(birlesikTablo["title"].value_counts())
popularite_tablosu["popularite"] = pd.DataFrame(birlesikTablo["title"].value_counts())
popularite_tablosu = popularite_tablosu.drop(columns='title')

def normalize_populerlik(popularite_tablosu):
    min_deger = popularite_tablosu["popularite"].min()
    max_deger = popularite_tablosu["popularite"].max()
    for i in popularite_tablosu["popularite"]:
        popularite_tablosu["normalized_pop"] = ((popularite_tablosu["popularite"] - min_deger) / (max_deger - min_deger))

    return popularite_tablosu

normalized_popularite_tablosu = normalize_populerlik(popularite_tablosu)
normalized_popularite_tablosu.reset_index(inplace=True)
normalized_popularite_tablosu.rename(columns={'index': 'title'}, inplace=True)
normalized_popularite_tablosu = normalized_popularite_tablosu.merge(filmler, how="left", on="title")
normalized_popularite_tablosu = pd.DataFrame(normalized_popularite_tablosu)
normalized_popularite_tablosu.drop(columns='genres', inplace=True)
#************************#
#########################################################################################33

kullaniciFilmDf = yeterli_degerlenmis.pivot_table(index=["userId"], columns=["title"], values="rating")
kullaniciFilmDf.head()

# buraya kadarki tüm işlemleri fonksiyonlaştıralım
"""
def kullanici_film_df_olustur():
    import pandas as pd
    filmler = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/movies.csv")
    puanlar = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/ratings.csv")
    birlesikTablo = filmler.merge(puanlar, how="left", on="movieId")
    degerlendirme_sayisi = pd.DataFrame(birlesikTablo["title"].value_counts())
    yetersiz_degerlenmis = degerlendirme_sayisi[degerlendirme_sayisi["title"] <= 5].index
    yeterli_degerlenmis = birlesikTablo[~birlesikTablo["title"].isin(yetersiz_degerlenmis)]
    popularite_tablosu = pd.DataFrame(birlesikTablo["title"].value_counts())
    popularite_tablosu["popularite"] = pd.DataFrame(birlesikTablo["title"].value_counts())
    popularite_tablosu = popularite_tablosu.drop(columns='title')
    normalized_popularite_tablosu = normalize_populerlik(popularite_tablosu)
    normalized_popularite_tablosu.reset_index(inplace=True)
    normalized_popularite_tablosu.rename(columns={'index': 'title'}, inplace=True)
    normalized_popularite_tablosu = normalized_popularite_tablosu.merge(filmler, how="left", on="title")
    normalized_popularite_tablosu = pd.DataFrame(normalized_popularite_tablosu)
    normalized_popularite_tablosu.drop(columns='genres', inplace=True)
    kullaniciFilmDf = yeterli_degerlenmis.pivot_table(index=["userId"], columns=["title"], values="rating")
    return kullaniciFilmDf
kullaniciFilmDf = kullanici_film_df_olustur()
"""
#************************************************************************
#tavsiye Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#************************************************************************
rasgele_kullanici = 25

rasgele_kullanici_df = kullaniciFilmDf[kullaniciFilmDf.index == rasgele_kullanici]
rasgele_kullanici_df.head()
rasgele_kullanici_df.shape

izlenenFilmler = rasgele_kullanici_df.columns[rasgele_kullanici_df.notna().any()].to_list()
len(izlenenFilmler)
filmler.columns[filmler.notna().any()].to_list()

#**********************************************************************************************
#Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerini buluyoruz
#************************************************************************************************

izlenenFilmler_df = kullaniciFilmDf[izlenenFilmler]
izlenenFilmler_df.head()
izlenenFilmler_df.shape
izlenenFilmler_df.columns.value_counts().sum()
# üsteki koda göre-> sutunlarda ki benzersiz film sayisi = rasgele kullanıcının izlediği film sayisi

kullanici_film_sayisi = izlenenFilmler_df.T.notnull().sum()
kullanici_film_sayisi = kullanici_film_sayisi.reset_index()
kullanici_film_sayisi.columns = ["userId", "film_sayisi"]
kullanici_film_sayisi.head(5)
kullanici_film_sayisi.shape
max(kullanici_film_sayisi["film_sayisi"])

yuzde = len(izlenenFilmler) * 80 / 100

ortak_film_izleyenler = kullanici_film_sayisi[kullanici_film_sayisi["film_sayisi"] > yuzde]["userId"]
len(ortak_film_izleyenler)

#*******************************************************************************
# Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#*******************************************************************************

guncelVeri = izlenenFilmler_df[izlenenFilmler_df.index.isin(ortak_film_izleyenler)]
guncelVeri.head()
guncelVeri.shape

# Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir korelasyonVerisi oluşturruldu
korelasyonVerisi = guncelVeri.T.corr().unstack().sort_values()
korelasyonVerisi = pd.DataFrame(korelasyonVerisi, columns=["korelasyon"])
korelasyonVerisi.index.names = ['Kullanici_id_1', 'Kullanici_id_2']
korelasyonVerisi = korelasyonVerisi.reset_index()


en_iyi_kullanicilar = korelasyonVerisi[(korelasyonVerisi["Kullanici_id_1"] == rasgele_kullanici) & (korelasyonVerisi["korelasyon"] >= 0.60)][
    ["Kullanici_id_2", "korelasyon"]].reset_index(drop=True)

en_iyi_kullanicilar = en_iyi_kullanicilar.sort_values(by='korelasyon', ascending=False)
en_iyi_kullanicilar.rename(columns={"Kullanici_id_2": "userId"}, inplace=True)

en_iyi_kullanicilar.shape
en_iyi_kullanicilar.head()

puanlar = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/ratings.csv")
topnKullaniciPuan = en_iyi_kullanicilar.merge(puanlar[["userId", "movieId", "rating"]], how='inner')

topnKullaniciPuan = normalized_popularite_tablosu.merge(topnKullaniciPuan[["userId", "movieId", "korelasyon", "rating"]], how='inner')
########################################################################################3333

topnKullaniciPuan = topnKullaniciPuan[topnKullaniciPuan["userId"] != rasgele_kullanici]
topnKullaniciPuan["userId"].unique()
topnKullaniciPuan.head()

#############################################
# Ağırlıklı ortalama Score'un Hesabı
#############################################

topnKullaniciPuan['weighted_rating'] = topnKullaniciPuan['korelasyon'] * topnKullaniciPuan['rating']
bias = 0.5
topnKullaniciPuan['normalized_weighted_rating'] = ((topnKullaniciPuan['weighted_rating']) - ((topnKullaniciPuan['normalized_pop']) * bias))

topnKullaniciPuan.head()

#########################################################################################################################333
tavsiyeEt = topnKullaniciPuan[topnKullaniciPuan["normalized_weighted_rating"] > 3].sort_values("korelasyon", ascending=False)
#type(tavsiyeEt)

def hata_k_o_k(tahminler):
    h_k_Ort = np.mean(
        [float((x - y) ** 2) for x, y in zip(tahminler['rating'], tahminler['weighted_rating'])]
    )
    h_k_o_Karekok = np.sqrt(h_k_Ort)

    return h_k_o_Karekok

a = hata_k_o_k(tavsiyeEt)
print("hata oranı : ", a)
############################################################################################################################

tavsiyeler = topnKullaniciPuan.groupby('movieId').agg({"normalized_weighted_rating": "mean"})
tavsiyeler = tavsiyeler.reset_index()
tavsiyeler.head()

tavsiyeler[tavsiyeler["normalized_weighted_rating"] > 3]
tavsiyeEdilecekler = tavsiyeler[tavsiyeler["normalized_weighted_rating"] > 3].sort_values("normalized_weighted_rating", ascending=False)

filmler = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/movies.csv")
tavsiyeEdilecekler.merge(filmler[["movieId", "title"]])["title"][:5]

#############################################
#Item-Based Recommendation
#############################################
kullanici = 25

filmler = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/movies.csv")
puanlar = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/ratings.csv")

#kullanıcının 5 puan verdiği filmlerden puanı en güncel olan film alınır.
film_id = puanlar[(puanlar["userId"] == kullanici) & (puanlar["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

#kullanıcı tabanlı bölümde oluşturulan KullaniciFilmDf dataframe’ini seçilen film id’sine göre filtreleyiniz.
filmler[filmler["movieId"] == film_id]["title"].values[0]
film_df = kullaniciFilmDf[filmler[filmler["movieId"] == film_id]["title"].values[0]]
film_df.shape
# Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunur
item_based_öneriler = kullaniciFilmDf.corrwith(film_df).sort_values(ascending=False)
item_based_öneriler[1:6]






