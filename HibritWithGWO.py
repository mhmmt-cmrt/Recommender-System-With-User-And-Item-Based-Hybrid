#coding=utf-8

import numpy as np
import pandas as pd
from GriKurtOptimizasyon import Gri_kurt_opt
from GugukKusuOptimizasyon import guguk_kusu_opt
from gerekli_fonksiyonlar import hata_k_o_k
from gerekli_fonksiyonlar import normalize_populerlik
import time

pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 500)

filmler = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/movies.csv")
puanlar = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/ratings.csv")
birlesikTablo = filmler.merge(puanlar, how="left", on="movieId")
degerlendirme_sayisi = pd.DataFrame(birlesikTablo["title"].value_counts())
yetersiz_degerlenmis = degerlendirme_sayisi[degerlendirme_sayisi["title"] <= 5]
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



def optimize_ratio(oran):
    rasgele_kullanici = 25
    rasgele_kullanici_df = kullaniciFilmDf[kullaniciFilmDf.index == rasgele_kullanici]
    izlenenFilmler = rasgele_kullanici_df.columns[rasgele_kullanici_df.notna().any()].to_list()
    filmler.columns[filmler.notna().any()].to_list()
    izlenenFilmler_df = kullaniciFilmDf[izlenenFilmler]
    izlenenFilmler_df.columns.value_counts().sum()
    kullanici_film_sayisi = izlenenFilmler_df.T.notnull().sum()
    kullanici_film_sayisi = kullanici_film_sayisi.reset_index()
    kullanici_film_sayisi.columns = ["userId", "film_sayisi"]
    kullanici_film_sayisi.sort_values(by='film_sayisi', ascending=False)
    yuzde = len(izlenenFilmler) * oran / 100
    ortak_film_izleyenler = kullanici_film_sayisi[kullanici_film_sayisi["film_sayisi"] > yuzde]["userId"]
    ortak_film_izlemeyenler = kullanici_film_sayisi[~kullanici_film_sayisi["userId"].isin(ortak_film_izleyenler)]
    guncelVeri = izlenenFilmler_df[izlenenFilmler_df.index.isin(ortak_film_izleyenler)]
    korelasyonVerisi = guncelVeri.T.corr().unstack().sort_values()
    korelasyonVerisi = pd.DataFrame(korelasyonVerisi, columns=["korelasyon"])
    korelasyonVerisi.index.names = ['Kullanici_id_1', 'Kullanici_id_2']
    korelasyonVerisi = korelasyonVerisi.reset_index()
    en_iyi_kullanicilar = korelasyonVerisi[
        (korelasyonVerisi["Kullanici_id_1"] == rasgele_kullanici) & (korelasyonVerisi["korelasyon"] >= 0.60)][
        ["Kullanici_id_2", "korelasyon"]].reset_index(drop=True)

    en_iyi_kullanicilar = en_iyi_kullanicilar.sort_values(by='korelasyon', ascending=False)
    en_iyi_kullanicilar.rename(columns={"Kullanici_id_2": "userId"}, inplace=True)
    puanlar = pd.read_csv("C:/Users/Muhammet/PycharmProjects/HybridModel/datasetSmall/ratings.csv")
    topnKullaniciPuan = en_iyi_kullanicilar.merge(puanlar[["userId", "movieId", "rating"]], how='inner')

    topnKullaniciPuan = normalized_popularite_tablosu.merge(
        topnKullaniciPuan[["userId", "movieId", "korelasyon", "rating"]], how='inner')

    topnKullaniciPuan = topnKullaniciPuan[topnKullaniciPuan["userId"] != rasgele_kullanici]
    topnKullaniciPuan["userId"].unique()
    topnKullaniciPuan.head()
    topnKullaniciPuan['weighted_rating'] = topnKullaniciPuan['korelasyon'] * topnKullaniciPuan['rating']
    bias = 0.5
    topnKullaniciPuan['normalized_weighted_rating'] = ((topnKullaniciPuan['weighted_rating']) - (topnKullaniciPuan['normalized_pop'] * bias))

    topnKullaniciPuan.head()
    tavsiyeEt = topnKullaniciPuan[topnKullaniciPuan["weighted_rating"] > 3].sort_values("korelasyon",
                                                                                                   ascending=False)
    a = hata_k_o_k(tavsiyeEt)
    print("{:.4f} Oran degeri icin Sonuc : {:.6f} ".format(oran, a))

    return a

# GWO Optimizasyonu için gerekli parametreler
def fitness_function(a):
    return optimize_ratio(a)
alt_sinir = 50
ust_sinir = 80
populasyon_sayisi = 50
max_iterasyon = 100
arama_uzayi = 1

start = time.perf_counter()
gwo = Gri_kurt_opt(fitness_function, alt_sinir,ust_sinir,
                         populasyon_sayisi, max_iterasyon)
best_solution, best_fitness = gwo.optimize()
end = time.perf_counter()
print("***********************************************")
print("Optimize edilmis oran degeri: {:.4f}".format(best_solution))
print("Uygunluk fonksiyonu degeri: {:.7f}".format(best_fitness))
print("Optimizasyon süresi : ", (end - start))
print("***********************************************")

