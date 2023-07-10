#coding=utf-8

import numpy as np
def hata_k_o_k(tahminler):
    h_k_Ort = np.mean(
        [float((x - y) ** 2) for x, y in zip(tahminler['rating'], tahminler['weighted_rating'])]
    )
    h_k_o_Karekok = np.sqrt(h_k_Ort)

    return h_k_o_Karekok

def normalize_populerlik(popularite_tablosu):
    min_deger = popularite_tablosu["popularite"].min()
    max_deger = popularite_tablosu["popularite"].max()
    for i in popularite_tablosu["popularite"]:
        popularite_tablosu["normalized_pop"] = ((popularite_tablosu["popularite"] - min_deger) / (max_deger - min_deger))

    return popularite_tablosu