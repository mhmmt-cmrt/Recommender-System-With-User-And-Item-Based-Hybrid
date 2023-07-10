#coding=utf-8

import pandas as pd
import numpy as np
import random
class Gri_kurt_opt:
        def __init__(self, fitness_fonksiyonu, alt_sinir, ust_sinir, populasyon_sayisi, max_iterasyon):
            self.fitness_fonksiyonu = fitness_fonksiyonu
            self.populasyon_sayisi = populasyon_sayisi
            self.max_iterasyon = max_iterasyon
            self.alt_sinir = alt_sinir
            self.ust_sinir = ust_sinir

        def optimize(self):
            # gri kurtların pozisyonları rasgele başlatılıyor
            kurt_pozisyonlari = np.random.uniform(self.alt_sinir, self.ust_sinir, self.populasyon_sayisi).T

            #Her bir gri kurdun pozisyonu için fitness değeri hesaplanır
            fitness_degerleri = np.array([self.fitness_fonksiyonu(pos) for pos in kurt_pozisyonlari])

            # fitnesslara göre alfa,beta, delta kurtlar bulunur
            alpha_idx, beta_idx, delta_idx = np.argsort(fitness_degerleri)[:3]

            #en iyi gri kurt alfa, 2. beta, 3. delta kurt olarak kaydedilir.
            alpha = kurt_pozisyonlari[alpha_idx]
            beta = kurt_pozisyonlari[beta_idx]
            delta = kurt_pozisyonlari[delta_idx]

            # gri kurt iterasyonlarına başlanır
            for iter_idx in range(self.max_iterasyon):
                # a lineer olarak 2 den 0 a düşürülür ve [0,1] arası rassal vektördür.
                # kurtların arama uzayındaki noktalardan herhangi bir yere gitmesini sağlar
                a = 2 - iter_idx * (2 / self.max_iterasyon)

                for i in range(self.populasyon_sayisi):
                        # bu kısımdan x+1 değerine kadar olan kısım avlanma stratejisi kısmıdır.
                        r1, r2 = np.random.rand(2)  # rasgele katsayılar.

                        #alfa beta ve delta için katsayı vektörleridir.
                        A_alpha = 2 * a * r1 - a
                        A_beta = 2 * a * r1 - a
                        A_delta = 2 * a * r1 - a

                        C_alpha = 2 * r2
                        C_beta = 2 * r2
                        C_delta = 2 * r2

                        # av ve kurt arası mesafeler (mesafe vektörleri)
                        D_alpha = abs(C_alpha * alpha - kurt_pozisyonlari[i])
                        D_beta = abs(C_beta * beta - kurt_pozisyonlari[i])
                        D_delta = abs(C_delta * delta - kurt_pozisyonlari[i])

                        #alfa beta delta değerleri; alfa beta delta kurtları için avın pozisyonu gösterir

                        # X1, X2, X3 alfa beta delta kurtlar için deneme vektörleridir.
                        X1 = (alpha - A_alpha * D_alpha)
                        X2 = (beta -  A_beta *  D_beta)
                        X3 = (delta - A_delta * D_delta)
                        # X(t+1) = (X1 + X2 + X3) / 3 ---> avın yeni konumu
                        kurt_pozisyonlari[i] = (X1 + X2 + X3) / 3

                        kurt_pozisyonlari[i] = np.clip(kurt_pozisyonlari[i], self.alt_sinir, self.ust_sinir)
                        # güncel değerlerin maliyeti(fitness) hesaplanır
                        guncel_fitness_degeri = self.fitness_fonksiyonu(kurt_pozisyonlari[i])

                        #maliyete(fitness) göre kurtlar(çözümler) güncellenir
                        if guncel_fitness_degeri < fitness_degerleri[alpha_idx]:
                            delta_idx = beta_idx
                            beta_idx = alpha_idx
                            alpha_idx = i
                        elif guncel_fitness_degeri < fitness_degerleri[beta_idx]:
                            delta_idx = beta_idx
                            beta_idx = i
                        elif guncel_fitness_degeri < fitness_degerleri[delta_idx]:
                            delta_idx = i

                        # fitness degerleri güncellenir
                        fitness_degerleri[i] = guncel_fitness_degeri

                        #güncellemeye göre en iyi kurtlar(seçilenler) yeniden belirlenir
                        alpha = kurt_pozisyonlari[alpha_idx]
                        beta = kurt_pozisyonlari[beta_idx]
                        delta = kurt_pozisyonlari[delta_idx]

            return alpha, guncel_fitness_degeri


