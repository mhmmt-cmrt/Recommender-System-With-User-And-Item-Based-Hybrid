# coding=utf-8


def guguk_kusu_opt(yuva_sayisi, pa, yuva_boyutu, alt_sinir, ust_sinir,
                   fitness_fonksiyonu, max_iterasyon=100, stdtol=1e-2):
    import numpy as np
    from copy import deepcopy
    def sinir_kontrol(s):
        # **************************************************************
        # sınırlar kontrol ediliyor
        # Üretilen bu yuvaların alt ve üst limitler
        # arasında olup olmadığı  kontrol edilir. Eğer değilse limitlenir.
        # **************************************************************
            if s < alt_sinir:   # sınırların altında ise alt sınıra eşitler
                 s = alt_sinir
            elif s > ust_sinir:  # sınırların üstünde ise üst sınıra eşitler
                 s = ust_sinir
            return s

    def yuva_uret(yuva, bestt):
        from scipy.special import gamma
        #**************************************************************
        # fonksiyon, yeni yuvalar (çözüm adayları) üretmek için levy uçuşu
        # yöntemini kullanır. Her bir yuvanın değerleri rassal olarak değiştirilir
        # ve sinir_kontrol fonksiyonu ile sınırlar kontrol edilir.
        # bu kısım levy uçuşunun formüle edilmiş halinin kodlamasıdır.
        #**************************************************************
        n1 = np.shape(yuva)[0]  # yuva dizisinin satır sayısı
        m = np.shape(yuva)[1]   # yuva dizisinin sütun sayısı
        yuva_old = deepcopy(yuva)
        beta = 3 / 4  # genellikle 1.5 alınır
        sigma = ((gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (
                gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (
                        1 / beta)
        for j in range(n1):
            s = yuva_old[j]
            # u, v dağılımı 0 olan rassal değişkenlerdir.
            u = sigma * np.random.randn(1, m)
            v = np.random.randn(1, m)
            step = u / (np.abs(v) ** (1 / beta))  # Mantegna algoritmasına göre adım büyüklüğü
            stepsize = 0.01 * step * (s - bestt)
            s = s + stepsize * np.random.randn(1, m)
            yuva_old[j] = sinir_kontrol(s)

        return yuva_old

    def get_best_yuva(yuva, newnest, best_value):
        #**********************************************************************
        # mevcut yuvalardan ve yeni yuvalardan en iyi yuvalar seçilir.
        # fitness fonksiyonunu kullanılarak uygunluk değerleri hesaplanır ve
        # en iyi yuva ve en düşük uygunluk değeri döndürülür.
        #**********************************************************************
        yuva_old = deepcopy(yuva)
        n2 = np.shape(yuva)[0]
        # satir sayisi (n2). yuva dizisinin boyutunu elde ederek,
        # algoritmanın çalışması için gerekli olan popülasyon
        # büyüklüğünü belirlenir
        fnew = fitness_fonksiyonu(newnest)
        for j in range(n2):
            if fnew[j] <= best_value[j]:
                best_value[j] = fnew[j]
                yuva_old[j] = newnest[j]
        f_minn = np.amin(best_value)  # dizisinin en küçük değeri
        besta = yuva_old[np.argmin(best_value)]
        # yuva_old[np.argmin(best_value)] ifadesiyle best_value dizisindeki
        # en küçük değerin indeksi bulunur ve bu indekse karşılık gelen
        # yuva_old dizisinin değeri elde edilir.
        return f_minn, besta, yuva_old, best_value

    def empty_yuvalar(yuva):
        from copy import deepcopy
        # ************************************************************************
        # yabancı yumurta stratejisi uygulanır ve yuvaların çeşitliliği artırılır.
        # pa fark edilme olasılığı ile en kötü yuvalar terkedilerek yeni
        # yuvalar kurulur (en iyi yuvalar tutularak)
        # Fark edilirse K=1, edilmezse K=0 (K=0 iken yuva değişmez).
        # ************************************************************************
        n1 = np.shape(yuva)[0]
        n2 = np.shape(yuva)[1]
        k = np.random.rand(n1, n2) > pa
        yuva_cop = deepcopy(yuva)
        a = np.random.permutation(yuva_cop)
        b = np.random.permutation(yuva_cop)
        stepsize = np.random.rand(n1, n2) * (a - b)
        new_yuva = yuva + stepsize * k
        for j in range(n1):
            s = new_yuva[j]
            new_yuva[j] = sinir_kontrol(s)
        return new_yuva

    # *************************************************************
    # n tane nd boyutunda rastgele yuva (çözüm) alt ve üst limitler
    # içerisinde üretilir.
    # *************************************************************
    list1 = []
    yuvalar = alt_sinir + (ust_sinir - alt_sinir) * np.random.rand(yuva_sayisi, yuva_boyutu)
    fitness = 10 ** 5 * np.ones((yuva_sayisi, 1))
    f_min, best_yuva, yuvalar, fitness = get_best_yuva(yuvalar, yuvalar, fitness)
    for z in range(max_iterasyon):
        # Levy uçuşuyla yeni yuvalar üretilir (en iyi yuvalar tutularak).
        new_yuvalar = yuva_uret(yuvalar, best_yuva)
        f_new, best, yuvalar, fitness = get_best_yuva(yuvalar, new_yuvalar, fitness)
        new_yuvalar = empty_yuvalar(yuvalar)
        f_new, best, yuvalar, fitness = get_best_yuva(yuvalar, new_yuvalar, fitness)
        if f_new < f_min:
            f_min = f_new
            best_yuva = best
        std = np.std(fitness)  # standart sapması alınır
        list1.append(np.amin(fitness))
        # guguk kuşlarının uygunluk değerlerinin en küçüğünü list1 listesine ekle
        if std <= stdtol:
            break
    return f_min, best_yuva
