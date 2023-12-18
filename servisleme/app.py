import streamlit as st
import pandas as pd  
import numpy as np  
import joblib  
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt  
import seaborn as sns




def main():

    st.sidebar.title('Streamlit ile ML Uygulaması')
    selected_page = st.sidebar.selectbox('Sayfa Seçiniz..',["-","Tahmin Yap","İstatislik Görüntüle","Hakkında"])

    if selected_page == "-":
        st.title("Streamlit Uygulamasına Hoşgeldiniz")

        st.markdown(
            """
            Bu proje makine öğrenmesi uygulamalarının web ortamında streamlit kullanılarak yayınlanmasına örnek olarak geliştirilmiştir.
            Bir emlak sitesi üzerinden yaklaşık 700 küsürat veri çekilmiştir ve incelenmiştir. Bu veriler kullanılarak
            makine öğrenmesi modelleri eğitilmiş ve projeye dahil edilmiştir.

            """
            )
        st.info("Tahmin yapmak, istatislikleri görüntülemek ve proje hakkında daha fazla bilgi edinmek için sol tarafta bulunan menüyü kullanınız.")


    if selected_page == "Tahmin Yap":
        predict()

    if selected_page == "İstatislik Görüntüle":
        eda()

    if selected_page == "Hakkında":
        about()


def about():
    st.title("Geliştirici Bilgileri")
    st.subheader("GitHub : (http://github.com/aliihsand/)")

def eda():
    st.title("İstatislikler")

    data = pd.read_excel("../algoritmalar/DüzeltilmişEmlakVerileri.xlsx")

    st.header("Bütün Veriler")
    st.dataframe(data)


def predict():

    # Datalar ve modellerin yüklenmesi
    lokasyonlar = load_data_lokasyon()
    kat_gruplari = load_data_kat_gruplari()
    oda_sayilari = load_data_oda_sayilari()
    isitma_tipleri = load_data_isitma_tipleri()
    bina_yaslari = load_data_bina_yaslari()


    # Kullanıcı arayüzü ve değer alma
    st.title("Merhaba, Streamlit!")
    selected_lokasyon = lokasyon_index(lokasyonlar, st.selectbox("Lokasyon seçiniz...", lokasyonlar))

    selected_kat_grubu = kat_grubu_index(kat_gruplari, st.selectbox("Kat Grubu...", kat_gruplari))

    selected_oda_sayisi = oda_sayisi_index(oda_sayilari, st.selectbox("Oda Sayısı seçiniz...", oda_sayilari))

    selected_isitma_tipi = isitma_tipi_index(isitma_tipleri, st.selectbox("Isıtma Tipi seçiniz...", isitma_tipleri))
    
    selected_bina_yasi = binanin_yasi_index(bina_yaslari, st.selectbox("Bina Yaşı seçiniz...", bina_yaslari))

    selected_yapi_durumu = yapi_durumu(st.radio("Yapı Durumu", ("Sıfır", "İkinci El")))

    selected_site = site_icerisinde(st.radio("Site İçerisinde", ("Evet", "Hayır")))

    selected_esya = esya_durumu(st.radio("Eşya Durumu", ("Eşyalı", "Boş")))

    selected_balkon = balkon_durumu(st.radio("Balkon Durumu", ("Var", "Yok")))

    selected_metrekare = st.number_input("Net Metrekare", min_value=50, max_value=300)
    st.write("Net Metrekare : " +str(selected_metrekare) + " M2")

    selected_bina_kat_sayisi = st.number_input("Bina Kat Sayısı", min_value=1, max_value=20)
    st.write("Bina Kat Sayısı : " +str(selected_bina_kat_sayisi) + " Kat")

    selected_banyo = st.number_input("Banyo Sayısı", min_value=0, max_value=5)
    st.write("Banyo Sayısı: " +str(selected_banyo))

    selected_wc = st.number_input("WC Sayısı", min_value=0, max_value=5)
    st.write("WC Sayısı : " +str(selected_wc))

    selected_aidat = st.number_input("Aidat", min_value=0, max_value=20000)
    st.write("Aidat : " +str(selected_aidat) + " TL")

    predict_value = create_prediction_value(selected_lokasyon, selected_kat_grubu, selected_oda_sayisi, selected_isitma_tipi,
                                            selected_bina_yasi, selected_site, selected_esya, selected_balkon, selected_yapi_durumu,
                                            selected_metrekare, selected_bina_kat_sayisi, selected_banyo, selected_wc, selected_aidat)
    predict_models = load_models()

    if  st.button("Tahmin Yap"):
        result = predict_models1(predict_models,predict_value)
        if result != None:
            st.success("Tahmin Başarılı")
            st.balloons()
            st.write("Tahmin Edilen Fiyat: " + result + " TL")
        else:
            st.error("Tahmin yaparken hata meydana geldi..!")


def load_data_lokasyon():
    data = pd.read_excel("../algoritmalar/DüzeltilmişEmlakVerileri.xlsx")
    lokasyonlar = data["Lokasyon"].values
    lokasyonlar = np.unique(lokasyonlar)
    lokasyonlar = pd.DataFrame(data=lokasyonlar, columns=["Lokasyon"])
    return lokasyonlar

def load_data_kat_gruplari():
    data = pd.read_excel("../algoritmalar/DüzeltilmişEmlakVerileri.xlsx")
    kat_gruplari = data["Kat Grubu"].values
    kat_gruplari = np.unique(kat_gruplari)
    kat_gruplari = pd.DataFrame(data=kat_gruplari, columns=["Kat Grubu"])
    return kat_gruplari

def load_data_oda_sayilari():
    data = pd.read_excel("../algoritmalar/DüzeltilmişEmlakVerileri.xlsx")
    oda_sayilari = data["Oda Sayısı"].values
    oda_sayilari = np.unique(oda_sayilari)
    oda_sayilari = pd.DataFrame(data=oda_sayilari, columns=["Oda Sayısı"])
    return oda_sayilari

def load_data_isitma_tipleri():
    data = pd.read_excel("../algoritmalar/DüzeltilmişEmlakVerileri.xlsx")
    isitma_tipleri = data["Isıtma Tipi"].values
    isitma_tipleri = np.unique(isitma_tipleri)
    isitma_tipleri = pd.DataFrame(data=isitma_tipleri, columns=["Isıtma Tipi"])
    return isitma_tipleri

def load_data_bina_yaslari():
    data = pd.read_excel("../algoritmalar/DüzeltilmişEmlakVerileri.xlsx")
    bina_yaslari = data["Binanın Yaşı"].values
    bina_yaslari = np.unique(bina_yaslari)
    bina_yaslari = pd.DataFrame(data=bina_yaslari, columns=["Binanın Yaşı"])
    return bina_yaslari

def load_models():
    best_model = joblib.load("../algoritmalar/emlak_best_model.pkl")
    return best_model

def lokasyon_index(lokasyonlar,lokasyon):
    index = int(lokasyonlar[lokasyonlar["Lokasyon"] == lokasyon].index.values)
    return index

def kat_grubu_index(kat_gruplari,kat_grubu):
    index = int(kat_gruplari[kat_gruplari["Kat Grubu"] == kat_grubu].index.values)
    return index

def oda_sayisi_index(oda_sayilari,oda_sayisi):
    index = int(oda_sayilari[oda_sayilari["Oda Sayısı"] == oda_sayisi].index.values)
    return index

def isitma_tipi_index(isitma_tipleri,isitma_tipi):
    index = int(isitma_tipleri[isitma_tipleri["Isıtma Tipi"] == isitma_tipi].index.values)
    return index

def binanin_yasi_index(bina_yaslari,binanin_yasi):
    index = int(bina_yaslari[bina_yaslari["Binanın Yaşı"] == binanin_yasi].index.values)
    return index

def site_icerisinde(site_icerisinde):
    if site_icerisinde == "Hayır":
        return 0
    else:
        return 1

def esya_durumu(esya_durumu):
    if esya_durumu == "Boş":
        return 0
    else:
        return 1

def balkon_durumu(balkon_durumu):
    if site_icerisinde == "Yok":
        return 0
    else:
        return 1

def yapi_durumu(yapi_durumu):
    if yapi_durumu == "İkinci El":
        return 0
    else:
        return 1

def predict_models1(model,res):
    result = str(int(model.predict(res))).strip('[]')
    return result

def create_prediction_value(lokasyon, 
                            kat_grubu,
                            oda_sayisi,
                            isitma_tipi, 
                            binanin_yasi,
                            site_icerisinde, 
                            esya_durumu, 
                            balkon_durumu, 
                            yapi_durumu,
                            net_metrekare, 
                            binanin_kat_sayisi,
                            banyo_sayisi,
                            wc_sayisi,
                            aidat):
    res = pd.DataFrame(data = 
                      {"Lokasyon":[lokasyon],
                       "Kat Grubu":[kat_grubu],
                       "Oda Sayısı":[oda_sayisi],
                       "Isıtma Tipi":[isitma_tipi],
                       "Binanın Yaşı":[binanin_yasi],
                       "Site İçerisinde":[site_icerisinde],
                       "Eşya Durumu":[esya_durumu],
                       "Balkon Durumu":[balkon_durumu],
                       "Yapı Durumu":[yapi_durumu],
                       "Net Metrekare":[net_metrekare],
                       "Binanın Kat Sayısı":[binanin_kat_sayisi],
                       "Banyo Sayısı":[banyo_sayisi], 
                       "WC Sayısı":[wc_sayisi],
                       "Aidat":[aidat]})
    return res


if __name__ == "__main__":
    main()