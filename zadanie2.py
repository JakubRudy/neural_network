import numpy as np
import pandas as p
import glob as g
import analizator as a
import pandas as pd

WINDOW_SIZE = 3          # Liczba próbek z poprzednich chwil (historia)
LEARNING_RATE = 0.01     # Krok nauki algorytmu (alfa)
EPOCHS = 50              # Ile razy sieć przejrzy cały zbiór danych

# Nazwy kolumn dokładnie z Twojego pliku Excel/CSV
FEATURE_COLS = [
    "data__tagData__gyro__z",
    "data__tagData__linearAcceleration__x",
    "data__tagData__linearAcceleration__y",
    "data__orientation__yaw",
    "data__tagData__pressure",
    "data__coordinates__x",  # Surowe, błędne współrzędne UWB X
    "data__coordinates__y"   # Surowe, błędne współrzędne UWB Y
]
TARGET_COLS = ["reference__x", "reference__y"]  # Idealne współrzędne (Ground Truth)

def polacz_pliki(pliki_statczne):
    lista_df = []
    for plik in pliki_statczne:
        df = pd.read_excel(plik)
        lista_df.append(df)
    return lista_df
def wybierz_sale():
    while(True):
        print("Wybierz sale do analizy:")
        print("A) Sala F10")
        print("B) Sala F8")
        wybor = input("Opcja: ")
        if wybor == "A" or wybor == "B":
            break
    return wybor

def main():
    wybor = wybierz_sale()
    sciezka = "sciezka"
    sciezka_test = "test"
    if wybor == "A":
        sciezka = "F8/*_stat_*.xlsx"
        sciezka_test = "F8/f8_random_1p.xlsx"
    elif wybor == "B":
        sciezka = "F10/*_stat_*.xlsx"
        sciezka_test = "F10/f10_random_1p.xlsx"

    pliki_statyczne = g.glob(sciezka)
    print(f"Znaleziono {len(pliki_statyczne)} plików statycznych do uczenia.")

    lista_danych = polacz_pliki(pliki_statyczne)

    dane_treningowe = pd.concat(lista_danych, ignore_index=True)

    X_trening_wiersze = dane_treningowe[FEATURE_COLS].values
    y_trening = dane_treningowe[TARGET_COLS].values

    dane_testowe = pd.read_excel(sciezka_test)
    print(f"Wczytano plik dynamiczny do weryfikacji: {sciezka_test}")
if __name__ == "__main__":
    main()