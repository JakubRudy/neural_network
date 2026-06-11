import numpy as np
import pandas as pd
import glob as g
import matplotlib.pyplot as plt
import analizator as a 

WINDOW_SIZE = 3          
LEARNING_RATE = 0.001    # Niższe = bezpieczniejsze
EPOCHS = 1000             

FEATURE_COLS = [
    "data__tagData__gyro__z",
    "data__tagData__linearAcceleration__x",
    "data__tagData__linearAcceleration__y",
    "data__orientation__yaw",
    "data__tagData__pressure",
    "data__coordinates__x",  
    "data__coordinates__y"   
]

def polacz_pliki(pliki_statyczne):
    return pd.concat([pd.read_excel(p) for p in pliki_statyczne], ignore_index=True)

def main():
    wybor = "A" # Domyślnie A dla szybszych testów
    sciezka_statyczne = "F10/*_stat_*.xlsx" if wybor == "A" else "F8/*_stat_*.xlsx"
    sciezka_test = "F10/f10_1p.xlsx" if wybor == "A" else "F8/f8_1p.xlsx"
    
    pliki_statyczne = g.glob(sciezka_statyczne)
    if not pliki_statyczne:
        print("Nie znaleziono plików!")
        return

    dane_treningowe = polacz_pliki(pliki_statyczne)
    dane_testowe = pd.read_excel(sciezka_test)

    # 5.0 Filtracja sygnałów
    dane_treningowe = a.usun_grube_bledy(dane_treningowe, FEATURE_COLS)
    dane_testowe = a.usun_grube_bledy(dane_testowe, FEATURE_COLS)

    X_train_raw = dane_treningowe[FEATURE_COLS].values
    X_test_raw = dane_testowe[FEATURE_COLS].values

    # Indeksy kolumn z pozycją UWB
    uwb_x_idx = FEATURE_COLS.index("data__coordinates__x")
    uwb_y_idx = FEATURE_COLS.index("data__coordinates__y")

    # =========================================================================
    # REWOLUCJA: Zamiast przewidywać pozycję, sieć przewiduje BŁĄD (RESZTĘ)!
    # y = Prawda - Zmierzony UWB
    # =========================================================================
    y_train_raw = np.column_stack((
        dane_treningowe["reference__x"].values - X_train_raw[:, uwb_x_idx],
        dane_treningowe["reference__y"].values - X_train_raw[:, uwb_y_idx]
    ))
    y_test_raw = np.column_stack((
        dane_testowe["reference__x"].values - X_test_raw[:, uwb_x_idx],
        dane_testowe["reference__y"].values - X_test_raw[:, uwb_y_idx]
    ))

    X_train_scaled, X_test_scaled = a.scale_data(X_train_raw, X_test_raw)
    
    # Skalowanie samego błedu (targetu)
    y_mean = np.mean(y_train_raw, axis=0)
    y_std = np.std(y_train_raw, axis=0)
    y_std[y_std < 0.1] = 1.0 
    y_train_scaled = (y_train_raw - y_mean) / y_std

    X_train, y_train = a.create_time_windows(X_train_scaled, y_train_scaled, WINDOW_SIZE)
    X_test, _ = a.create_time_windows(X_test_scaled, np.zeros_like(y_test_raw), WINDOW_SIZE)

    model = a.SiecNeuronowa(X_train.shape[1])
    
    print("\nRozpoczynam trening (Residual Learning)...")
    for epoka in range(EPOCHS):
        przewidywania = model.ruch(X_train)
        model.backward(X_train, y_train, przewidywania, lr=LEARNING_RATE)
        
        if epoka % 100 == 0 or epoka == EPOCHS - 1:
            mse = np.mean((przewidywania - y_train)**2)
            print(f"Epoka {epoka:04d}/{EPOCHS} | Błąd na resztach (MSE): {mse:.4f}")

    # ==============================================
    # REKONSTRUKCJA: Odszyfrowujemy wyniki
    # ==============================================
    przewidywany_blad_scaled = model.ruch(X_test)
    przewidywany_blad_raw = przewidywany_blad_scaled * y_std + y_mean
    
    # Odzyskujemy bazową trasę (odrzucamy pierwsze okna)
    bazowe_uwb_x = X_test_raw[WINDOW_SIZE:, uwb_x_idx]
    bazowe_uwb_y = X_test_raw[WINDOW_SIZE:, uwb_y_idx]
    bazowe_uwb = np.column_stack((bazowe_uwb_x, bazowe_uwb_y))
    
    prawdziwe_referencje_x = dane_testowe["reference__x"].values[WINDOW_SIZE:]
    prawdziwe_referencje_y = dane_testowe["reference__y"].values[WINDOW_SIZE:]
    prawdziwe_referencje = np.column_stack((prawdziwe_referencje_x, prawdziwe_referencje_y))

    # WYNIK FINALNY = UWB + Przewidziany Błąd
    przewidywania_test_raw = bazowe_uwb + przewidywany_blad_raw

    # OBLICZANIE BŁĘDÓW RZECZYWISTYCH W METRACH/CM
    bledy_sieci = a.calculate_errors(prawdziwe_referencje, przewidywania_test_raw)
    bledy_bazowe = a.calculate_errors(prawdziwe_referencje, bazowe_uwb)

    print(f"\nŚredni fizyczny błąd (surowy UWB): {np.mean(bledy_bazowe):.2f}")
    print(f"Średni fizyczny błąd PO FILTRACJI:  {np.mean(bledy_sieci):.2f}")

    df_blad = pd.DataFrame({"Dystrybuanta_Bledu": np.sort(bledy_sieci)})
    plik_xls = f"Wynikowa_Dystrybuanta_NN_Sala_{wybor}.xlsx"
    df_blad.to_excel(plik_xls, index=False)

    p_nn = np.arange(len(bledy_sieci)) / (len(bledy_sieci) - 1)
    p_base = np.arange(len(bledy_bazowe)) / (len(bledy_bazowe) - 1)

    plt.figure(figsize=(8, 6))
    plt.plot(np.sort(bledy_bazowe), p_base, label="Brak filtra (Surowy UWB)", linestyle="--", color="red")
    plt.plot(np.sort(bledy_sieci), p_nn, label="Korekta Siecią Neuronową", linewidth=2, color="green")
    plt.title("Dystrybuanta Błędu Lokalizacji")
    plt.xlabel("Błąd lokalizacji [Jednostki z pliku]")
    plt.ylabel("Prawdopodobieństwo P(X <= x)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()