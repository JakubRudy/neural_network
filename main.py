import random

xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]

w = random.uniform(-1.0,1.0)
b = random.uniform(-1.0,1.0)

learning_rate = 0.01
iteracje = 500

for i in range(iteracje):
    total_loss = 0

    for x,y_true in zip(xs,ys):
        y_pred = w * x + b
        loss = (y_pred - y_true)**2
        total_loss += loss

        dw = 2 * (y_pred - y_true) * x
        db = 2 * (y_pred - y_true)

        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
    if i % 100 == 0:
        avg_loss = total_loss / len(xs)
        print(f"Epoka {i} | Średni błąd: {avg_loss:.4f} | Waga: {w:.4f} | Bias: {b:.4f}")

print(f"\nKoniec treningu! Ostateczne parametry: Waga = {w:.4f}, Bias = {b:.4f}")
print("Prawidłowe wartości ze wzoru (y = 2x - 1) to Waga = 2.0, Bias = -1.0")

test_x = 10.0
wynik = w * test_x + b
print(f"Przewidywanie dla x=10 (powinno być 19): {wynik:.4f}")