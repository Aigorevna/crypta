import socket
import pickle
import random
from crypta import fast_exponentiation_mod

def save_parameters(filename, p, g, y, beta, k):
    with open(filename, 'w') as f:
        f.write(f"Prime (p): {p}\n")
        f.write(f"Primitive Root (g): {g}\n")
        f.write(f"Y: {y}\n")
        f.write(f"Beta: {beta}\n")
        f.write(f"K: {k}\n")

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    print("Подключено")

    data_received = pickle.loads(client_socket.recv(4096))
    p = data_received['p']
    g = data_received['g']
    alpha = data_received['alpha']
    print("Получены p, g, and alpha")

    y = random.randint(1, p-1)
    beta = fast_exponentiation_mod(g, y, p)

    data_to_send = {'beta': beta}
    client_socket.send(pickle.dumps(data_to_send))
    print("Отправлено beta")

    k = fast_exponentiation_mod(alpha, y, p)
    print(f"Вычислено k: {k}")

    save_parameters('17b.txt', p, g, y, beta, k)
    print("Параметры сохранены в  17b.txt")

    client_socket.close()

if __name__ == "__main__":
    main()