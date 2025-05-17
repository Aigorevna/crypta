import socket
import pickle
import random
from crypta import fast_exponentiation_mod, sha512

def compute_generator(password, p):
    hash_hex = sha512(password.encode('ascii'))
    h = int(hash_hex, 16)
    g = fast_exponentiation_mod(h, 2, p)
    if g == 0 or g == 1:
        raise ValueError("Неподходящий g (0 или 1)")
    return g

def save_parameters(filename, p, g, y, beta, k):
    with open(filename, 'w') as f:
        f.write(f"Prime (p): {p}\n")
        f.write(f"Generator (g): {g}\n")
        f.write(f"Y: {y}\n")
        f.write(f"Beta: {beta}\n")
        f.write(f"K: {k}\n")

def main():
    password = input("Введите общий пароль: ")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    print("Подключено ")

    data_received = pickle.loads(client_socket.recv(4096))
    p = data_received['p']
    g_received = data_received['g']
    alpha = data_received['alpha']
    print("Получены p, g и alpha")

    g = compute_generator(password, p)
    if g != g_received:
        print("Ошибка: g не совпадает, возможно, пароль неверный")
        client_socket.close()
        return

    y = random.randint(1, p-1)
    beta = fast_exponentiation_mod(g, y, p)

    data_to_send = {'beta': beta}
    client_socket.send(pickle.dumps(data_to_send))
    print("Отправлено beta")

    k = fast_exponentiation_mod(alpha, y, p)
    print(f"Вычислено K: {k}")

    save_parameters('18b.txt', p, g, y, beta, k)
    print("Параметры сохранены в 18b.txt")

    client_socket.close()

if __name__ == "__main__":
    main()