import socket
import pickle
import random
from crypta import generate_prime, fast_exponentiation_mod, sha512

def generate_parameters(password):
    p = generate_prime(256)
    hash_hex = sha512(password.encode('ascii'))
    h = int(hash_hex, 16)
    g = fast_exponentiation_mod(h, 2, p)
    if g == 0 or g == 1:
        raise ValueError("Неподходящий g (0 или 1)")
    return p, g


def save_parameters(filename, p, g, x, alpha, k):
    with open(filename, 'w') as f:
        f.write(f"Prime (p): {p}\n")
        f.write(f"Generator (g): {g}\n")
        f.write(f"X: {x}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"K: {k}\n")


def main():
    password = input("Введите общий пароль: ")

    p, g = generate_parameters(password)
    x = random.randint(1, p - 1)
    alpha = fast_exponentiation_mod(g, x, p)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 12345))
    server_socket.listen(1)
    print("Ожидание подключения...")

    conn, addr = server_socket.accept()
    print(f"Подключено : {addr}")

    data_to_send = {'p': p, 'g': g, 'alpha': alpha}
    conn.send(pickle.dumps(data_to_send))
    print("Отправлены p, g и alpha")

    data_received = pickle.loads(conn.recv(4096))
    beta = data_received['beta']
    print("Получено beta")

    k = fast_exponentiation_mod(beta, x, p)
    print(f"Вычислено K: {k}")

    save_parameters('18a.txt', p, g, x, alpha, k)
    print("Параметры сохранены в 18a.txt")

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    main()