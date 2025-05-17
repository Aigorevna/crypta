#Реализация протокола Диффи-Хеллмана
import socket
import pickle
import random
import sympy
from crypta import generate_prime, fast_exponentiation_mod

def generate_parameters():
    p = generate_prime(256)
    g = sympy.primitive_root(p)
    return p, g

def save_parameters(filename, p, g, x, alpha, k):
    with open(filename, 'w') as f:
        f.write(f"Prime (p): {p}\n")
        f.write(f"Primitive Root (g): {g}\n")
        f.write(f"X: {x}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"K: {k}\n")

def main():
    p, g = generate_parameters()
    x = random.randint(1, p-1)
    alpha = fast_exponentiation_mod(g, x, p)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 12345))
    server_socket.listen(1)
    print("Ожидание подключения...")

    conn, addr = server_socket.accept()

    data_to_send = {'p': p, 'g': g, 'alpha': alpha}
    conn.send(pickle.dumps(data_to_send))
    print("Отправлены p, g, и alpha")

    data_received = pickle.loads(conn.recv(4096))
    beta = data_received['beta']
    print("Получено beta")

    k = fast_exponentiation_mod(beta, x, p)
    print(f"Вычислено K: {k}")

    save_parameters('17a.txt', p, g, x, alpha,k)
    print("Параметры сохранены в  17a.txt")

    conn.close()
    server_socket.close()


if __name__ == "__main__":
    main()