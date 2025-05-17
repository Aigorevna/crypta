import socket
import pickle
import random

from crypta import fast_exponentiation_mod

p = None
q = None
alpha = None
k_2 = None
P_2 = None

def load_params():
    global p, q, alpha, k_2, P_2
    with open("10_param.txt", 'r') as f:
        lines = f.readlines()
        p = int(lines[0].split(": ")[1].strip())
        q = int(lines[1].split(": ")[1].strip())
        alpha = int(lines[2].split(": ")[1].strip())
        k_2 = int(lines[5].split(": ")[1].strip())
        P_2 = int(lines[6].split(": ")[1].strip())

def process_message():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 1338))
    server.listen(1)
    print("Боб слушает на порту 1338...")

    conn, addr = server.accept()
    print(f"Боб: Подключился лидер с адреса {addr}")

    try:
        delta = pickle.loads(conn.recv(4096))

        t_2 = random.randint(1, q)
        R_2 = fast_exponentiation_mod(alpha, t_2, p)
        print(f"Б: Вычислено R_2 = {R_2}")

        conn.send(pickle.dumps(R_2))

        E = pickle.loads(conn.recv(4096))
        print(f"Б: Получено E = {E}")

        S_b = t_2 + (k_2 * delta * E) % q
        print(f"Б: Вычислено S_b = {S_b}")

        conn.send(pickle.dumps(S_b))

    finally:
        conn.close()
        server.close()
        print("Б: Соединение закрыто.")

if __name__ == "__main__":
    load_params()
    process_message()