import socket
import pickle
import random

from crypta import fast_exponentiation_mod

p = None
q = None
alpha = None
k_1 = None
P_1 = None

def load_params():
    global p, q, alpha, k_1, P_1
    with open("10_param.txt", 'r') as f:
        lines = f.readlines()
        p = int(lines[0].split(": ")[1].strip())
        q = int(lines[1].split(": ")[1].strip())
        alpha = int(lines[2].split(": ")[1].strip())
        k_1 = int(lines[3].split(": ")[1].strip())  # Member 1 k_1
        P_1 = int(lines[4].split(": ")[1].strip())  # Member 1 P_1

def process_message():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 1337))
    server.listen(1)
    print("А слушает на порту 1337...")

    conn, addr = server.accept()
    print(f"А: Подключился лидер с адреса {addr}")


    try:
        delta = pickle.loads(conn.recv(4096))

        t_1 = random.randint(1, q)
        R_1 = fast_exponentiation_mod(alpha, t_1, p)
        print(f"А: Вычислено R_1 = {R_1}")

        conn.send(pickle.dumps(R_1))

        E = pickle.loads(conn.recv(4096))
        print(f"А: Получено E = {E}")


        S_a = t_1 + (k_1 * delta * E) % q
        print(f"А: Вычислено S_a = {S_a}")

        conn.send(pickle.dumps(S_a))

    finally:
        conn.close()
        server.close()
        print("А: Соединение закрыто.")

if __name__ == "__main__":
    load_params()
    process_message()