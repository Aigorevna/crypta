#Реализация протокола идентификации Фиата-Шамира
import socket
import pickle
import logging
import random
from crypta import fast_exponentiation_mod, gcd
logging.basicConfig(filename='prover_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_data(direction, data):
    logging.info(f"{direction}: {data}")

def find_s(n):
    while True:
        s = random.randint(2, n - 1)
        if gcd(s, n) == 1:
            return s

def prover():
    tc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tc_socket.connect(('localhost', 9998))
    log_data("Connection", "Connected to Trusted Center")

    data = tc_socket.recv(4096)
    n_v = pickle.loads(data)
    n = n_v['n']
    log_data("Received from Trusted Center", n_v)
    tc_socket.close()

    s = find_s(n)
    log_data("Find S", s)
    v = fast_exponentiation_mod(s, 2, n)

    verifier_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    verifier_socket.connect(('localhost', 9997))
    log_data("Connection", "Connected to Verifier")

    verifier_socket.send(pickle.dumps({'v': v}))
    log_data("Sent to Verifier", v)

    #итеративная процедура
    for i in range(10):
        z = random.randint(2, n - 1)
        x = fast_exponentiation_mod(z, 2, n)# r^2 * v^-1 mod n
        log_data(f"Iteration {i+1} Sent to Verifier", f"x={x}")

        verifier_socket.send(pickle.dumps({'x': x}))

        data = verifier_socket.recv(4096)
        c = pickle.loads(data)['c']
        log_data(f"Iteration {i+1} Received from Verifier", f"c={c}")
        if c == 0:
            y = z
        else:
            y = (z * s) % n
#если 1 что нуддно отправить чтобы обман
        log_data(f"Iteration {i+1} Sent to Verifier", f"y={y}")

        verifier_socket.send(pickle.dumps({'y': y}))

    data = verifier_socket.recv(4096)
    confirmation = pickle.loads(data)['status']
    log_data("Received from Verifier", confirmation)
    print("Identification successful" if confirmation == "success" else "Identification failed")

    verifier_socket.close()

if __name__ == "__main__":
    prover()