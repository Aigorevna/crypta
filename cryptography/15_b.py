import socket
import pickle
import logging
import random
from crypta import fast_exponentiation_mod

logging.basicConfig(filename='verifier_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_data(direction, data):
    logging.info(f"{direction}: {data}")

def verifier():
    tc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tc_socket.connect(('localhost', 9998))
    log_data("Connection", "Connected to Trusted Center")

    data = tc_socket.recv(4096)
    n_v = pickle.loads(data)
    n = n_v['n']
    log_data("Received from Trusted Center", n_v)
    tc_socket.close()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 9997))
    server_socket.listen(1)
    log_data("Server", "Started on localhost:9997")

    conn_prover, addr_prover = server_socket.accept()
    log_data("Connection", f"Prover connected from {addr_prover}")
    data = conn_prover.recv(4096)
    v = pickle.loads(data)['v']


    for i in range(10):
        data = conn_prover.recv(4096)
        x = pickle.loads(data)['x']
        log_data(f"Iteration {i+1} Received from Prover", f"x={x}")

        c = random.randint(0, 1)
        log_data(f"Iteration {i+1} Sent to Prover", f"c={c}")
        conn_prover.send(pickle.dumps({'c': c}))

        data = conn_prover.recv(4096)
        y = pickle.loads(data)['y']
        log_data(f"Iteration {i+1} Received from Prover", f"y={y}")

        left = fast_exponentiation_mod(y, 2, n)
        right = (x * fast_exponentiation_mod(v, c, n)) % n
        if left != right:
            log_data(f"Iteration {i+1} Verification", "Failed")
            conn_prover.send(pickle.dumps({'status': "failed"}))
            conn_prover.close()
            server_socket.close()
            print("Identification failed")
            return
        log_data(f"Iteration {i+1} Verification", "Passed")

    conn_prover.send(pickle.dumps({'status': "success"}))
    log_data("Final", "Identification successful")
    print("Identification successful")

    conn_prover.close()
    server_socket.close()

if __name__ == "__main__":
    verifier()