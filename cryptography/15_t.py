import socket
import pickle
from crypta import generate_prime
import logging

logging.basicConfig(filename='trusted_center_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_data(direction, data):
    logging.info(f"{direction}: {data}")

def trusted_center():
    p = generate_prime(512)
    q = generate_prime(512)
    n = p * q
    log_data("Generated", f"p={p}, q={q}, n={n}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 9998))
    server_socket.listen(2)
    log_data("Server", "Started on localhost:9998")
    print("Ключи сгенерированы и сервер запущен")

    conn_prover, addr_prover = server_socket.accept()
    log_data("Connection", f"Prover connected from {addr_prover}")
    conn_verifier, addr_verifier = server_socket.accept()
    log_data("Connection", f"Verifier connected from {addr_verifier}")

    data_to_both = {'n': n}
    serialized_data = pickle.dumps(data_to_both)
    conn_prover.send(serialized_data)
    conn_verifier.send(serialized_data)
    log_data("Sent to Prover", data_to_both)
    log_data("Sent to Verifier", data_to_both)

    conn_prover.close()
    conn_verifier.close()
    server_socket.close()
    log_data("Server", "Closed")

if __name__ == "__main__":
    trusted_center()