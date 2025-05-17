import socket
import pickle
import sys

def evaluate_polynomial(coeffs, x, modulus):
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % modulus
    return result

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(('localhost', 12345))
    except ConnectionRefusedError:
        print("Error: Trusted Authority is not running.")
        sys.exit(1)

    data = pickle.loads(client_socket.recv(4096))
    user_id = data['user_id']
    other_id = data['other_id']
    key_material = data['key_material']
    modulus = data['modulus']
    client_socket.close()

    print(f"User ID: {user_id}")
    print(f"Key material: {key_material}")
    print(f"Finite field modulus: {modulus}")

    shared_key = evaluate_polynomial(key_material, other_id, modulus)
    print(f"Shared session key with User {other_id}: {shared_key}")

if __name__ == "__main__":
    main()