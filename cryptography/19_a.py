import socket
import pickle
import random
from sympy import isprime

def generate_large_prime(bits=16):
    num = random.getrandbits(bits)
    while not isprime(num):
        num = random.getrandbits(bits)
    return num

def create_polynomial(modulus, degree=1):
    return [random.randint(0, modulus - 1) for _ in range(degree + 1)]

def evaluate_polynomial(coeffs, x, modulus):
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % modulus
    return result

def main():
    modulus = generate_large_prime()
    print(f"Finite field modulus: {modulus}")

    id1 = random.randint(1, modulus - 1)
    id2 = random.randint(1, modulus - 1)
    while id2 == id1:
        id2 = random.randint(1, modulus - 1)
    print(f"User 1 ID: {id1}, User 2 ID: {id2}")

    poly = create_polynomial(modulus)
    print(f"Secret polynomial coefficients: {poly} (kept secret)")

    key_material1 = [evaluate_polynomial(poly, (id1 + i) % modulus, modulus) for i in range(2)]
    key_material2 = [evaluate_polynomial(poly, (id2 + i) % modulus, modulus) for i in range(2)]
    print(f"Key material for User 1: {key_material1}")
    print(f"Key material for User 2: {key_material2}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 12345))
    server_socket.listen(2)
    print("Trusted Authority waiting for connections...")

    for user_id, key_material in [(id1, key_material1), (id2, key_material2)]:
        client_socket, addr = server_socket.accept()
        print(f"Connected to client at {addr}")
        data = {'user_id': user_id, 'other_id': id2 if user_id == id1 else id1,
                'key_material': key_material, 'modulus': modulus}
        client_socket.send(pickle.dumps(data))
        client_socket.close()

    server_socket.close()

if __name__ == "__main__":
    main()