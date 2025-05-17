#Реализация протокола передачи ключа
import socket
import pickle
import os
from datetime import datetime, UTC
from crypta import sha512,generate_keypair, encrypt, save_pkcs8, save_pkcs12

IDENTIFIER2 = 'User2'

def sign_message_rsa(message, private_key):
    d, n, p, q = private_key
    message = message.encode('utf-8')
    message_hash = sha512(message)
    if isinstance(message_hash, bytes):
        message_hash = message_hash.hex()
    hash_int = int(message_hash, 16)
    signature = hex(pow(hash_int, d, n))[2:]
    return signature

def create_timestamp():
    utc_time = datetime.now(UTC).strftime("%y%m%d%H%M%SZ")
    return utc_time


def main():
    public_key_a, private_key_a = generate_keypair(512)
    save_pkcs8(public_key_a, 'public_key_a.txt')
    save_pkcs12(private_key_a, 'private_key_a.txt')

    with open('public_key_b.txt', 'r') as f:
        lines = f.readlines()
        e_b = int(lines[0].split(': ')[1])
        n_b = int(lines[1].split(': ')[1])
    public_key_b = (e_b, n_b)

    session_key = os.urandom(32)
    session_key = str(session_key.hex())
    with open("key.txt", 'w') as f:
        f.write(f"Session_key: {session_key}\n")
    block_size = 128

    time = create_timestamp()
    for_sign = IDENTIFIER2 + session_key + time

    signature = sign_message_rsa(for_sign, private_key_a)
    ciphertext = encrypt(public_key_b, session_key+time+signature, block_size)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 5001))
        data = {'ciphertext': ciphertext}
        s.sendall(pickle.dumps(data))
        print("Сообщение отправлено другому пользователю")

if __name__ == "__main__":
    main()