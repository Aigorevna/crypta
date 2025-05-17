#Протокол аутентификации на основе ассиметричного шифра
import socket
import pickle
import os
from crypta import sha256, encrypt

IDENTIFIER1 = 'User1'
IDENTIFIER2 = 'User2'


def load_public_key(filename):
    with open(filename, 'r') as f:
        e = int(f.readline().split(': ')[1])
        n = int(f.readline().split(': ')[1])
    return (e, n)

def main():
    public_key = load_public_key("public_key.txt")
    z_byte = os.urandom(16)
    z = z_byte.hex()
    h_z = sha256(z_byte)
    block_size =  128
    data_to_encrypt = z + IDENTIFIER1

    encrypt_data = encrypt(public_key, data_to_encrypt, block_size)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 12345))
        s.sendall(pickle.dumps({
            'h_z': h_z,
            'A': IDENTIFIER1,
            'encrypted_data': encrypt_data
        }))
        response = s.recv(4096)
        received = pickle.loads(response)
        z_1 = received['z']

        if z_1 == z:
            print("Успех")


if __name__ == '__main__':
    main()