import socket
import pickle
from crypta import generate_keypair, save_pkcs8, save_pkcs12, decrypt, sha256

IDENTIFIER1 = 'User1'


def main():
    public_key, private_key = generate_keypair(1024)

    save_pkcs8(public_key, "public_key.txt")
    save_pkcs12(private_key, "private_key.txt")

    block_size = 128

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 12345))
        s.listen()
        print("User 2 ожидает подключения...")
        conn, addr = s.accept()

        with conn:
            data = conn.recv(4096)
            received = pickle.loads(data)

            print(f"Получено сообщение от: {received['A']}")

            decryrt_data = decrypt(private_key, received['encrypted_data'], block_size)
            z = decryrt_data[:-len(received['A'])]
            A = decryrt_data[-len(received['A']):]
            z_bytes = bytes.fromhex(z)
            h_z = sha256(z_bytes)
            if h_z == received['h_z'] and A == received['A']:
                print ("Успешный успех")

            conn.sendall(pickle.dumps({
                'z': z
            }))
            print("Ответ отправлен. Идентификация успешна!")

if __name__ == '__main__':
    main()
