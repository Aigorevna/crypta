import socket
import pickle
from datetime import datetime, UTC
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import datetime
import os

IDENTIFIER1 = 'User1'
IDENTIFIER2 = 'User2'


def encrypt_data(data, key):
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.IV
    padded_data = pad(data, AES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return iv + ciphertext


def decrypt_data(ciphertext, key):
    iv = ciphertext[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    padded_data = cipher.decrypt(ciphertext[AES.block_size:])
    return unpad(padded_data, AES.block_size)


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 12345))
        s.listen()
        print("User 2 ожидает подключения...")
        conn, addr = s.accept()

        with conn:
            KEY = conn.recv(16)
            data = conn.recv(4096)
            received = pickle.loads(data)
            print(f"Получено открытое сообщение M1: {received['open_message']}")
            r_a = received['r_a']
            r_b = str(os.urandom(16).hex())

            message2 = input("Введите ответное сообщение M2 для шифрования: ")
            message3 = input("Введите ответное сообщение M3 для открытой передачи: ")

            response_encrypted = {
                'r_a': r_a,
                'r_b': r_b,
                'identifier': IDENTIFIER2,
                'message2': message2
            }
            encrypted_response = encrypt_data(pickle.dumps(response_encrypted), KEY)

            conn.sendall(pickle.dumps({
                'open_message': message3,
                'encrypted_data': encrypted_response
            }))
            print("Ответ отправлен. Идентификация успешна!")

            data1 = conn.recv(4096)
            received = pickle.loads(data1)
            print(f"Получено открытое сообщение M5: {received['open_message']}")

            decrypted_data = decrypt_data(received['encrypted_data'], KEY)
            encrypted_data = pickle.loads(decrypted_data)

            if encrypted_data['identifier'] == IDENTIFIER1 and encrypted_data['r_b'] == r_b:
                print("Проверка прошла успешно ")
                print(f"Получено зашифрованное сообщение М4: {encrypted_data['message4']}")


if __name__ == '__main__':
    main()
