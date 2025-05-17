#Трехпроходный протокол идентификации
import socket
import pickle
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
from Crypto.Random import get_random_bytes

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
    KEY = get_random_bytes(16)
    message1 = input("Введите сообщение M1 для открытой передачи: ")
    r_a = str(os.urandom(16).hex())

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 12345))
        s.sendall(KEY)
        s.sendall(pickle.dumps({
            'r_a': r_a,
            'open_message': message1
        }))
        response = s.recv(4096)
        received = pickle.loads(response)

        decrypted_data = decrypt_data(received['encrypted_data'], KEY)
        encrypted_data = pickle.loads(decrypted_data)
        r_b = encrypted_data['r_b']

        if encrypted_data['identifier'] == IDENTIFIER2 and encrypted_data['r_a'] == r_a:
            print("Первая проверка прошла успешно")
            print(f"Получено открытое сообщение M3: {received['open_message']}")
            print(f"Получено зашифрованное сообщение М2: {encrypted_data['message2']}")
            message4 = input("Введите сообщение M4 для шифрования: ")
            message5 = input("Введите сообщение M5 для открытой передачи: ")

            response_encrypted = {
                'r_b': r_b,
                'r_a': r_a,
                'identifier': IDENTIFIER1,
                'message4': message4
            }
            encrypted_response = encrypt_data(pickle.dumps(response_encrypted), KEY)

            s.sendall(pickle.dumps({
                'open_message': message5,
                'encrypted_data': encrypted_response
            }))
            print("Ответ отправлен. Идентификация успешна!")



if __name__ == '__main__':
    main()