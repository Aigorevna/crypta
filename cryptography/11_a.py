#Двухпроходный протокол идентификации
import socket
import pickle
from datetime import datetime, UTC
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
import datetime
from datetime import timezone
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

def get_timestamp():
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')

def main():
        choice = input("Выберите тип идентификации (1 для временной метки, 2 для случайного числа): ")
        message1 = input("Введите сообщение M1 для шифрования: ")
        message2 = input("Введите сообщение M2 для открытой передачи: ")
        KEY = get_random_bytes(16)

        if choice == '1':
            auth_data = get_timestamp()
        else:
            auth_data = str(os.urandom(16).hex())

        data = {
            'identifier': IDENTIFIER1,
            'auth_data': auth_data,
            'message': message1
        }

        serialized_data = pickle.dumps(data)
        encrypted_payload = encrypt_data(serialized_data, KEY)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 12345))
            s.sendall(KEY)
            s.sendall(pickle.dumps({
                'open_message': message2,
                'encrypted_data': encrypted_payload
            }))

            response = s.recv(4096)
            received = pickle.loads(response)

        print(f"Получено открытое сообщение M2: {received['open_message']}")

        decrypted_data = decrypt_data(received['encrypted_data'], KEY)
        encrypted_data = pickle.loads(decrypted_data)

        if encrypted_data['identifier'] == IDENTIFIER2:
            if choice == '1':
                received_time = datetime.datetime.strptime(encrypted_data['auth_data'], '%Y-%m-%dT%H:%M:%SZ')
                received_time = received_time.replace(tzinfo=timezone.utc)
                current_time = datetime.datetime.now(datetime.UTC)
                time_diff = (current_time - received_time).total_seconds()
                if abs(time_diff) < 30:
                    print("Идентификация успешна!")
                    print(f"Получено сообщение: {encrypted_data['message']}")
                else:
                    print("Идентификация провалена: Временная метка устарела.")
            else:
                random = encrypted_data['auth_data']
                if random == auth_data:
                    print("Идентификация успешна!")
                    print(f"Получено сообщение: {encrypted_data['message']}")
                else:
                    print("Идентификация провалена")

        else:
            print("Идентификация провалена: Неверный идентификатор.")

if __name__ == '__main__':
    main()