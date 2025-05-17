import socket
import pickle
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

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

            print(f"Получено открытое сообщение M2: {received['open_message']}")

            decrypted_data = decrypt_data(received['encrypted_data'], KEY)
            encrypted_data = pickle.loads(decrypted_data)

            if encrypted_data['identifier'] == IDENTIFIER1:
                print(f"Получены зашифрованные данные: {encrypted_data}")
                response_message3 = input("Введите ответное сообщение M3 для шифрования: ")
                response_message4 = input("Введите ответное сообщение M4 для открытой передачи: ")

                response_encrypted = {
                    'identifier': IDENTIFIER2,
                    'auth_data': encrypted_data['auth_data'],
                    'message': response_message3
                }
                encrypted_response = encrypt_data(pickle.dumps(response_encrypted), KEY)

                conn.sendall(pickle.dumps({
                    'open_message': response_message4,
                    'encrypted_data': encrypted_response
                }))
                print("Ответ отправлен. Идентификация успешна!")
            else:
                print("Идентификация провалена: Неверный идентификатор.")


if __name__ == '__main__':
    main()
