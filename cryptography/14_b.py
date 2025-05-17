import socket
import pickle
from crypta import  sha256, sha512, stribog_main

HASH_FUNCTIONS = {
    'SHA256': sha256,
    'SHA512': sha512,
    'GOST256': lambda m: stribog_main(m, 256),
    'GOST512': lambda m: stribog_main(m, 512)
}

users = {}
host = '127.0.0.1'
port = 12345

def register_user(data) :
    user_id = data['user_id']
    if user_id in users:
        return {'success': False, 'message': 'Пользователь уже существует'}

    if data['hash_func'] not in HASH_FUNCTIONS:
        return {'success': False, 'message': 'Неподдерживаемая хеш-функция'}

    users[user_id] = {
        'hash_func': data['hash_func'],
        'current_hash': data['final_hash'],
        'attempt': 0,
        'max_attempts': data['max_attempts']
    }
    return {'success': True, 'message': 'Регистрация успешна'}

def authenticate_user(data) :
    user_id = data['user_id']
    if user_id not in users:
        return {'success': False, 'message': 'Пользователь не найден'}

    user = users[user_id]
    attempt = data['attempt']
    received_hash = data['hash']

    if attempt != user['attempt'] + 1:
        return {'success': False, 'message': 'Неверный номер попытки'}

    hash_func = HASH_FUNCTIONS[user['hash_func']]
    current_hash_str = received_hash
    expected_hash_str = hash_func(current_hash_str)
    expected_hash = bytes.fromhex(expected_hash_str)

    if expected_hash != user['current_hash']:
        return {'success': False, 'message': 'Неверный хеш'}

    user['current_hash'] = received_hash
    user['attempt'] = attempt

    return {'success': True, 'message': 'Аутентификация успешна'}

def handle_request(data) :
    action = data.get('action')
    if action == 'register':
        return register_user(data)
    elif action == 'authenticate':
        return authenticate_user(data)
    return {'success': False, 'message': 'Неверное действие'}

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Центр аутентификации запущен на {host}:{port}")
        while True:
            conn, addr = s.accept()
            with conn:
                data = pickle.loads(conn.recv(1024))
                response = handle_request(data)
                conn.sendall(pickle.dumps(response))

if __name__ == "__main__":
    start_server()