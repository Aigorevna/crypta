#одноразовые пароли
import socket
import pickle
from crypta import sha256, sha256, sha512, stribog_main

HASH_FUNCTIONS = {
    'SHA256': sha256,
    'SHA512': sha512,
    'GOST256': lambda m: stribog_main(m, 256),
    'GOST512': lambda m: stribog_main(m, 512)
}

client_data = {
    'user_id': None,
    'secret': None,
    'hash_func_name': None,
    'hash_func': None,
    'attempt': 0,
    'host': '127.0.0.1',
    'port': 12345
}


def compute_hash_chain(secret: bytes, n: int, hash_func) -> bytes:
    result = secret
    for _ in range(n):
        result_str = hash_func(result)
        result = bytes.fromhex(result_str)
    return result


def register(user_id: str, secret: str, hash_func_name: str, n: int = 1000):
    if hash_func_name not in HASH_FUNCTIONS:
        print(f"Ошибка: Неподдерживаемая хеш-функция")
        return

    client_data['user_id'] = user_id
    client_data['secret'] = secret.encode('ascii')
    client_data['hash_func_name'] = hash_func_name
    client_data['hash_func'] = HASH_FUNCTIONS[hash_func_name]
    client_data['attempt'] = 0

    try:
        final_hash = compute_hash_chain(client_data['secret'], n, client_data['hash_func'])
    except ValueError as e:
        print(f"Ошибка при вычислении хэша: {e}")
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((client_data['host'], client_data['port']))
        data = {
            'action': 'register',
            'user_id': user_id,
            'hash_func': hash_func_name,
            'final_hash': final_hash,
            'max_attempts': n
        }
        s.sendall(pickle.dumps(data))
        response = pickle.loads(s.recv(1024))
        print(f"Регистрация: {response['message']}")


def authenticate():
    if not client_data['user_id'] or not client_data['secret']:
        print("Ошибка: Пользователь не зарегистрирован")
        return

    client_data['attempt'] += 1
    try:
        current_hash = compute_hash_chain(client_data['secret'], 1000 - client_data['attempt'], client_data['hash_func'])
    except ValueError as e:
        print(f"Ошибка при вычислении хэша: Неверный формат строки (возможно, не шестнадцатеричное значение): {e}")
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((client_data['host'], client_data['port']))
        data = {
            'action': 'authenticate',
            'user_id': client_data['user_id'],
            'attempt': client_data['attempt'],
            'hash': current_hash
        }
        s.sendall(pickle.dumps(data))
        response = pickle.loads(s.recv(1024))
        print(f"Аутентификация (попытка {client_data['attempt']}): {response['message']}")
        if not response['success']:
            print("Аутентификация не удалась")


def main():
    while True:
        print("\n1. Регистрация")
        print("2. Аутентификация")
        print("3. Выход")
        choice = input("Выберите действие: ")

        if choice == '1':
            user_id = input("Введите идентификатор пользователя: ")
            secret = input("Введите секретный пароль: ")
            print("Доступные хеш-функции:", ', '.join(HASH_FUNCTIONS.keys()))
            hash_func = input("Выберите хеш-функцию: ")
            register(user_id, secret, hash_func)

        elif choice == '2':
            authenticate()

        elif choice == '3':
            break


if __name__ == "__main__":
    main()