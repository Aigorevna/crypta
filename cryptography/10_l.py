import socket
import random
import os
import pickle

from crypta import sha512, sha256, stribog_main, fast_exponentiation_mod, generate_prime, miller_rabin, gcd, reverse, evklid

HASH_FUNCTIONS = {
    'SHA256': sha256,
    'SHA512': sha512,
    'GOST256': lambda m: stribog_main(m, 256),
    'GOST512': lambda m: stribog_main(m, 512)
}


def generate_p_q():
    bits = 1024
    while True:
        q = generate_prime(bits)
        p = 2 * q + 1
        if miller_rabin(p):
            if 0.547 <= q <= (p - 1) / 2:
                return p, q


def generate_alpha(p, q):
    while True:
        alpha = random.randint(2, p - 2)
        if fast_exponentiation_mod(alpha, q, p) == 1:
            is_order_q = True
            for k in range(1, 100):
                if fast_exponentiation_mod(alpha, k, p) == 1:
                    is_order_q = False
                    break
            if is_order_q:
                return alpha
    return None


def generate_group_signature_params(g):
    p, q = generate_p_q()
    print(f"p: {p}\nq: {q}")

    alpha = generate_alpha(p, q)
    print(f"alpha: {alpha}")

    group_members = []
    for j in range(g):
        k_j = random.randint(1, q - 1)
        P_j = fast_exponentiation_mod(alpha, k_j, p)
        group_members.append((k_j, P_j))
        print(f"Member {j + 1}: Secret key k_{j + 1} = {k_j}, Public key P_{j + 1} = {P_j}")

    z = random.randint(1, q - 1)
    L = fast_exponentiation_mod(alpha, z, p)
    print(f"Leader: Secret key z = {z}, Public key L = {L}")



    bits = 1024
    p1 = generate_prime(bits)
    p2 = generate_prime(bits)

    n = p1 * p2
    phi = (p1 - 1) * (p2 - 1)

    e = random.randint(1, phi)
    g, _, _ = evklid(e, phi)
    while g != 1:
        e = random.randint(1, phi)
        g, _, _ = evklid(e, phi)
    d = reverse(e, phi)
    print(f"RSA parameters: n = {n}, e = {e}, d = {d}")

    return p, q, alpha, group_members, z, L, e, d, n


def _compute_deltas(message, hash_algo, params):

    p, q, alpha, group_members, z, L, e, d, n = params
    k_1, P_1, k_2, P_2 = group_members[0][0], group_members[0][1], group_members[1][0], group_members[1][1]

    message_bytes = message.encode('utf-8')
    hash_func = HASH_FUNCTIONS.get(hash_algo)
    H = hash_func(message_bytes)
    H_int = int(H, 16)

    delta_1 = fast_exponentiation_mod(H_int + P_1, d, n)
    delta_2 = fast_exponentiation_mod(H_int + P_2, d, n)
    return delta_1, delta_2

def _compute_u(P_1, delta_1, P_2, delta_2, p):
    return (fast_exponentiation_mod(P_1, delta_1, p) * fast_exponentiation_mod(P_2, delta_2, p)) % p

def _send_and_receive(socket, data, action):
    try:
        socket.send(pickle.dumps(data))
        return pickle.loads(socket.recv(8192))
    except (ConnectionError, pickle.PickleError) as e:
        raise RuntimeError(f"Socket communication failed during {action}: {e}")

def _compute_e(message, R, U, hash_func):

    M = str(message.encode('utf-8').hex())
    R_hex = str(hex(R)[2:])
    U_hex = str(hex(U)[2:])
    E = (M + R_hex + U_hex).encode('utf-8')
    E = hash_func(E)
    E = int(E, 16)

    return E

def _verify_signatures(R_1, S_a, P_1, delta_1, E, R_2, S_b, P_2, delta_2, alpha, p):

    check_a = R_1 % p == (pow(P_1, -(delta_1 * E), p) * pow(alpha, S_a, p)) % p
    check_b = R_2 % p == (pow(P_2, -(delta_2 * E), p) * pow(alpha, S_b, p)) % p
    return check_a and check_b


def sign_message(message, hash_algo, params, socket_a, socket_b):

    if not message or hash_algo not in HASH_FUNCTIONS:
        raise ValueError(f"Invalid message or hash algorithm: {hash_algo}")

    p, q, alpha, group_members, z, L, e, d, n = params
    if len(group_members) < 2:
        raise ValueError("At least two group members are required")

    hash_func = HASH_FUNCTIONS.get(hash_algo)

    delta_1, delta_2 = _compute_deltas(message, hash_algo, params)

    R_1 = _send_and_receive(socket_a, delta_1, "отправка delta_1 A, получение R_1")
    R_2 = _send_and_receive(socket_b, delta_2, "отправка delta_2 B, получение R_2")

    _, P_1, _, P_2 = group_members[0][0], group_members[0][1], group_members[1][0], group_members[1][1]

    U = _compute_u(P_1, delta_1, P_2, delta_2, p)

    T = random.randint(1, q)
    R_l = pow(alpha, T, p)
    R = (R_l * R_1 * R_2) % p

    E = _compute_e(message, R, U, hash_func)

    S_a = _send_and_receive(socket_a, E, "отправка E A, получение S_a")
    S_b = _send_and_receive(socket_b, E, "отправка E B, получение S_b")

    if not _verify_signatures(R_1, S_a, P_1, delta_1, E, R_2, S_b, P_2, delta_2, alpha, p):
        socket_a.close()
        socket_b.close()
        raise ValueError("Проверка не пройдена")

    S_hatch = (T + z * E) % q
    S = (S_hatch + S_a + S_b) % q

    return U, E, S

def pkcs7_structure(message, hash_algo, signature):
    U, E, S = signature
    message_encoded = message.encode("utf-8").hex()

    return {
        "CMSVersion": "1",
        "DigestAlgorithmIdentifiers": hash_algo,
        "EncapsulatedContentInfo": {
            "ContentType": "Data",
            "OCTET_STRING_OPTIONAL": message_encoded
        },
        "SignerInfos": {
            "CMSVersion": "1",
            "SignerIdentifier": "Maslenko AI",
            "DigestAlgorithmIdentifiers": hash_algo,
            "SignatureAlgorithmIdentifier": "Groupdsi",
            "SignatureValue": {
                "U": U,
                "E": E,
                "S": S
            },
            "UnsignedAttributes": {
                "ObjectIdentifier": "signature-time-stamp",
                "SET_OF_AttributeValue": {}
            }
        }
    }


def verify_server_signature(data):
    try:
        timestamp = data['SignerInfos']['UnsignedAttributes']['SET_OF_AttributeValue']
        utc_time = timestamp["Timestamp"]["UTCTime"]
        DataWithHash = timestamp["DataWithHash"]
        public_key = data["SignerInfos"]["SignatureValue"]["key"]
        TimestampCenterSignature = data["SignerInfos"]["UnsignedAttributes"]["SET_OF_AttributeValue"]["TimestampCenterSignature"]

        hash_algo = data["DigestAlgorithmIdentifiers"]

        DUG = (DataWithHash + utc_time[:-1]).encode('utf-8')
        hash_func = HASH_FUNCTIONS.get(hash_algo)
        DUG_hash = hash_func(DUG)
        e, n = public_key

        hash_int = int(DUG_hash, 16)
        signature_int = int(TimestampCenterSignature, 16)

        if hash_int >= n:
            print("Ошибка: хеш превышает модуль RSA!")
            return False

        if hash_int == pow(signature_int, e, n):
            print("Подпись сервера действительна!")
            return True
        else:
            print("Подпись сервера недействительна!")
            return False

    except (ValueError, KeyError) as e:
        print(f"Ошибка проверки подписи: {str(e)}")
        return False

def deserialize_cades(data):
    try:
        timestamp = data['SignerInfos']['UnsignedAttributes']['SET_OF_AttributeValue']
        utc_time = timestamp["Timestamp"]["UTCTime"]
        hash_algo = data["DigestAlgorithmIdentifiers"]
        is_valid = verify_server_signature(data)
        return f"""
Результат проверки подписи: {'Валидна' if is_valid else print() }
Хеш: {hash_algo}
Алгоритм подписи: {data['SignerInfos']['SignatureAlgorithmIdentifier']}
Автор: {data['SignerInfos']['SignerIdentifier']}
UTC время: {utc_time}
"""
    except KeyError as e:
        return f"Ошибка десериализации: отсутствует поле {e}"




if __name__ == "__main__":

    g = 2
    L_To_A = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    L_To_B = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    L_To_C = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    params = None

    while True:
        print("\nМеню:\n1. Сгенерировать ключи\n2. Сформировать подпись\n3. Проверить подпись\n4. Выход")
        choice = input("Выберите действие (1-4): ").strip()

        if choice == "1":
            filename1 = "10_param.txt"
            filename2 = "10_l.txt"
            if os.path.exists(filename1) and os.path.getsize(filename1) > 0 and os.path.exists(
                    filename2) and os.path.getsize(filename2) > 0:
                try:
                    with open(filename1, 'r') as f:
                        lines = f.readlines()
                        p = int(lines[0].split(": ")[1].strip())
                        q = int(lines[1].split(": ")[1].strip())
                        alpha = int(lines[2].split(": ")[1].strip())
                        group_members = []
                        for i in range(g):
                            k_line = lines[3 + 2 * i].split(": ")[1].strip()
                            P_line = lines[4 + 2 * i].split(": ")[1].strip()
                            k_j = int(k_line)
                            P_j = int(P_line)
                            group_members.append((k_j, P_j))
                        z = int(lines[3 + 2 * g].split(": ")[1].strip())
                        L = int(lines[4 + 2 * g].split(": ")[1].strip())

                    with open(filename2, 'r') as f:
                        lines = f.readlines()
                        e = int(lines[0].split(": ")[1].strip())
                        d = int(lines[1].split(": ")[1].strip())
                        n = int(lines[2].split(": ")[1].strip())

                    params = (p, q, alpha, group_members, z, L, e, d, n)
                    print("Параметры загружены из файлов 10_param.txt и 10_l.txt:")
                    print(f"p: {p}\nq: {q}")
                    print(f"alpha: {alpha}")
                    for j, (k_j, P_j) in enumerate(group_members):
                        print(f"Member {j + 1}: Secret key k_{j + 1} = {k_j}, Public key P_{j + 1} = {P_j}")
                    print(f"Leader: Secret key z = {z}, Public key L = {L}")
                    print(f"RSA parameters: n = {n}, e = {e}, d = {d}")

                except (ValueError, IndexError) as e:
                    print(f"Ошибка при чтении файлов {filename1} или {filename2}: {e}. Генерируются новые параметры.")
                    params = generate_group_signature_params(g)
                    p, q, alpha, group_members, z, L, e, d, n = params

                    with open(filename1, 'w') as f:
                        f.write(f"p: {p}\n")
                        f.write(f"q: {q}\n")
                        f.write(f"alpha: {alpha}\n")
                        for j, (k_j, P_j) in enumerate(group_members):
                            f.write(f"Member {j + 1} k_{j + 1}: {k_j}\n")
                            f.write(f"Member {j + 1} P_{j + 1}: {P_j}\n")
                        f.write(f"z: {z}\n")
                        f.write(f"L: {L}\n")

                    with open(filename2, 'w') as f:
                        f.write(f"e: {e}\n")
                        f.write(f"d: {d}\n")
                        f.write(f"n: {n}\n")

                    print(f"Параметры сгенерированы и сохранены в {filename1} и {filename2}")
            else:
                params = generate_group_signature_params(g)
                p, q, alpha, group_members, z, L, e, d, n = params

                with open(filename1, 'w') as f:
                    f.write(f"p: {p}\n")
                    f.write(f"q: {q}\n")
                    f.write(f"alpha: {alpha}\n")
                    for j, (k_j, P_j) in enumerate(group_members):
                        f.write(f"Member {j + 1} k_{j + 1}: {k_j}\n")
                        f.write(f"Member {j + 1} P_{j + 1}: {P_j}\n")
                        f.write(f"z: {z}\n")
                        f.write(f"L: {L}\n")

                with open(filename2, 'w') as f:
                    f.write(f"e: {e}\n")
                    f.write(f"d: {d}\n")
                    f.write(f"n: {n}\n")

                print(f"Параметры сгенерированы и сохранены в {filename1} и {filename2}")

        elif choice == "2":
            try:
                L_To_A.connect(("localhost", 1337))
                print("Успешно подключено к А на порту 1337")
                L_To_B.connect(("localhost", 1338))
                print("Успешно подключено к Б на порту 1338")
                L_To_C.connect(("localhost", 1339))
                print("Успешно подключено к серверу на порту 1339")

                if params is None:
                    print("Ошибка: параметры не сгенерированы. Выполните действие 1 сначала.")
                    L_To_A.close()
                    L_To_B.close()
                    L_To_C.close()
                    exit()

                hash_algo = input(f"\nХеши: {', '.join(HASH_FUNCTIONS)}\nВыберите: ").strip()
                if hash_algo not in HASH_FUNCTIONS:
                    print(f"Ошибка: неподдерживаемый алгоритм хеширования {hash_algo}!")
                    L_To_A.close()
                    L_To_B.close()
                    L_To_C.close()
                    exit()

                message = input("Сообщение: ").strip()
                signature = sign_message(message, hash_algo, params, L_To_A, L_To_B)
                pkcs7 = pkcs7_structure(message, hash_algo, signature)
                print("Отправленная структура:", pkcs7)
                L_To_C.send(pickle.dumps(pkcs7))

                response = pickle.loads(L_To_C.recv(4096))
                if response.get('status') == 'error':
                    print(f"Ошибка от сервера: {response.get('error')}")
                else:
                    print("Полученная структура от сервера:", response)

            except ConnectionRefusedError as e:
                print(f"Не удалось подключиться к одному из участников: {e}")
                L_To_A.close()
                L_To_B.close()
                L_To_C.close()
                exit()
            except Exception as e:
               print(f"Произошла ошибка: {e}")
               L_To_A.close()
               L_To_B.close()
               L_To_C.close()
               exit()

        elif choice == "3":
            try:

                if params is None:
                    print("Ошибка: параметры не сгенерированы. Выполните действие 1 сначала.")
                    L_To_C.close()
                    exit()

                with open("CAdES-T_Server.txt", "rb") as f:
                    cades_t_data = pickle.load(f)

                print("Проверка подписи сервера:", deserialize_cades(cades_t_data))


            except FileNotFoundError:
                print("Ошибка: файл CAdES-T_Server не найден. Сначала выполните действие 2.")
            except ConnectionRefusedError as e:
                print(f"Не удалось подключиться к серверу: {e}")
            except Exception as e:
                print(f"Произошла ошибка: {e}")
            finally:
                L_To_C.close()

        elif choice == "4":
            print("Выход из программы.")
            L_To_A.close()
            L_To_B.close()
            L_To_C.close()
            exit()
        else:
            print("Некорректный выбор. Пожалуйста, выберите число от 1 до 4.")
            L_To_A.close()
            L_To_B.close()
            L_To_C.close()