import socket
import pickle
import random

from crypta import sha512, sha256, stribog_main, save_pkcs8, save_pkcs12, generate_keypair, fast_exponentiation_mod, generate_prime, reverse, gcd, generate_keys

HASH_FUNCTIONS = {
    'SHA256': sha256,
    'SHA512': sha512,
    'GOST256': lambda m: stribog_main(m, 256),
    'GOST512': lambda m: stribog_main(m, 512)
}


def generate_keys_fiat_shamir(filename1, filename2, bit_length, m):
    p = generate_prime(bit_length // 2)
    q = generate_prime(bit_length // 2)
    n = p * q
    a = []
    b = []
    for _ in range(m):
        while True:
            a_i = random.randint(2, n - 1)
            if gcd(a_i, n) == 1:
                break
        s_i_square = fast_exponentiation_mod(a_i, 2, n)
        b_i = reverse(s_i_square, n)
        if b_i is None:
            continue
        a.append(a_i)
        b.append(b_i)

    public_key = (b, n)
    private_key = (a, p, q)
    with open(filename1, 'w') as f:
        f.write(f"b: {b}\n")
        f.write(f"n: {n}\n")
    with open(filename2, 'w') as f:
        f.write(f"s: {a}\n")
        f.write(f"p: {p}\n")
        f.write(f"q: {q}\n")
    return public_key, private_key


def generate_keys_cp(filename1, filename2, bit_length, algo, m):
    if algo == 'RSA':
        pub, priv = generate_keypair(bit_length)
        save_pkcs8(pub, filename1)
        save_pkcs12(priv, filename2)
        return pub, priv
    elif algo == 'ELGAMAL':
        pub, priv = generate_keys(filename1, filename2)
        return pub, priv
    elif algo == 'FIAT-SHAMIR':
        pub, priv = generate_keys_fiat_shamir(filename1, filename2, bit_length, m)
        return pub, priv
    else:
        raise ValueError(f"Неподдерживаемый алгоритм: {algo}")


def sign_message_fiat_shamir(message, private_key, hash_algo, m):
    a, p, q = private_key
    n = p * q
    hash_func = HASH_FUNCTIONS.get(hash_algo)

    r = random.randint(1, n - 1)

    u = fast_exponentiation_mod(r, 2, n)
    if isinstance(message, bytes):
        message = message.decode('utf-8')

    concat = message + str(u)
    concat = concat.encode('utf-8')
    s = hash_func(concat)
    s_bits = bin(int(s, 16))[2:].zfill(m)
    s = [int(bit) for bit in s_bits]

    t = r
    for i in range(len(s)):
        if s[i] == 1:
            t = (t * a[i]) % n

    return f"{hex(t)[2:]}:{':'.join(map(str, s))}"


def sign_message(message, private_key, hash_algo, algo, m):
    message = message.encode('utf-8')
    hash_func = HASH_FUNCTIONS.get(hash_algo)
    if not hash_func:
        raise ValueError(f"Неподдерживаемый алгоритм: {hash_algo}")
    message_hash = hash_func(message)
    if isinstance(message_hash, bytes):
        message_hash = message_hash.hex()

    if algo == 'RSA':
        d, n, p, q = private_key
        hash_int = int(message_hash, 16)
        return hex(pow(hash_int, d, n))[2:]
    elif algo == 'ELGAMAL':
        a = private_key
        p, alpha, _ = public_key
        h = int(message_hash, 16) % (p - 1)
        while True:
            k = random.randint(1, p - 2)
            if gcd(k, p - 1) == 1:
                break
        gamma = fast_exponentiation_mod(alpha, k, p)
        k_inv = reverse(k, p - 1)
        s = ((h - a * gamma) * k_inv) % (p - 1)
        return f"{hex(gamma)[2:]}:{hex(s)[2:]}"
    elif algo == 'FIAT-SHAMIR':
        return sign_message_fiat_shamir(message, private_key, hash_algo, m)
    else:
        raise ValueError(f"Неподдерживаемый алгоритм: {algo}")


def verify_signature_fiat_shamir(message, signature, public_key, hash_algo):
    try:
        b, n = public_key
        t_hex, s_str = signature.split(':', 1)
        t = int(t_hex, 16)
        s = [int(x) for x in s_str.split(':')]
        if len(s) != len(b):
            return False

        hash_func = HASH_FUNCTIONS.get(hash_algo)

        if hash_algo == 'SHA256' or hash_algo == 'GOST256':
            m = 256
        elif hash_algo == 'SHA512' or hash_algo == 'GOST512':
            m = 512

        w = fast_exponentiation_mod(t, 2, n)
        for i in range(m):
            if s[i] == 1:
                w = (w * b[i]) % n

        if isinstance(message, bytes):
            message = message.decode('utf-8')
        concat = message + str(w)
        concat = concat.encode('utf-8')

        s_prime = hash_func(concat)
        if isinstance(s_prime, bytes):
            s_prime = s_prime.hex()
        s_prime_bits = bin(int(s_prime, 16))[2:].zfill(m)
        s_prime = [int(bit) for bit in s_prime_bits]

        return s == s_prime
    except (ValueError, KeyError):
        return False


def verify_signature(message, signature, public_key, hash_algo, algo):
    try:
        message = message.encode('utf-8')
        hash_func = HASH_FUNCTIONS.get(hash_algo)
        if not hash_func:
            return False
        message_hash = hash_func(message)
        if isinstance(message_hash, bytes):
            message_hash = message_hash.hex()

        if algo == 'RSA':
            e, n = public_key
            hash_int = int(message_hash, 16)
            return hash_int == pow(int(signature, 16), e, n)
        elif algo == 'ELGAMAL':
            p, alpha, beta = public_key
            r, s = map(lambda x: int(x, 16), signature.split(':'))
            if r <= 0 or r >= p or s <= 0 or s >= p - 1:
                return False
            h = int(message_hash, 16) % (p - 1)
            return (fast_exponentiation_mod(beta, r, p) * fast_exponentiation_mod(r, s, p)) % p == fast_exponentiation_mod(alpha, h, p)
        elif algo == 'FIAT-SHAMIR':
            return verify_signature_fiat_shamir(message, signature, public_key, hash_algo)
        else:
            return False
    except (KeyError, ValueError):
        return False


def pkcs7_structure(message, hash_algo, signature, pub_key, algo):
    if algo == 'RSA':
        key_info = {"publicExponent": pub_key[0], "N": pub_key[1]}
    elif algo == 'ELGAMAL':
        key_info = {"alpha": pub_key[1], "beta": pub_key[2], "p": pub_key[0]}
    elif algo == 'FIAT-SHAMIR':
        b, n = pub_key
        key_info = {"b": b, "n": n}
    return {
        "CMSVersion": "1",
        "DigestAlgorithmIdentifiers": hash_algo,
        "EncapsulatedContentInfo": {
            "ContentType": "Data",
            "OCTET_STRING_OPTIONAL": message
        },
        "SignerInfos": {
            "CMSVersion": "1",
            "SignerIdentifier": "Maslenko AI",
            "DigestAlgorithmIdentifiers": hash_algo,
            "SignatureAlgorithmIdentifier": "RSAdsi" if algo == 'RSA' else "DSAdsi" if algo == 'ELGAMAL' else "FiatShamir",
            "SignatureValue": signature,
            "SubjectPublicKeyInfo": key_info,
            "UnsignedAttributes": {
                "ObjectIdentifier": "signature-time-stamp",
                "SET_OF_AttributeValue": {}
            }
        }
    }


def save_pkcs7_to_file(pkcs7, filename):

    def write_section(f, title, data):
        f.write(f"{title}:\n")
        for k, b in data.items():
            if k == 'b':
                f.write(f"{k}: [{', '.join(map(str, b))}]\n")
            else:
                f.write(f"{k}: {b}\n")

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"CMSVersion: {pkcs7['CMSVersion']}\n")
            f.write(f"DigestAlgorithmIdentifiers: {pkcs7['DigestAlgorithmIdentifiers']}\n\n")
            write_section(f, "EncapsulatedContentInfo", pkcs7['EncapsulatedContentInfo'])
            f.write("\nSignerInfos:\n")
            signer = pkcs7['SignerInfos']
            for k in ["CMSVersion", "SignerIdentifier", "DigestAlgorithmIdentifiers", "SignatureAlgorithmIdentifier",
                      "SignatureValue"]:
                f.write(f"{k}: {signer[k]}\n")
            write_section(f, "SubjectPublicKeyInfo", signer['SubjectPublicKeyInfo'])
            f.write("\nUnsignedAttributes:\nObjectIdentifier: signature-time-stamp\n")
            if signer['UnsignedAttributes']['SET_OF_AttributeValue']:
                timestamp = signer['UnsignedAttributes']['SET_OF_AttributeValue']
                f.write(f"\nTimestamp Data:\nStatus: {timestamp.get('status', 'N/A')}\n")
                if 'cades_t' in timestamp:
                    cades_t = timestamp['cades_t']
                    f.write(f"UTCTime: {cades_t['Timestamp']['UTCTime']}\n")
                    f.write(f"ServerTime: {cades_t['Timestamp']['ServerTime']}\n")
                    f.write(f"Signature: {cades_t['SignerInfos']['SignatureValue']}\n")
                    write_section(f, "TSA PublicKeyInfo", cades_t['SignerInfos']['SubjectPublicKeyInfo'])
        print(f"Структура сохранена в {filename}")
    except Exception as e:
        print(f"Ошибка сохранения: {str(e)}")


def deserialize_cades(data, message, signature, public_key, algo):
    try:
        timestamp = data['SignerInfos']['UnsignedAttributes']['SET_OF_AttributeValue']
        cades_t = timestamp.get('cades_t', {})
        hash_algo = data['DigestAlgorithmIdentifiers']
        is_valid = verify_signature(message, signature, public_key, hash_algo, algo)
        return f"""
Результат проверки подписи: {'Валидна' if is_valid else print() }
Хеш: {hash_algo}
Алгоритм подписи: {data['SignerInfos']['SignatureAlgorithmIdentifier']}
Автор: {data['SignerInfos']['SignerIdentifier']}
UTC время: {cades_t.get('Timestamp', {}).get('UTCTime', 'N/A')}
"""
    except KeyError as e:
        return f"Ошибка десериализации: отсутствует поле {e}"


def main_menu():
    global public_key, private_key, message, signature, pkcs7, hash_algo, sign_algo, m
    public_key = private_key = message = signature = pkcs7 = hash_algo = sign_algo = m = None

    while True:

        print("\nМеню:\n1. Сгенерировать ключи\n2. Сформировать подпись\n3. Проверить подпись\n4. Выход")
        choice = input("Выберите действие (1-4): ").strip()

        if choice == '1':
            sign_algo = input("\nАлгоритмы подписи: RSA, ELGAMAL, FIAT-SHAMIR\nВыберите: ").strip().upper()
            if sign_algo not in ['RSA', 'ELGAMAL', 'FIAT-SHAMIR']:
                print("Ошибка: неверный алгоритм")
                continue
            hash_algo = input(f"\nХеши: {', '.join(HASH_FUNCTIONS)}\nВыберите: ").strip()
            if hash_algo == 'SHA256' or hash_algo == 'GOST256':
                m = 256
            elif hash_algo == 'SHA512' or hash_algo == 'GOST512':
                m = 512
            public_key, private_key = generate_keys_cp('public_key.txt', 'private_key.txt', 1024, sign_algo, m)
            print("Ключи сохранены в public_key.txt и private_key.txt")


        elif choice == '2':

            if not public_key or not private_key:
                print("Ошибка: сгенерируйте ключи")
                continue
            hash_algo = input(f"\nХеши: {', '.join(HASH_FUNCTIONS)}\nВыберите: ").strip()
            if hash_algo not in HASH_FUNCTIONS:
                print("Ошибка: неверный хеш")
                continue

            message = input("Сообщение: ").strip()
            if not message:
                print("Ошибка: пустое сообщение")
                continue


            signature = sign_message(message, private_key, hash_algo, sign_algo, m)
            pkcs7 = pkcs7_structure(message, hash_algo, signature, public_key, sign_algo)
            save_pkcs7_to_file(pkcs7, "save.txt")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect(("localhost", 5001))
                    s.sendall(pickle.dumps({'action': 'get_timestamp', 'data': pkcs7}))
                    response = pickle.loads(s.recv(10 * 1024 * 1024))
                    if response.get('status') == 'error':
                        print("Ошибка сервера:", response.get('error', 'Нет CAdES-T'))
                        continue
                    print("TSA ответ:", response)
                    pkcs7['SignerInfos']['UnsignedAttributes']['SET_OF_AttributeValue'] = response
                    save_pkcs7_to_file(pkcs7, "text.txt")
                except Exception as e:
                    print(f"Ошибка TSA: {e}")

        elif choice == '3':
            if not all([pkcs7, message, signature, public_key]):
                print("Ошибка: сформируйте подпись")
                continue

            signature1 = pkcs7['SignerInfos']['SignatureValue']
            message = pkcs7['EncapsulatedContentInfo']['OCTET_STRING_OPTIONAL']
            print("Проверка подписи:", deserialize_cades(pkcs7, message, signature1, public_key, sign_algo))

        elif choice == '4':
            print("Выход")
            break

        else:
            print("Неверный выбор")


if __name__ == "__main__":
    main_menu()