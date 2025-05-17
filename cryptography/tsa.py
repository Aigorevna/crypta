import socket
import pickle
import random
from datetime import datetime, UTC
from crypta import sha512, sha256, stribog_main, generate_keypair, fast_exponentiation_mod, generate_prime, save_pkcs8, \
    save_pkcs12, generate_keys, reverse, gcd

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
        s_i_square = (a_i * a_i) % n
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


def generate_key(sign_algo, filename1, filename2, bit_length, m):
    global public_key, private_key
    if sign_algo == 'RSAdsi':
        public_key, private_key = generate_keypair(bit_length)
        save_pkcs8(public_key, filename1)
        save_pkcs12(private_key, filename2)
    elif sign_algo == 'DSAdsi':
        public_key, private_key = generate_keys(filename1, filename2)
    elif sign_algo == 'FiatShamir':
        public_key, private_key = generate_keys_fiat_shamir(filename1, filename2, bit_length, m)
    else:
        raise ValueError("Неподдерживаемый алгоритм подписи")

    return public_key, private_key


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


def sign_message_rsa(message, private_key ,hash_algo):
    d, n, p, q = private_key
    message = message.encode('utf-8')

    hash_func = HASH_FUNCTIONS.get(hash_algo)
    if not hash_func:
        raise ValueError(f"Неподдерживаемый алгоритм: {hash_algo}")
    message_hash = hash_func(message)
    if isinstance(message_hash, bytes):
        message_hash = message_hash.hex()
    hash_int = int(message_hash, 16)

    signature = hex(pow(hash_int, d, n))[2:]
    return signature


def sign_message_elgamal(message, private_key, hash_algo):
    try:
        a = private_key
        p, alpha, _ = public_key
    except ValueError:
        raise ValueError("Неверный формат ключей для ElGamal")

    message = message.encode('utf-8')
    hash_func = HASH_FUNCTIONS.get(hash_algo)
    if not hash_func:
        raise ValueError(f"Неподдерживаемый алгоритм: {hash_algo}")
    message_hash = hash_func(message)
    if isinstance(message_hash, bytes):
        message_hash = message_hash.hex()
    h = int(message_hash, 16) % (p - 1)

    while True:
        k = random.randint(1, p - 2)
        if gcd(k, p - 1) == 1:
            break

    r = fast_exponentiation_mod(alpha, k, p)
    k_inv = reverse(k, p - 1)
    s = ((h - a * r) * k_inv) % (p - 1)
    if s == 0:
        raise ValueError("Недопустимая подпись: s равно нулю")

    return f"{hex(r)[2:]}:{hex(s)[2:]}"


def sign_message(sign_algo, message, hash_algo, m):
    if sign_algo == 'RSAdsi':
        return sign_message_rsa(message, private_key, hash_algo)
    elif sign_algo == 'DSAdsi':
        return sign_message_elgamal(message, private_key, hash_algo)
    elif sign_algo == 'FiatShamir':
        return sign_message_fiat_shamir(message, private_key, hash_algo, m)
    else:
        raise ValueError("Алгоритм подписи не установлен")


def validate_signature_fiat_shamir(data):
    try:
        message = data['EncapsulatedContentInfo']['OCTET_STRING_OPTIONAL']
        hash_algo = data['DigestAlgorithmIdentifiers']
        b = data['SignerInfos']['SubjectPublicKeyInfo']['b']
        n = data['SignerInfos']['SubjectPublicKeyInfo']['n']
        signature = data['SignerInfos']['SignatureValue']

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
        for i in range(len(b)):
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
    except (KeyError, ValueError):
        return False


def validate_signature_rsa(data):
    try:
        message = data['EncapsulatedContentInfo']['OCTET_STRING_OPTIONAL']
        hash_type = data['DigestAlgorithmIdentifiers']
        pub_exp = data['SignerInfos']['SubjectPublicKeyInfo']['publicExponent']
        n = data['SignerInfos']['SubjectPublicKeyInfo']['N']
        sig = data['SignerInfos']['SignatureValue']

        hash_func = HASH_FUNCTIONS.get(hash_type)
        if not hash_func:
            return False
        hash_user = hash_func(message.encode('utf-8'))
        if isinstance(hash_user, bytes):
            hash_user = hash_user.hex()
        verified_hash = hex(pow(int(sig, 16), pub_exp, n))[2:]
        return verified_hash == hash_user
    except (KeyError, ValueError):
        return False


def validate_signature_elgamal(data):
    try:
        message = data['EncapsulatedContentInfo']['OCTET_STRING_OPTIONAL']
        hash_type = data['DigestAlgorithmIdentifiers']
        alpha = data['SignerInfos']['SubjectPublicKeyInfo']['alpha']
        beta = data['SignerInfos']['SubjectPublicKeyInfo']['beta']
        p = data['SignerInfos']['SubjectPublicKeyInfo']['p']
        sig = data['SignerInfos']['SignatureValue']
        r, s = map(lambda x: int(x, 16), sig.split(':'))

        hash_func = HASH_FUNCTIONS.get(hash_type)
        if not hash_func:
            return False
        message_hash = hash_func(message.encode('utf-8'))
        if isinstance(message_hash, bytes):
            message_hash = message_hash.hex()
        h = int(message_hash, 16) % (p - 1)

        left = (fast_exponentiation_mod(beta, r, p) * fast_exponentiation_mod(r, s, p)) % p
        right = fast_exponentiation_mod(alpha, h, p)
        return left == right
    except (KeyError, ValueError):
        return False


def validate_signature(data):
    try:
        algo = data['SignerInfos']['SignatureAlgorithmIdentifier']
        if algo == 'RSAdsi':
            return validate_signature_rsa(data)
        elif algo == 'DSAdsi':
            return validate_signature_elgamal(data)
        elif algo == 'FiatShamir':
            return validate_signature_fiat_shamir(data)
        return False
    except KeyError:
        return False


def create_timestamp():
    utc_time = datetime.now(UTC).strftime("%y%m%d%H%M%SZ")
    server_time = datetime.now(UTC).strftime("%y%m%d%H%M%SZ")
    return utc_time, server_time


def create_cades_t_structure(sign_algo, message, hash_algo):
    utc_time, server_time = create_timestamp()
    timestamp_data = f"{utc_time}:{server_time}"
    signature = message + utc_time[:-1]
    signature = str(signature)
    if hash_algo == 'SHA256' or hash_algo == 'GOST256':
        m = 256
    elif hash_algo == 'SHA512' or hash_algo == 'GOST512':
        m = 512
    signature = sign_message(sign_algo, signature, hash_algo, m)

    signed_data = {
        'version': 'v1',
        'DigestAlgorithmIdentifiers': hash_algo,
        'EncapsulatedContentInfo': {
            'contentType': 'timestamp',
            'OCTET_STRING_OPTIONAL': timestamp_data
        },
        'SignerInfos': {
            'version': 'v1',
            'SubjectPublicKeyInfo': (
                {'algorithm': 'rsa', 'publicExponent': public_key[1], 'N': public_key[0]}
                if sign_algo == 'RSAdsi' else
                {'algorithm': 'elgamal', 'alpha': public_key[1], 'beta': public_key[2], 'p': public_key[0]}
                if sign_algo == 'DSAdsi' else
                {'algorithm': 'fiat-shamir', 'v': public_key[0], 'N': public_key[1]}
            ),
            'SignatureAlgorithmIdentifier': (
                'RSAdsi' if sign_algo == 'RSA' else
                'DSAdsi' if sign_algo == 'ELGAMAL' else
                'FiatShamir'
            ),
            'SignatureValue': signature
        },
        'Timestamp': {
            'UTCTime': utc_time,
            'ServerTime': server_time
        }
    }
    return signed_data


def process_request(request):
    if 'data' not in request or not validate_signature(request['data']):
        return {'status': 'error', 'error': 'Неверный запрос или подпись'}

    try:
        hash_algo = request['data']['DigestAlgorithmIdentifiers']
        messagetsa = request['data']['EncapsulatedContentInfo']['OCTET_STRING_OPTIONAL']
        signaturetsa = request['data']['SignerInfos']['SignatureValue']
        sign_algo = request['data']['SignerInfos']['SignatureAlgorithmIdentifier']
        if hash_algo == 'SHA256' or hash_algo == 'GOST256':
            m = 256
        elif hash_algo == 'SHA512' or hash_algo == 'GOST512':
            m = 512
        generate_key(sign_algo, "public_keytsa.txt", "private_keytsa.txt", 1024, m)
        new_message = messagetsa + signaturetsa
        cades_t_response = create_cades_t_structure(sign_algo, new_message, hash_algo)
        return {
            'status': 'success',
            'cades_t': cades_t_response,
            'public_key': public_key
        }
    except KeyError:
        return {'status': 'error', 'error': 'Неверная структура данных'}



def run_server(host='localhost', port=5001):

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            sock.listen(1)
            print(f"Сервер TSA запущен на {host}:{port} с алгоритмом ")

            while True:
                conn, addr = sock.accept()
                print(f"Подключение: {addr}")
                with conn:
                    try:
                        data = pickle.loads(conn.recv(10 * 1024 * 1024))
                        if not isinstance(data, dict) or data.get('action') != 'get_timestamp':
                            response = {'status': 'error', 'error': 'Неверный запрос'}
                        else:
                            response = process_request(data)
                        conn.sendall(pickle.dumps(response))
                    except Exception as e:
                        conn.sendall(pickle.dumps({'status': 'error', 'error': str(e)}))
                    print("Ответ отправлен")
    except OSError as e:
        print(f"Ошибка сокета: {e}")


if __name__ == '__main__':
    run_server()