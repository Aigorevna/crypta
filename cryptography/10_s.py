import socket
import pickle
import random
from datetime import datetime, UTC
from crypta import sha512, sha256, stribog_main, generate_keypair

HASH_FUNCTIONS = {
    'SHA256': sha256,
    'SHA512': sha512,
    'GOST256': lambda m: stribog_main(m, 256),
    'GOST512': lambda m: stribog_main(m, 512)
}

def sign_message_rsa(message,private_key, hash_algo):

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


PARAMS_FILE = "10_param.txt"


def _load_params(file_path: str):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return {
            'p': int(lines[0].split(": ")[1].strip()),
            'alpha': int(lines[2].split(": ")[1].strip()),
            'L': int(lines[8].split(": ")[1].strip())
        }
    except (IndexError, ValueError, FileNotFoundError) as e:
        raise ValueError(f"Failed to load parameters from {file_path}: {e}")


def _extract_signature_data(data):
    try:
        return {
            'message': data["EncapsulatedContentInfo"]["OCTET_STRING_OPTIONAL"],
            'U': data["SignerInfos"]["SignatureValue"]["U"],
            'E': data["SignerInfos"]["SignatureValue"]["E"],
            'S': data["SignerInfos"]["SignatureValue"]["S"],
            'hash_type': data["DigestAlgorithmIdentifiers"]
        }
    except KeyError as e:
        raise ValueError(f"Invalid data structure, missing key: {e}")


def _compute_r(p, U, L, E, alpha, S):
    return (pow(U * L, -E, p) * pow(alpha, S, p)) % p


def _to_hex(value):
    return hex(value)[2:]


def _compute_e(message, r_hex, u_hex, hash_func):
    data = (message + r_hex + u_hex).encode('utf-8')
    return int(hash_func(data), 16)


def validate_signature(data):
    try:
        signature_data = _extract_signature_data(data)
        message = signature_data['message']
        U = signature_data['U']
        E = signature_data['E']
        S = signature_data['S']
        hash_type = signature_data['hash_type']

        hash_func = HASH_FUNCTIONS.get(hash_type)
        if not hash_func:
            raise ValueError(f"Unsupported hash algorithm: {hash_type}")

        params = _load_params(PARAMS_FILE)
        p = params['p']
        alpha = params['alpha']
        L = params['L']

        r_computed = _compute_r(p, U, L, E, alpha, S)
        r_hex = _to_hex(r_computed)
        u_hex = _to_hex(U)

        e_computed = _compute_e(message, r_hex, u_hex, hash_func)

        return e_computed == E

    except ValueError as e:
        print(f"Signature validation failed: {e}")
        return False

def create_timestamp():
    utc_time = datetime.now(UTC).strftime("%y%m%d%H%M%SZ")
    server_time = datetime.now(UTC).strftime("%y%m%d%H%M%SZ")
    return utc_time, server_time


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 1339))
server.listen(1)
print("Сервер штампов времени слушает на порту 1339...")


while True:
    conn, addr = server.accept()
    print(f"Подключился клиент: {addr}")

    try:
        received_data = pickle.loads(conn.recv(4096))

        message_hex = received_data["EncapsulatedContentInfo"]["OCTET_STRING_OPTIONAL"]
        U = received_data["SignerInfos"]["SignatureValue"]["U"]
        E = received_data["SignerInfos"]["SignatureValue"]["E"]
        S = received_data["SignerInfos"]["SignatureValue"]["S"]
        hash_algo = received_data["DigestAlgorithmIdentifiers"]

        message_bytes = bytes.fromhex(message_hex)
        message = message_bytes.decode("utf-8")

        if validate_signature(received_data):
            print("Проверка прошла")
        else:
            print("Проверка не пройдена")
            conn.send(pickle.dumps({'status': 'error', 'error': 'Неверный запрос или подпись'}))
            conn.close()
            continue

        utc_time, server_time = create_timestamp()
        public_key, private_key = generate_keypair(1024)

        hash_func = HASH_FUNCTIONS.get(hash_algo)

        #new_message = (хеш(messagetsa + signaturetsa)+ время)
        DataWithHash = str(message +  str(U) + str(E) + str(S))
        DataWithHash = DataWithHash.encode('utf-8')

        DataWithHash = hash_func (DataWithHash)
        for_sign = (DataWithHash + utc_time[:-1])

        TimestampCenterSignature = sign_message_rsa(for_sign, private_key, hash_algo)

        cades_t_structure = {
            "CMSVersion": "1",
            "DigestAlgorithmIdentifiers": hash_algo,
            "EncapsulatedContentInfo": {
                "ContentType": "Data",
                "OCTET_STRING_OPTIONAL": message
            },
            "SignerInfos": {
                "CMSVersion": "1",
                "SignerIdentifier": "Maslenko A I ",
                "DigestAlgorithmIdentifiers": hash_algo,
                "SignatureAlgorithmIdentifier": "RSAdsi",
                "SignatureValue": {
                    "U": U,
                    "E": E,
                    "S": S,
                    "key": public_key
                },
                "UnsignedAttributes": {
                    "ObjectIdentifier": "signature-time-stamp",
                    "SET_OF_AttributeValue": {
                        "DataWithHash": DataWithHash,
                        "Timestamp": {
                            "UTCTime": utc_time,
                            "GeneralizedTime": server_time
                        },
                        "TimestampCenterSignature": TimestampCenterSignature
                    }
                }
            }
        }

        with open("CAdes-T_Server.txt", "wb") as f:
            pickle.dump(cades_t_structure, f)

        conn.send(pickle.dumps(cades_t_structure))
        print("CAdES-T отправлен клиенту.")
        conn.close()
        print("Соединение закрыто.")

    except Exception as e:
        conn.send(pickle.dumps({'status': 'error', 'error': f'Ошибка обработки: {str(e)}'}))
