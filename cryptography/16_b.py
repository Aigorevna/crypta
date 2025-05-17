import socket
import pickle
from crypta import sha512, generate_keypair, decrypt, save_pkcs8, save_pkcs12

IDENTIFIER2 = 'User2'

def validate_signature_rsa(data):
    try:
        message = data['message']
        pub_exp = data['public_key'][0]
        n = data['public_key'][1]
        sig = data['signature']

        hash_func = sha512
        hash_user = hash_func(message.encode('utf-8'))
        if isinstance(hash_user, bytes):
            hash_user = hash_user.hex()
        verified_hash = hex(pow(int(sig, 16), pub_exp, n))[2:]
        return verified_hash == hash_user
    except (KeyError, ValueError):
        return False


def main():
    bit_length = 512
    public_key_b, private_key_b = generate_keypair(bit_length)
    save_pkcs8(public_key_b, 'public_key_b.txt')
    save_pkcs12(private_key_b, 'private_key_b.txt')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 5001))
        s.listen(1)
        print("Ожидание подключения...")
        conn, addr = s.accept()
        with open('public_key_a.txt', 'r') as f:
            lines = f.readlines()
            e_a = int(lines[0].split(': ')[1])
            n_a = int(lines[1].split(': ')[1])
        public_key_a = (e_a, n_a)
        with conn:
            data = pickle.loads(conn.recv(4096))
            ciphertext = data['ciphertext']

    block_size = 128
    message = decrypt(private_key_b, ciphertext, block_size)
    session_key = message[:64]
    time = message[64:77]
    signature_hex = message[77:]

    message1 = IDENTIFIER2 + session_key + time

    signature_data = {
        'message': message1,
        'public_key': public_key_a,
        'signature': signature_hex
    }
    if validate_signature_rsa(signature_data):
        print("Подпись верна, все прошло успешно ")
        print(time)
        with open("text.txt", 'w') as f:
            f.write(f"Session_key: {session_key}\n")
    else:
        print("Signature verification failed")

if __name__ == "__main__":
    main()