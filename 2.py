import random

def evklid(x, y):  # 1.1

    a_2, a_1 = 1, 0
    b_2, b_1 = 0, 1

    while y != 0:
        q = x // y
        r = x - q * y
        a = a_2 - q * a_1
        b = b_2 - q * b_1

        x = y
        y = r
        a_2 = a_1
        a_1 = a
        b_2 = b_1
        b_1 = b

    m = x
    a = a_2
    b = b_2
    return m, a, b


def reverse(s, m):  # поиск обратного элемента
    c, a, b = evklid(m, s)

    if c != 1:
        print("Обратного элемента не существует")
        return None
    else:
        rev = b % m  # мод
        return rev

def fast_exponentiation_mod(a, s, m):

    x = 1

    while s > 0:
        if s & 1:
            x = (x * a) % m
        a = (a * a) % m
        s = s >> 1

    return x

def miller_rabin(n): #1.5
    if (n < 5) or (n % 2 == 0):
        return False
    k=5
    s = 0
    r = n - 1
    while r % 2 == 0:
        r //= 2
        s += 1

    for _ in range(k):
        a = random.randint(2, n - 2)
        y = fast_exponentiation_mod(a, r, n)
        if y != 1 and y != n - 1:
            j = 1
            while j < s and y != n - 1:
                y = fast_exponentiation_mod(y,2,n)
                if y == 1:
                    return False
                j += 1
            if y != n - 1:
                return False

    return True
def generate_prime(k, t=100): #1.6

    while True:

        p = random.getrandbits(k)
        p |= (1 << (k - 1)) | 1
        test = True
        for _ in range(t):
            if not miller_rabin(p):
                test = False
                break

        if test:
            return p

def generate_keypair(bit_length):
    p = generate_prime(bit_length)
    q = generate_prime(bit_length)
    n = p * q
    N = (p - 1) * (q - 1) #фия эйлера
    e = random.randint(1, N)
    g, _, _ = evklid(e, N)
    while g != 1:
        e = random.randint(1, N)
        g, _, _ = evklid(e, N)
    d = reverse(e, N)
    return ((e, n), (d, n, p, q))

def pkcs7_pad(data, block_size):
    pad_len = block_size - (len(data) % block_size)
    padding = bytes([pad_len] * pad_len)
    return data + padding

def pkcs7_unpad(data):
    pad_len = data[-1]
    return data[:-pad_len]

def encrypt(pk, plaintext, block_size):
    e, n = pk
    cipher = []
    padded_text = pkcs7_pad(plaintext.encode('utf-8'), block_size)
    for i in range(0, len(padded_text), block_size):
        block = padded_text[i:i + block_size]
        m = int.from_bytes(block, byteorder='big')
        if m >= n:
            raise ValueError("Block size is too large for the key size.")
        c = fast_exponentiation_mod(m, e, n)
        cipher.append(c)
    return cipher

def decrypt(pk, ciphertext, block_size):
    d, n, p, q = pk
    plain = b''
    for c in ciphertext:
        m = fast_exponentiation_mod(c, d, n)
        block = m.to_bytes(block_size, byteorder='big')
        plain += block
    unpadded_text = pkcs7_unpad(plain)
    return unpadded_text.decode('utf-8')

def save_pkcs8(public_key, filename):
    e, n = public_key
    with open(filename, 'w') as f:
        f.write(f"PublicExponent: {e}\n")
        f.write(f"Modulus: {n}\n")

def save_pkcs12(private_key, filename):
    d, n, p, q = private_key
    exponent1 = d % (p - 1)
    exponent2 = d % (q - 1)
    coefficient = reverse(q, p)
    with open(filename, 'w') as f:
        f.write(f"PrivateExponent: {d}\n")
        f.write(f"Prime1: {p}\n")
        f.write(f"Prime2: {q}\n")
        f.write(f"Exponent1: {exponent1}\n")
        f.write(f"Exponent2: {exponent2}\n")
        f.write(f"Coefficient: {coefficient}\n")

def save_pkcs7(ciphertext, filename, content_type="text"):
    if isinstance(ciphertext, bytes):
        encrypted_content_hex = ciphertext.hex()
    else:
        encrypted_content_hex = ''.join(f"{byte:02x}" for byte in ciphertext)  # Если ciphertext - список чисел

    pkcs7_structure = {
        "Version": 0,
        "EncryptedContentInfo": {
            "ContentType": content_type,
            "ContentEncryptionAlgorithmIdentifier": "rsaEncryption",
            "encryptedContent": encrypted_content_hex,
            "OPTIONAL": None
        }
    }

    with open(filename, 'w') as f:
        f.write("Version: " + str(pkcs7_structure["Version"]) + "\n")
        f.write("EncryptedContentInfo:\n")
        f.write("  ContentType: " + pkcs7_structure["EncryptedContentInfo"]["ContentType"] + "\n")
        f.write("  ContentEncryptionAlgorithmIdentifier: " + pkcs7_structure["EncryptedContentInfo"]["ContentEncryptionAlgorithmIdentifier"] + "\n")
        f.write("  encryptedContent: " + pkcs7_structure["EncryptedContentInfo"]["encryptedContent"] + "\n")
        f.write("  OPTIONAL: " + str(pkcs7_structure["EncryptedContentInfo"]["OPTIONAL"]) + "\n")
def main_menu():
  while True:
    print("Выберите действие:")
    print("1. Генерация ключей")
    print("2. Шифрование сообщения")
    print("3. Расшифрование сообщения")
    choice = input("Введите номер действия: ")

    if choice == '1':
        bit_length = 1024
        public_key, private_key = generate_keypair(bit_length)
        save_pkcs8(public_key, 'public_key.txt')
        save_pkcs12(private_key, 'private_key.txt')

    if choice == '2':
        plaintext = input("Введите сообщение для шифрования: ")
        block_size = 117
        ciphertext = encrypt(public_key, plaintext, block_size)
        save_pkcs7(ciphertext, 'encrypted_message.txt')

    if choice == '3':
        decrypted_text = decrypt(private_key, ciphertext, block_size)
        print("Расшифрованное сообщение: \n", decrypted_text)

if __name__ == "__main__":
    main_menu()