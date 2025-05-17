import random
import math
import re
import json
import os

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def evklid(x, y): #1.1
    
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


def reverse(s, m): # поиск обратного элемента
    c, a, b= evklid(m, s)

    if c != 1: 
        print ("Обратного элемента не существует")
        return None
    else:  
        rev= b % m  #мод 
        return rev


def fast_exponentiation(a, n):#1.2
    x = 1

    while n > 0:
        if n & 1:#сравниваем последний бит числа с 1
            x = x * a
        a = a * a
        n = n >> 1#побитовый сдвиг вправо 
        
    return x


def fast_exponentiation_mod(a, s, m): 

    x = 1

    while s > 0:
        if s & 1:
            x = (x * a) % m
        a = (a * a) % m
        s = s >> 1

    return x


def jacobi_symbol(a, n):
    if n < 3 or n % 2 == 0:
        raise ValueError("n должно быть нечетным целым числом, большим или равным 3.")
    if a < 0 or a >= n:
        raise ValueError("a должно быть целым числом, удовлетворяющим условию 0 ≤ a < n.")
    
    t, _, _ = evklid(a, n)
    if t!= 1:
        return 0

    g = 1

    while a != 0:
        a = a % n
        if a == 0:
            return 0
        if a == 1:
            return g

        k = 0
        while a % 2 == 0:
            a //= 2
            k += 1
        ai = a

        if k % 2 != 0: #проверяем k на нечетность
            if n % 8 in (3, 5):
              g = -g

        if n % 4 == 3 and ai % 4 == 3:
            g = -g
        
        a = n % ai
        n = ai

    return g
     
        
def is_prime_fermat(n):#1.4

    if (n < 5) or (n % 2 == 0):
        return "n должно быть нечетным целым и >= 5"

    a = random.randint(2, n - 2)
    if fast_exponentiation_mod(a, n-1, n) != 1:
            return "Число n составное"

    return "Число n, вероятно, простое"


def is_prime_solovay_strassen(n):

    if (n < 5) or (n % 2 == 0):
        return "n должно быть нечетным целым и >= 5"

    a = random.randint(2, n - 2)
    r = fast_exponentiation_mod(a, (n-1)//2, n) 
    if r != 1 and r != n - 1:
        return "Число n составное"

    s = jacobi_symbol(a, n)

    if r != s % n:
        return "Число n составное"

    return "Число n, вероятно, простое"


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


def generate_prime1(k, t=100):
    while True:
        p = random.getrandbits(k)
        p |= (1 << (k - 1)) | 1  # Убедитесь, что p является нечетным
        test = True
        for _ in range(t):
            if not miller_rabin(p):
                test = False
                break
        if test and p % 4 == 3:
            return p


def solve_congruence(a, b, m): #1.7
    
    d, _, _ = evklid(a, m) #сохраняем только нод
    
    if b % d != 0:
        print("Решений нет")
        return None
    
    a1 = a // d
    b1 = b // d
    m1 = m // d
    
    
    a1_inv = reverse(a1, m1)
    
    if a1_inv is None:
        return None
    
    # Находим одно из решений
    x0 = (a1_inv * b1) % m1
    
    # Шаг 3: Находим все решения
    solutions = [(x0 + k * m1) % m for k in range(d)]
    
    return solutions


def second_degree(p, a):

    # Проверка, что p — простое и не равно 2
    if (p == 2) or (not miller_rabin(p)):
        return "p должно быть простым и != 2"

    # Проверка, что (a/p) = 1
    if jacobi_symbol(a, p) != 1:
        return "Условие (a/p) = 1 не выполнено"

    # Генерация случайного N, такого что (N/p) = -1
    while True:
        N = random.randint(2, p - 1)  # Случайное N от 2 до p-1
        if jacobi_symbol(N, p) == -1:
            break

    k = 0
    h = p - 1
    while h % 2 == 0:
        h //= 2
        k += 1

    # Вычисление a1 и a2
    a1 = fast_exponentiation_mod(a, (h + 1) // 2, p)
    a2 = reverse(a, p)

    # Вычисление N1 и N2
    N1 = fast_exponentiation_mod(N, h, p)
    N2 = 1
    j = 0

    for i in range(k - 1):
        b = (a1 * N2) % p
        c = (a2 * fast_exponentiation_mod(b, 2, p)) % p
        d = fast_exponentiation_mod(c, fast_exponentiation(2, k - 2 - i), p)

        if d == 1:
            j = 0
        elif d == p - 1:
            j = 1

        N2 = (N2 * fast_exponentiation_mod(N1, fast_exponentiation(2, i) * j, p)) % p

    x1 = (a1 * N2) % p
    x2 = (-a1 * N2) % p

    return x1, x2


def system(): #1.9
    k = int(input("Введите количество сравнений: "))

    m = []
    b = []
    x = 0
    M = 1

    for i in range(k):
        b_i = int(input(f"Введите остаток b_{i + 1}: "))
        m_i = int(input(f"Введите модуль m_{i + 1}: "))

        for j in range(len(m)):
            g,_,_ = evklid(m_i,m[j])
            if g != 1:
                raise ValueError("Модули не являются попарно взаимно простыми")

        m.append(m_i)
        b.append(b_i)

    for m_i in m:
       M *= m_i

    for i in range(k):
       M_i = M // m[i]
       N_i = reverse(M_i, m[i])
       x += b[i] * N_i * M_i


    return x % M


#1.10
def print_polinom(polinom):
    if not polinom or all(coef == 0 for coef in polinom):
        return "0"

    polinom_copy = polinom.copy()
    while polinom_copy[0] == 0 and len(polinom_copy) != 1:
        polinom_copy.pop(0)

    terms = []
    degree = len(polinom_copy) - 1

    for i, coef in enumerate(polinom_copy):
        if coef != 0:
            if degree - i == 0:
                term = f"{coef}"
            elif degree - i == 1:
                term = f"{coef}x" if coef != 1 else "x"
            else:
                term = f"{coef}x^{degree - i}" if coef != 1 else f"x^{degree - i}"
            terms.append(term)

    return " + ".join(terms)


def generation_polinom(p, k):
    polinom = [random.randint(0, p - 1) for _ in range(k + 1)]
    if polinom[0] == 0:
        polinom[0] = random.randint(1, p - 1)
    return polinom


def polinom_sum(polinom1, polinom2, p):
    max_len = max(len(polinom1), len(polinom2))
    polinom1 = [0] * (max_len - len(polinom1)) + polinom1
    polinom2 = [0] * (max_len - len(polinom2)) + polinom2
    res = []
    for i in range(max_len):
        summ = (polinom1[i] + polinom2[i]) % p
        res.append(summ)
    return res


def normalization(polin):
    while polin and polin[0] == 0:
        polin.pop(0)
    return polin or [0]


def polinom_composition(polin1, polin2, p):#умнож
    k = len(polin1) + len(polin2) - 1
    composition = [0] * k
    for i in range(len(polin1)):
        for j in range(len(polin2)):
            composition[i + j] += (polin1[i] * polin2[j]) % p
    return normalization(composition)


def polinom_division(divisible, divider, p):#деление
    res = []
    polinom_div = divisible.copy()
    while len(polinom_div) >= len(divider):
        y = reverse(divider[0], p)
        coeff = - polinom_div[0] * y
         #искать по другому вычитание, делитель умножть на обратное число к старшему
        for i in range(len(divider)):#лгоритм евкл
            polinom_div[i] += divider[i] * coeff
        polinom_div = normalization(polinom_div)
    for coef in polinom_div:
        res.append(coef % p)
    return res


def polinom_proiz_mod(polinom1, polinom2, mod_polinom, p):
    proiz = polinom_composition(polinom1, polinom2, p)
    return polinom_division(proiz, mod_polinom, p)

#2
def pollards_rho(n):
    c = 1
    def f(x):
        return (fast_exponentiation(x,2) + 1) % n
    a = c
    b = c
    while True:
        a = f(a)
        b = f(f(b))
        d,_,_ = evklid(a-b,n)
        if 1 < d < n:
            return d
        if d == n:
            return "Делитель не найден"


def base_for2(x):
     B = []
     n = 2
     while len(B) < x:
        if all(n % p != 0 for p in B):
            B.append(n)
        n += 1
     return B


def pollards_rho_minus(n):
    A = 10000
    B = base_for2(A)
    a = random.randint(2, n - 2)
    d,_,_ = evklid(a, n)
    if d >= 2:
        return d


    for p in B:

        l = int(math.log(n) / math.log(p))
        a = fast_exponentiation_mod(a, fast_exponentiation(p, l), n)
        d,_,_ = evklid(a - 1, n)

        if d == 1 or d == n:
            continue
        if 1 < d < n:
            return d

    return "Делитель не найден"


#3
def read_input(file_name):
    with open(file_name, 'r') as file:
        p = int(file.readline().strip())
        if miller_rabin(p) == False:
            return ValueError("Число p не является простым.")
        a = int(file.readline().strip())
        b = int(file.readline().strip())
    return p, a, b

def find_order(a, p):
    order = 1
    while fast_exponentiation_mod(a, order, p) != 1:
        order += 1
    return order

def disc_log(a, b, p, r, max_steps=1000):
    u_c, v_c = 2, 2
    u_d, v_d = 2, 2
    c = (fast_exponentiation_mod(a, u_c, p) * fast_exponentiation_mod(b, v_c, p)) % p
    d = c

    for step in range(max_steps):

        if c < p // 2:
            c = (a * c) % p
            u_c = (u_c + 1) % r
        else:
            c = (b * c) % p
            v_c = (v_c + 1) % r
        for i in range(2):
            if d < p // 2:
                d = (a * d) % p
                u_d = (u_d + 1) % r
            else:
                d = (b * d) % p
                v_d = (v_d + 1) % r
        if c == d:
            log1 = (v_c - v_d) % r
            log2 = (u_d - u_c) % r
            return solve_congruence(log1, log2, r)[0]

    return "Решений нет"


#4
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

#5

def rabin_encrypt(m, n):
    c = fast_exponentiation_mod(m, 2, n)
    return c

def rabin_decrypt(c, p, q):
    n = p * q
    mp = fast_exponentiation_mod(c, (p + 1) // 4, p)
    mq = fast_exponentiation_mod(c, (q + 1) // 4, q)

    _, yp, yq = evklid(p, q)

    m1 = (yp * p * mq + yq * q * mp) % n
    m2 = n - m1
    m3 = (yp * p * mq - yq * q * mp) % n
    m4 = n - m3

    return m1, m2, m3, m4


def text_to_number(text, block_size=128):
    byte_data = text.encode('utf-8')
    padded_data = pkcs7_pad(byte_data, block_size)
    blocks = [padded_data[i:i + block_size] for i in range(0, len(padded_data), block_size)]
    numbers = []
    for block in blocks:
        number = 0
        for char in block:
            number = (number << 8) + char
        numbers.append(number)

    return numbers

def number_to_text(numbers, block_size=128):
    text = ""
    for number in numbers:
        byte_size = (number.bit_length() + 7) // 8
        block = number.to_bytes(byte_size, byteorder='big')
        unpadded_block = pkcs7_unpad(block)
        text += unpadded_block.decode('utf-8', errors='replace')
    return text


def is_valid_text(text):
    return all(32 <= ord(char) <= 126 for char in text)

def generate_and_save_parameters():
    p = generate_prime1(1024)
    q = generate_prime1(1024)
    n = p * q

    with open("p.txt", "w") as f:
        f.write(str(p))
    with open("q.txt", "w") as f:
        f.write(str(q))
    with open("n.txt", "w") as f:
        f.write(str(n))

    print(f"Параметры p, q и n сгенерированы и сохранены в файлы p.txt, q.txt и n.txt")

def read_parameters():
    with open("p.txt", "r") as f:
        p = int(f.read())
    with open("q.txt", "r") as f:
        q = int(f.read())
    with open("n.txt", "r") as f:
        n = int(f.read())
    return p, q, n



#6

def is_primitive_root(alfa, p):
    #if find_order(alfa, p) == p-1:
        if fast_exponentiation_mod(alfa, 2, p) != 1 and fast_exponentiation_mod(alfa, 2, p) != -1 and fast_exponentiation_mod(alfa, (p-1)//2,p) != 1:
            return True


def generate_keys(filename1, filename2, bit_length=1024):
    p = generate_prime(bit_length)

    while True:
        alpha = random.randint(2, p - 2)
        if is_primitive_root(alpha, p):
            break

    a = random.randint(2, p - 2)
    beta = fast_exponentiation_mod(alpha, a, p)

    with open(filename1, 'w') as f:
        f.write(f"p = {p}\n")
        f.write(f"alpha = {alpha}\n")
        f.write(f"beta = {beta}\n")

    with open(filename2, 'w') as f:
        f.write(f"a = {a}")

    return ((p, alpha, beta), (a))

def encrypt6(p, alpha, beta, plaintext, block_size=128):
    padded_text = pkcs7_pad(plaintext.encode('utf-8'), block_size)
    blocks = [padded_text[i:i + block_size] for i in range(0, len(padded_text), block_size)]
    print(blocks)
    cipher = []
    for block in blocks:
        m = int.from_bytes(block, byteorder='big')
        k = random.randint(2, p - 2)
        c1 = fast_exponentiation_mod(alpha, k, p)
        c2 = (m * fast_exponentiation_mod(beta, k, p)) % p
        cipher.append((c1, c2))
    with open('ciphertext.txt', 'w') as f:
        for c1, c2 in cipher:
            f.write(f"{c1}\n{c2}\n")

    return cipher

def decrypt6(a,p, ciphertext, block_size=128):

    plain = b''
    cipher_blocks = [ciphertext[i:i + 2] for i in range(0, len(ciphertext), 2)]

    for c1, c2 in cipher_blocks:
        s = fast_exponentiation_mod(c1, a, p)
        s_inv = reverse(s, p)
        m = (c2 * s_inv) % p
        block = m.to_bytes(block_size, byteorder='big')
        plain += block
    unpadded_text = pkcs7_unpad(plain)
    return unpadded_text.decode('utf-8')


#2 семестр
#base 64 base 32
with open('base64.json', 'r') as file:
    base64_table = json.load(file)
with open('base32.json', 'r') as file:
    base32_table = json.load(file)

def text_to_ascii(text):
    return ''.join(format(ord(char), '08b') for char in text)

def ascii_to_text(binary_message):
    del_length = len(binary_message) % 8
    if del_length > 0:
        binary_message = binary_message[:-del_length]
    chars = [binary_message[i:i+8] for i in range(0, len(binary_message), 8)]
    ascii_message = ''.join(chr(int(char, 2)) for char in chars)
    return ascii_message

def encode_base64(bits):
    padding_length = (24 - len(bits) % 24) % 24
    bits += '0' * padding_length
    groups = [bits[i:i + 6] for i in range(0, len(bits), 6)]
    base64_string = ''.join(base64_table[group] for group in groups)
    if padding_length // 8 > 0 :
        base64_string =  base64_string[:-(padding_length // 8)]
    if padding_length == 8:
        base64_string += '='
    elif padding_length == 16:
        base64_string += '=='

    return base64_string

def encode_base32(bits):
    N = len(bits) % 5
    if len(bits) %40 != 0:
        padding_length = (5 - len(bits) % 5) % 5
        bits += '0' * padding_length

    groups = [bits[i:i + 5] for i in range(0, len(bits), 5)]
    base32_string = ''.join(base32_table[group] for group in groups)

    if N == 4: base32_string += '==='
    if N == 3: base32_string += '======'
    if N == 2: base32_string += '='
    if N == 1: base32_string += '===='

    return base32_string

def decode_base64(base64_string):
    reverse_base64_table = {v: k for k, v in base64_table.items()}
    padding_length = base64_string.count('=')
    message = base64_string.rstrip('=')
    res = ''.join([reverse_base64_table[char] for char in message])
#    if padding_length == 1:
#        res = res[:-8]
#    elif padding_length == 2:
#        res = res[:-16]
    return ascii_to_text(res)

def decode_base32(base32_string):
    reverse_base32_table = {v: k for k, v in base32_table.items()}
    message = base32_string.rstrip('=')
    res = ''.join([reverse_base32_table[char] for char in message])
    print(res)
    return ascii_to_text(res)


#Стрибог


PI = [
    0xFC, 0xEE, 0xDD, 0x11, 0xCF, 0x6E, 0x31, 0x16,
    0xFB, 0xC4, 0xFA, 0xDA, 0x23, 0xC5, 0x04, 0x4D,
    0xE9, 0x77, 0xF0, 0xDB, 0x93, 0x2E, 0x99, 0xBA,
    0x17, 0x36, 0xF1, 0xBB, 0x14, 0xCD, 0x5F, 0xC1,
    0xF9, 0x18, 0x65, 0x5A, 0xE2, 0x5C, 0xEF, 0x21,
    0x81, 0x1C, 0x3C, 0x42, 0x8B, 0x01, 0x8E, 0x4F,
    0x05, 0x84, 0x02, 0xAE, 0xE3, 0x6A, 0x8F, 0xA0,
    0x06, 0x0B, 0xED, 0x98, 0x7F, 0xD4, 0xD3, 0x1F,
    0xEB, 0x34, 0x2C, 0x51, 0xEA, 0xC8, 0x48, 0xAB,
    0xF2, 0x2A, 0x68, 0xA2, 0xFD, 0x3A, 0xCE, 0xCC,
    0xB5, 0x70, 0x0E, 0x56, 0x08, 0x0C, 0x76, 0x12,
    0xBF, 0x72, 0x13, 0x47, 0x9C, 0xB7, 0x5D, 0x87,
    0x15, 0xA1, 0x96, 0x29, 0x10, 0x7B, 0x9A, 0xC7,
    0xF3, 0x91, 0x78, 0x6F, 0x9D, 0x9E, 0xB2, 0xB1,
    0x32, 0x75, 0x19, 0x3D, 0xFF, 0x35, 0x8A, 0x7E,
    0x6D, 0x54, 0xC6, 0x80, 0xC3, 0xBD, 0x0D, 0x57,
    0xDF, 0xF5, 0x24, 0xA9, 0x3E, 0xA8, 0x43, 0xC9,
    0xD7, 0x79, 0xD6, 0xF6, 0x7C, 0x22, 0xB9, 0x03,
    0xE0, 0x0F, 0xEC, 0xDE, 0x7A, 0x94, 0xB0, 0xBC,
    0xDC, 0xE8, 0x28, 0x50, 0x4E, 0x33, 0x0A, 0x4A,
    0xA7, 0x97, 0x60, 0x73, 0x1E, 0x00, 0x62, 0x44,
    0x1A, 0xB8, 0x38, 0x82, 0x64, 0x9F, 0x26, 0x41,
    0xAD, 0x45, 0x46, 0x92, 0x27, 0x5E, 0x55, 0x2F,
    0x8C, 0xA3, 0xA5, 0x7D, 0x69, 0xD5, 0x95, 0x3B,
    0x07, 0x58, 0xB3, 0x40, 0x86, 0xAC, 0x1D, 0xF7,
    0x30, 0x37, 0x6B, 0xE4, 0x88, 0xD9, 0xE7, 0x89,
    0xE1, 0x1B, 0x83, 0x49, 0x4C, 0x3F, 0xF8, 0xFE,
    0x8D, 0x53, 0xAA, 0x90, 0xCA, 0xD8, 0x85, 0x61,
    0x20, 0x71, 0x67, 0xA4, 0x2D, 0x2B, 0x09, 0x5B,
    0xCB, 0x9B, 0x25, 0xD0, 0xBE, 0xE5, 0x6C, 0x52,
    0x59, 0xA6, 0x74, 0xD2, 0xE6, 0xF4, 0xB4, 0xC0,
    0xD1, 0x66, 0xAF, 0xC2, 0x39, 0x4B, 0x63, 0xB6
]

TAU = [
    0, 8, 16, 24, 32, 40, 48, 56,
    1, 9, 17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58,
    3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60,
    5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62,
    7, 15, 23, 31, 39, 47, 55, 63
]

A = [
    0x8e20faa72ba0b470, 0x47107ddd9b505a38, 0xad08b0e0c3282d1c, 0xd8045870ef14980e,
    0x6c022c38f90a4c07, 0x3601161cf205268d, 0x1b8e0b0e798c13c8, 0x83478b07b2468764,
    0xa011d380818e8f40, 0x5086e740ce47c920, 0x2843fd2067adea10, 0x14aff010bdd87508,
    0x0ad97808d06cb404, 0x05e23c0468365a02, 0x8c711e02341b2d01, 0x46b60f011a83988e,
    0x90dab52a387ae76f, 0x486dd4151c3dfdb9, 0x24b86a840e90f0d2, 0x125c354207487869,
    0x092e94218d243cba, 0x8a174a9ec8121e5d, 0x4585254f64090fa0, 0xaccc9ca9328a8950,
    0x9d4df05d5f661451, 0xc0a878a0a1330aa6, 0x60543c50de970553, 0x302a1e286fc58ca7,
    0x18150f14b9ec46dd, 0x0c84890ad27623e0, 0x0642ca05693b9f70, 0x0321658cba93c138,
    0x86275df09ce8aaa8, 0x439da0784e745554, 0xafc0503c273aa42a, 0xd960281e9d1d5215,
    0xe230140fc0802984, 0x71180a8960409a42, 0xb60c05ca30204d21, 0x5b068c651810a89e,
    0x456c34887a3805b9, 0xac361a443d1c8cd2, 0x561b0d22900e4669, 0x2b838811480723ba,
    0x9bcf4486248d9f5d, 0xc3e9224312c8c1a0, 0xeffa11af0964ee50, 0xf97d86d98a327728,
    0xe4fa2054a80b329c, 0x727d102a548b194e, 0x39b008152acb8227, 0x9258048415eb419d,
    0x492c024284fbaec0, 0xaa16012142f35760, 0x550b8e9e21f7a530, 0xa48b474f9ef5dc18,
    0x70a6a56e2440598e, 0x3853dc371220a247, 0x1ca76e95091051ad, 0x0edd37c48a08a6d8,
    0x07e095624504536c, 0x8d70c431ac02a736, 0xc83862965601dd1b, 0x641c314b2b8ee083
]

C_const = [
    "b1085bda1ecadae9ebcb2f81c0657c1f2f6a76432e45d016714eb88d7585c4fc4b7ce09192676901a2422a08a460d31505767436cc744d23dd806559f2a64507",
    "6fa3b58aa99d2f1a4fe39d460f70b5d7f3feea720a232b9861d55e0f16b501319ab5176b12d699585cb561c2db0aa7ca55dda21bd7cbcd56e679047021b19bb7",
    "f574dcac2bce2fc70a39fc286a3d843506f15e5f529c1f8bf2ea7514b1297b7bd3e20fe490359eb1c1c93a376062db09c2b6f443867adb31991e96f50aba0ab2",
    "ef1fdfb3e81566d2f948e1a05d71e4dd488e857e335c3c7d9d721cad685e353fa9d72c82ed03d675d8b71333935203be3453eaa193e837f1220cbebc84e3d12e",
    "4bea6bacad4747999a3f410c6ca923637f151c1f1686104a359e35d7800fffbdbfcd1747253af5a3dfff00b723271a167a56a27ea9ea63f5601758fd7c6cfe57",
    "ae4faeae1d3ad3d96fa4c33b7a3039c02d66c4f95142a46c187f9ab49af08ec6cffaa6b71c9ab7b40af21f66c2bec6b6bf71c57236904f35fa68407a46647d6e",
    "f4c70e16eeaac5ec51ac86febf240954399ec6c7e6bf87c9d3473e33197a93c90992abc52d822c3706476983284a05043517454ca23c4af38886564d3a14d493",
    "9b1f5b424d93c9a703e7aa020c6e41414eb7f8719c36de1e89b4443b4ddbc49af4892bcb929b069069d18d2bd1a5c42f36acc2355951a8d9a47f0dd4bf02e71e",
    "378f5a541631229b944c9ad8ec165fde3a7d3a1b258942243cd955b7e00d0984800a440bdbb2ceb17b2b8a9aa6079c540e38dc92cb1f2a607261445183235adb",
    "abbedea680056f52382ae548b2e4f3f38941e71cff8a78db1fffe18a1b3361039fe76702af69334b7a1e6c303b7652f43698fad1153bb6c374b4c7fb98459ced",
    "7bcd9ed0efc889fb3002c6cd635afe94d8fa6bbbebab076120018021148466798a1d71efea48b9caefbacd1d7d476e98dea2594ac06fd85d6bcaa4cd81f32d1b",
    "378ee767f11631bad21380b00449b17acda43c32bcdf1d77f82012d430219f9b5d80ef9d1891cc86e71da4aa88e12852faf417d5d9b21b9948bc924af11bd720"
]

def padding_str(message: bytes) -> bytes:
    m_len = len(message) * 8
    m = (511 - m_len) % 512
    m_bit = ''.join(f'{byte:08b}' for byte in message)
    padded_bits = '0' * m + '1' + m_bit

    return bytes(int(padded_bits[i:i + 8], 2) for i in range(0, 512, 8))

def X(a: bytes, b:bytes)-> bytes:
    return bytes(x^y for x, y in zip(a, b))

def S_str(a:bytes)-> bytes:
    return bytes(PI[b] for b in a)

def P_str(a):
    result = [0] * 64
    for i in range(64):
        result[i] = a[TAU[i]]
    return bytes(result)

def L_str(state: bytes) -> bytes:

    result = bytearray(64)

    for i in range(8):
        block = state[i * 8:(i + 1) * 8]

        block_int = int.from_bytes(block, 'big')
        t = 0
        for j in range(64):
            if (block_int >> (63 - j)) & 1:
                t ^= A[j]

        result[i * 8:(i + 1) * 8] = t.to_bytes(8, 'big')

    return bytes(result)

C_bytes = [bytes.fromhex(s) for s in C_const]

def E(K: bytes, m: bytes) -> bytes:
    state = X(K, m)
    for i in range(12):
        L = L_str(P_str(S_str(state)))
        K = L_str(P_str(S_str(X(K, C_bytes[i]))))
        state = X(L, K)

    return bytes(state)

def g(h, m, N):
    K = X(h, N)
    M = L_str(P_str(S_str(K)))
    S = E(M, m)

    new_h = X(X(S,h),m)

    return bytes(new_h)

def add_mod(x_bytes: bytes, addition: int) -> bytes:
    x_int = int.from_bytes(x_bytes, byteorder='big')
    r_1= fast_exponentiation(2, 512)
    result = (x_int + addition) % r_1
    return result.to_bytes(64, byteorder='big')

def stribog_main(message, digest_size):

    message = message[::-1]

    if digest_size == 512:
        IV= b'\x00' * 64
    elif digest_size == 256:
        IV = b'\x01' * 64
    h = IV
    N = b'\x00' * 64
    sigma = b'\x00' * 64

    len_message = len(message) * 8

    a = 0
    while len_message >= 512:
        a += 1
        m = message[-a * 64:][:64]
        h = g (h, m, N)
        N = add_mod(N, 512)
        sigma = add_mod(sigma, int.from_bytes(m, byteorder='big'))
        len_message-= 512

    cut = a * 64
    for_padd = message[0: len(message) - cut]

    m = padding_str(for_padd)
    len_M = len(for_padd) * 8
    m_int = int.from_bytes(m, byteorder='big')

    h = g(h, m, N)
    N = add_mod(N, len_M)
    sigma = add_mod(sigma, m_int)
    zero = b'\x00' * 64
    h = g(h, N, zero)
    h = g(h, sigma, zero)

    if digest_size == 256:
        h = h[:32]
    return h[::-1].hex()


#SHA
H256 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

K256 = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

H512 = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
]

K512 = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817
]


def right_rotate(value, shift, bits):
    return ((value >> shift) | (value << (bits - shift))) & ((1 << bits) - 1)

def right_shift(value, shift):
    return value >> shift

def ch(x, y, z):
    return (x & y) ^ (~x & z)

def maj(x, y, z):
    return (x & y) ^ (x & z) ^ (y & z)

def padding_sha(message, block_size):
    bit_length = len(message) * 8
    message = bytearray(message)
    message.append(0x80)

    if block_size == 1024:  # SHA-512
        zero_bytes = ((896 - bit_length - 1) % 1024) // 8
        message += b'\x00' * zero_bytes
        message += bit_length.to_bytes(16, 'big')#128
    elif block_size == 512:
        zero_bytes = ((448 - bit_length - 1) % 512) // 8
        message += b'\x00' * zero_bytes
        message += bit_length.to_bytes(8, 'big')

    return bytes(message)

def prepare_message_schedule(block, word_size, rounds):
    W = []
    bytes_per_word = word_size // 8 #размер слова

    for i in range(0, min(len(block), 16 * bytes_per_word), bytes_per_word):
        word = 0
        for j in range(bytes_per_word):
            if i + j < len(block):
                word = (word << 8) | block[i + j]
        W.append(word)

    for t in range(16, rounds):
        if word_size == 32:  # 256
            s0 = (right_rotate(W[t - 15], 7, word_size) ^ right_rotate(W[t - 15], 18, word_size) ^ right_shift(W[t - 15], 3))
            s1 = (right_rotate(W[t - 2], 17, word_size) ^ right_rotate(W[t - 2], 19, word_size) ^ right_shift(W[t - 2], 10))
            new_word = (W[t - 16] + s0 + W[t - 7] + s1) & 0xFFFFFFFF
        elif word_size == 64: #512
            s0 = (right_rotate(W[t - 15], 1, 64) ^ right_rotate(W[t - 15], 8, 64) ^ right_shift(W[t - 15], 7))
            s1 = (right_rotate(W[t - 2], 19, 64) ^ right_rotate(W[t - 2], 61, 64) ^ right_shift(W[t - 2], 6))
            new_word = (W[t - 16] + s0 + W[t - 7] + s1) & 0xFFFFFFFFFFFFFFFF

        W.append(new_word)

    return W

def process_block(h_values, W, K, word_size, rounds):
    a, b, c, d, e, f, g, h = h_values
    mask = (1 << word_size) - 1

    for t in range(rounds):#256
        if word_size == 32:
            S1 = right_rotate(e, 6, word_size) ^ right_rotate(e, 11, word_size) ^ right_rotate(e, 25, word_size)
            Sigma0 = right_rotate(a, 2, word_size) ^ right_rotate(a, 13, word_size) ^ right_rotate(a, 22, word_size)
        else:
            S1 = right_rotate(e, 14, 64) ^ right_rotate(e, 18, 64) ^ right_rotate(e, 41, 64)
            Sigma0 = right_rotate(a, 28, 64) ^ right_rotate(a, 34, 64) ^ right_rotate(a, 39, 64)

        temp1 = (h + S1 + ch(e, f, g) + K[t] + W[t]) & mask
        temp2 = (Sigma0 + maj(a, b, c)) & mask

        h = g
        g = f
        f = e
        e = (d + temp1) & mask
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & mask

    return [
        (h_values[0] + a) & mask,
        (h_values[1] + b) & mask,
        (h_values[2] + c) & mask,
        (h_values[3] + d) & mask,
        (h_values[4] + e) & mask,
        (h_values[5] + f) & mask,
        (h_values[6] + g) & mask,
        (h_values[7] + h) & mask
    ]

def finalize_hash(h_values, word_size):

    digest = bytearray()
    bytes_per_word = word_size // 8

    for val in h_values:
        for i in range(bytes_per_word - 1, -1, -1):
            digest.append((val >> (8 * i)) & 0xFF)

    return bytes(digest)

def sha256(message):
    h = list(H256)
    padded_msg = padding_sha(message, 512)

    for i in range(0, len(padded_msg), 64):
        block = padded_msg[i:i + 64]
        W = prepare_message_schedule(block, 32, 64)
        h = process_block(h, W, K256, 32, 64)

    result = finalize_hash(h, 32).hex()

    return result

def sha512(message):

    h = list(H512)
    padded_msg = padding_sha(message, 1024)

    for i in range(0, len(padded_msg), 128):
        block = padded_msg[i:i + 128]
        W = prepare_message_schedule(block, 64, 80)
        h = process_block(h, W, K512, 64, 80)

    result = finalize_hash(h, 64).hex()

    return result



#HMAC
def generate_key(length=None):
    if length is None:
        length = random.randint(8, 64)
    else:
        length = max(8, min(length, 64))

    key = os.urandom(length)
    with open('key.txt', 'wb') as f:
        f.write(key)
    return key

def read_key():
    try:
        with open('key.txt', 'rb') as f:
            return f.read()
    except FileNotFoundError:
        print("Файл ключа не найден. Сгенерируйте ключ сначала.")
        return None

def xor_bytes(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

def hmac(hash_choice, key, message):
    if hash_choice  == '1':
        block_size = 32
    elif hash_choice == '2' or hash_choice == '3':
        block_size = 64
    elif hash_choice == '4':
        block_size = 128

    if len(key) > block_size:
        if hash_choice == '1':
            key_padded = stribog_main(key, 256)
            key_padded =  bytes.fromhex(key_padded)

        #elif hash_choice == '2':
        #    key_padded = stribog_main(key, 512)
        #    key_padded = bytes.fromhex(key_padded)

        #elif hash_choice == '3':
        #    key_padded = sha256(key)
        #    key_padded = bytes.fromhex(key_padded)

        #elif hash_choice == '4':
        #    key_padded = sha512(key)
        #    key_padded = bytes.fromhex(key_padded)

    if len(key) < block_size:
        key_padded =  key + bytes([0] * (block_size - len(key)))

    ipad = bytes([0x36] * block_size)
    inner_key = xor_bytes(key_padded, ipad) + message

    opad = bytes([0x5C] * block_size)
    outer_key = xor_bytes(key_padded, opad)


    if hash_choice == '1':
        hmac_result_1 = stribog_main(inner_key, 256)
        hmac_result_1 = bytes.fromhex(hmac_result_1)
        hmac_result = stribog_main(outer_key + hmac_result_1, 256)

    elif hash_choice == '2':
        hmac_result_1 = stribog_main(inner_key, 512)
        hmac_result_1 = bytes.fromhex(hmac_result_1)
        hmac_result = stribog_main(outer_key + hmac_result_1, 512)

    elif hash_choice == '3':
        hmac_result_1 = sha256(inner_key)
        hmac_result_1 = bytes.fromhex(hmac_result_1)
        hmac_result = sha256(outer_key + hmac_result_1)

    elif hash_choice == '4':
        hmac_result_1 = sha512(inner_key)
        hmac_result_1 = bytes.fromhex(hmac_result_1)
        hmac_result = sha512(outer_key + hmac_result_1)

    return hmac_result

def get_input_source():
    print("Выберите источник ввода данных:")
    print("1 - Консоль")
    print("2 - Файл (text.txt)")
    choice = input("Ваш выбор: ")
    if choice == '1':
        return input("Введите сообщение: ").encode('utf-8')
    elif choice == '2':
        try:
            with open('text.txt', 'rb') as f:
                return f.read()
        except FileNotFoundError:
            print("Файл text.txt не найден.")
            return None
    else:
        print("Неверный выбор.")
        return None

def main_menu():
    while True:
     print("Семестр 1 или 2?")
     choice = input("Число ")
     if choice == '1':
        print("\nМеню:")
        print("1. Алгоритм Евклида")
        print("2. Алгоритм быстрого возведения в степень")
        print("3. Символ Якоби")
        print("4. Проверка на простоту")
        print("5. Тест Миллера")
        print("6. Генерация простого числа")
        print("7. Решение сравнения 1 степени")
        print("8. Решение сравнения 2 степени")
        print("9. Решение системы сравнений")
        print("10. Конечные поля")
        print("11. Факторизация чисел")
        print("12. Дискретное логарифмирование")
        print("13. RSA")
        print("14. Реализация шифра Рабина")
        print("15. Реализация шифра Эль-Гамаля")
        choice = input("Задание: ")

        if choice == '1':
            x = int(input("Введите первое число: "))
            y = int(input("Введите второе число: "))
            m, a, b = evklid(x, y)
            print(f"НОД({x}, {y}) = {m}")
            print(f"Коэффициенты a и b: {a}, {b}")


        elif choice == '2':
            print("Просто возведение(1) или по модулю?(2).")
            choice = input("Введите номер ")

            if choice == '1':
                a = int(input("Введите основание: "))
                n = int(input("Введите степень: "))
                result = fast_exponentiation(a, n)
                print(f"{a}^{n} = {result}")

            elif choice == '2': 
                a = int(input("Введите основание: "))
                s = int(input("Введите степень: "))
                n = int(input("Введите модуль: "))
                result = fast_exponentiation_mod(a, s, n)
                print(f"{a}^{s} mod {n} = {result}")
            break


        elif choice == '3':
            a = int(input("Введите целое число (числитель): "))
            n = int(input("Введите нечетное число (знаменатель): "))
            result = jacobi_symbol(a, n)
            print(f"Символ Якоби ({a}/{n}) = {result}")


        elif choice == '4':
            print("Тест Ферма(1) или тест Соловэя-Штрассена(2)?")
            choice = input("Введите номер ")

            if choice == '1':
                n = int(input("Введите число: "))
                result = is_prime_fermat(n)
                print(result)    

            elif choice == '2':
                n = int(input("Введите число: "))
                result = is_prime_solovay_strassen(n)
                print(result)


        elif choice == '5':
            
            n= int(input("Введите число: "))
            if miller_rabin(n):
             print(f"{n} - вероятно простое.")
            else:
             print(f"{n} - составное.")


        elif choice == '6':
            k = int(input("Введите размерность числа: "))
            prime_number = generate_prime(k)
            print(f"Сгенерированное простое число: {prime_number}")


        elif choice == '7':
            a = int(input("Введите a: "))
            b = int(input("Введите b: "))
            m = int(input("Введите m: "))

            solutions = solve_congruence(a, b, m)
            if solutions:
             print(f"Решения сравнения {a}x ≡ {b} (mod {m}): {solutions}")


        elif choice == '8':
            p = int(input("Введите p: "))
            a = int(input("Введите a: "))
            N = int(input("Введите N: "))
            result = second_degree(p, a, N)
            print(result)


        elif choice == '9':
            result = system()
            print(result)


        elif choice == '10':
            p = int(input("Введите над каким полем происходит построение: "))
            k = int(input("Введите степень построения: "))
            elements = input("Введите коэффициенты неприводимого полинома через запятую: ")
            polinom = list(map(int, elements.split(',')))

            polinom1 = generation_polinom(p, random.randint(1, k - 1))
            polinom2 = generation_polinom(p, random.randint(1, k - 1))

            print("Неприводимый полином: ", print_polinom(polinom))
            print("Полином 1: ", print_polinom(polinom1))
            print("Полином 2: ", print_polinom(polinom2))

            print("Сумма 1 и 2: ", print_polinom(polinom_sum(polinom1, polinom2, p)))
            print("Произведение 1 и 2: ", print_polinom(polinom_proiz_mod(polinom1, polinom2, polinom, p)))


        elif choice == '11':
            print("ρ-метод Полларда(1) или (ρ-1)-метод Полларда(2)")
            choice = input("Введите номер ")
            if choice == '1':
                n = int(input("Введите число: "))
                result = pollards_rho(n)
                print(f"ρ-метод Полларда: Нетривиальный делитель числа {n}: {result}")
            elif choice == '2':
                n = int(input("Введите число: "))
                result = pollards_rho_minus(n)
                print(f"ρ-метод Полларда: Нетривиальный делитель числа {n}: {result}")


        elif choice == '12':
            file_name = '3.txt'
            p, a, b = read_input(file_name)
            r = find_order(a, p)

            result = disc_log(a, b, p, r)

            with open('3out.txt', 'w') as file:
                file.write(str(result))


        elif choice == '13':
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
                block_size = 128
                ciphertext = encrypt(public_key, plaintext, block_size)
                save_pkcs7(ciphertext, 'encrypted_message.txt')

            if choice == '3':
                decrypted_text = decrypt(private_key, ciphertext, block_size)
                print("Расшифрованное сообщение: \n", decrypted_text)


        elif choice == '14':
            print("Выберите действие:")
            print("1. Генерация параметров p, q и n")
            print("2. Шифрование сообщения")
            print("3. Расшифрование сообщения")
            choice = input("Введите номер действия: ")

            if choice == "1":
                generate_and_save_parameters()
            elif choice == "2":
                p, q, n = read_parameters()
                message = input("Введите сообщение для шифрования: ")
                m = text_to_number(message)
                m = int(m[0])
                c = rabin_encrypt(m, n)
                print(f"Зашифрованное сообщение: {c}")
                with open("encrypted_message.txt", "w") as f:
                    f.write(str(c))
                print("Зашифрованное сообщение сохранено в файл encrypted_message.txt")
            elif choice == "3":
                p, q, n = read_parameters()
                with open("encrypted_message.txt", "r") as f:
                    c = int(f.read())
                m1, m2, m3, m4 = rabin_decrypt(c, p, q)
                print(f"Возможные расшифрованные сообщения: {m1}, {m2}, {m3}, {m4}")

                for m in [m1, m2, m3, m4]:
                    # Преобразуем число в текст
                    text = number_to_text([m])  # Передаём число как список из одного элемента
                    if text is not None and is_valid_text(text):
                        print(f"Правильное сообщение: {text}")
                        break


        elif choice == '15':
            print("Выберите действие:")
            print("1. Генерация ключей")
            print("2. Зашифрование")
            print("3. Расшифрование")

            choice = input("Введите номер действия: ")

            if choice == '1':
                p, alpha, beta, a = generate_keys("p.txt", "q.txt")

                # Сохраняем параметры в файлы
                with open('p6.txt', 'w') as f:
                    f.write(str(p))
                with open('alpha6.txt', 'w') as f:
                    f.write(str(alpha))
                with open('beta6.txt', 'w') as f:
                    f.write(str(beta))
                with open('a6.txt', 'w') as f:
                    f.write(str(a))

                print("Параметры сохранены в файлы p6.txt, alpha6.txt, beta6.txt и a6.txt")

            elif choice == '2':
                try:
                    # Чтение параметров из файлов
                    with open('p6.txt', 'r') as f:
                        p = int(f.readline())
                    with open('alpha6.txt', 'r') as f:
                        alpha = int(f.readline())
                    with open('beta6.txt', 'r') as f:
                        beta = int(f.readline())

                    # Ввод сообщения для зашифрования
                    message = input("Введите сообщение для зашифрования: ")
                    cifr = encrypt6(p, alpha, beta, message)

                    # Сохранение шифртекста в файл
                    with open('ciphertext.txt', 'w') as f:
                        f.write(f"{cifr}")

                    print("Шифртекст сохранен в файл ciphertext.txt")

                except FileNotFoundError:
                    print("Ошибка: файл с параметрами не найден.")
                except Exception as e:
                    print(f"Ошибка при зашифровании: {e}")

            elif choice == '3':
                try:
                    with open('p6.txt', 'r') as f:
                        p = int(f.readline())
                    with open('a6.txt', 'r') as f:
                        a = int(f.readline())
                    with open('ciphertext.txt', 'r') as f:
                        text = f.read().strip()
                        numbers = re.findall(r'\d+', text)

                        ciphertext = list(map(int, numbers))
                    message = decrypt6(a, p, ciphertext)
                    print(f"Расшифрованное сообщение: {message}")

                except FileNotFoundError:
                    print("Ошибка: файл с параметрами не найден.")
                except Exception as e:
                    print(f"Ошибка при расшифровании: {e}")

            else:
                print("Неверный выбор.")


     if choice == '2':
         print ("1. Base64 Base 32")
         print ("2. Стрибог")
         print ("3. SHA")
         print ("4. HMAC")

         choice = input("Введите номер лабораторной ")

         if choice == '1':
             operation = input("Выберите операцию (1 - кодирование, 2 - декодирование): ").strip()

             method = input("Выберите метод кодирования (1 - Base64, 2 - Base32): ").strip()

             input_type = input("Выберите метод ввода (1 - с клавиатуры, 2 - из файла): ").strip()

             if input_type == '1':
                 message = input("Введите сообщение: ").strip()
             else:
                 input_file = 'encode.txt' if operation == '2' else 'text.txt'
                 try:
                     with open(input_file, 'r') as file:
                         message = file.read().strip()
                 except FileNotFoundError:
                     print(f"Файл {input_file} не найден.")
                     return

             if operation == '1':
                 bits = text_to_ascii(message)
                 if method == '1':
                     result = encode_base64(bits)
                 else:
                     result = encode_base32(bits)
                 print("Закодированное сообщение:", result)
                 output_file = 'encode.txt'
             else:
                 if method == '1':
                     result = decode_base64(message)
                 else:
                     result = decode_base32(message)
                 print("Декодированное сообщение:", result)
                 output_file = 'decode.txt'

             try:
                 with open(output_file, 'w') as file:
                     file.write(result)
                 print(f"Результат успешно записан в файл {output_file}.")
             except Exception as e:
                 print(f"Ошибка при записи в файл: {e}")

         if choice == '2':
             while True:
                 try:
                     digest_size = int(input("Выберите размер свертки (256 или 512 бит): "))
                     if digest_size not in [256, 512]:
                         raise ValueError
                     break
                 except ValueError:
                     print("Ошибка: введите 256 или 512")
             print("\nВыберите источник данных:")
             print("1. Ввод с консоли")
             print("2. Чтение из файла (text.txt)")

             while True:
                 choice = input("Ваш выбор (1/2): ")
                 if choice in ['1', '2']:
                     break
                 print("Ошибка: введите 1 или 2")

             if choice == '1':
                 data = input("Введите строку для хеширования: ").encode('utf-8')
             else:
                 try:
                     with open('text.txt', 'r', encoding='utf-8') as f:
                         data = f.read().encode('utf-8')
                 except FileNotFoundError:
                     print("Ошибка: файл text.txt не найден")

             hash_result = stribog_main(data, digest_size)
             print(f"\nРезультат ({digest_size}-битная свертка):")
             print(hash_result)

         if choice == '3':
             print("Выберите алгоритм хеширования:")
             print("1. SHA-256")
             print("2. SHA-512")
             algorithm = input("Введите номер (1 или 2): ").strip()

             while algorithm not in ['1', '2']:
                 print("Некорректный ввод!")
                 algorithm = input("Введите номер (1 или 2): ").strip()

             print("\nВыберите способ ввода данных:")
             print("1. Ввести текст в консоли")
             print("2. Прочитать из файла text.txt")
             input_method = input("Введите номер (1 или 2): ").strip()

             while input_method not in ['1', '2']:
                 print("Некорректный ввод!")
                 input_method = input("Введите номер (1 или 2): ").strip()

             if input_method == '1':
                 print("\nВведите текст для хеширования (завершите ввод нажатием Enter):")
                 data = input().encode('utf-8')
             else:
                 try:
                     with open('text.txt', 'rb') as f:
                         data = f.read()
                     print("\nДанные успешно прочитаны из файла text.txt")
                 except FileNotFoundError:
                     print("\nОшибка: файл text.txt не найден!")
                     return
                 except IOError:
                     print("\nОшибка при чтении файла text.txt!")
                     return

             if algorithm == '1':
                 hash_result = sha256(data)
                 print("\nSHA-256 хеш:", hash_result)
             else:
                 hash_result = sha512(data)
                 print("\nSHA-512 хеш:", hash_result)

         if choice == '4':
             print("Выберите хеш-функцию:")
             print("1 - Stribog-256 ")
             print("2 - Stribog-512 ")
             print("3 - SHA-256")
             print("4 - SHA-512")
             hash_choice = input("Ваш выбор: ")

             message = get_input_source()

             print("Выберите действие с ключом:")
             print("1 - Сгенерировать новый ключ")
             print("2 - Использовать существующий ключ")
             key_choice = input("Ваш выбор: ")

             if key_choice == '1':
                 key = generate_key()
             elif key_choice == '2':
                 key = read_key()
                 if key is None:
                     return
             else:
                 print("Неверный выбор.")
                 return

             hmac_result = hmac(hash_choice, key, message)

             # Вывод результата
             print("HMAC результат (hex):", hmac_result)


if __name__ == "__main__":
    main_menu()