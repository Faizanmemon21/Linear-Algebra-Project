"""
=====================================================================
 LINEAR ALGEBRA CCP - TOPIC 4: CRYPTOGRAPHY
 The Hill Cipher: A Matrix-Based Polygraphic Substitution Cipher
=====================================================================
This script implements, tests, and analyzes the Hill Cipher, a classical
cryptographic scheme that relies entirely on Linear Algebra concepts:
matrix multiplication, determinants, matrix inverses, and modular
arithmetic over Z_26.
"""

import numpy as np
import sympy as sp
import time
import string
import matplotlib.pyplot as plt
from math import gcd

ALPHABET = string.ascii_uppercase
MOD = 26

# ---------------------------------------------------------------
# 1. CORE LINEAR ALGEBRA UTILITIES
# ---------------------------------------------------------------

def text_to_vector(text):
    """Convert a string of letters into a list of integers (A=0,...,Z=25)."""
    return [ALPHABET.index(c) for c in text.upper() if c in ALPHABET]


def vector_to_text(vec):
    return "".join(ALPHABET[int(x) % MOD] for x in vec)


def is_valid_key(K):
    """A Hill cipher key matrix K is valid iff gcd(det(K), 26) == 1,
    which guarantees K is invertible modulo 26."""
    det = int(round(np.linalg.det(K)))
    return gcd(det % MOD, MOD) == 1, det % MOD


def matrix_mod_inverse(K, mod=MOD):
    """Compute the modular inverse of an n x n key matrix using
    sympy's exact (rational) determinant/adjugate arithmetic, then
    reduce modulo `mod`."""
    Ksp = sp.Matrix(K)
    det = int(Ksp.det())
    det_inv = sp.mod_inverse(det % mod, mod)
    adj = Ksp.adjugate()
    K_inv = (det_inv * adj) % mod
    K_inv = np.array(K_inv.tolist(), dtype=int) % mod
    return K_inv


def pad_text(vec, n):
    while len(vec) % n != 0:
        vec.append(ALPHABET.index('X'))
    return vec


def hill_encrypt(plaintext, K):
    n = K.shape[0]
    vec = pad_text(text_to_vector(plaintext), n)
    cipher_vec = []
    for i in range(0, len(vec), n):
        block = np.array(vec[i:i + n])
        c_block = K.dot(block) % MOD
        cipher_vec.extend(c_block.tolist())
    return vector_to_text(cipher_vec)


def hill_decrypt(ciphertext, K):
    n = K.shape[0]
    K_inv = matrix_mod_inverse(K, MOD)
    vec = text_to_vector(ciphertext)
    plain_vec = []
    for i in range(0, len(vec), n):
        block = np.array(vec[i:i + n])
        p_block = K_inv.dot(block) % MOD
        plain_vec.extend(p_block.tolist())
    return vector_to_text(plain_vec), K_inv


# ---------------------------------------------------------------
# 2. CAESAR CIPHER (BASELINE FOR COMPARISON)
# ---------------------------------------------------------------

def caesar_encrypt(text, shift=3):
    vec = text_to_vector(text)
    return vector_to_text([(c + shift) % MOD for c in vec])


# ---------------------------------------------------------------
# 3a. INTERACTIVE CUSTOM KEY + MESSAGE (YOUR OWN INPUT)
# ---------------------------------------------------------------
# Lets you type any size key matrix in one line, e.g.:
#   3x3 -> 6,24,1;13,16,10;20,17,15
#   2x2 -> 3,5;1,2
# Rows are separated by ';' and numbers in a row by ','.
# The matrix is checked for validity (gcd(det,26)=1) before use.
# If it's invalid, you'll be told why and asked to try again.

def parse_matrix(s):
    """Parse a one-line matrix string like '6,24,1;13,16,10;20,17,15'
    into a numpy array."""
    rows = [r.strip() for r in s.strip().split(";") if r.strip() != ""]
    matrix = [[int(x.strip()) for x in row.split(",")] for row in rows]
    arr = np.array(matrix)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Matrix must be square (got shape {arr.shape}). "
                          f"Each row needs the same number of entries as there are rows.")
    return arr


def get_custom_key():
    while True:
        s = input("\nEnter your key matrix (rows separated by ';', "
                   "numbers separated by ','), e.g. 6,24,1;13,16,10;20,17,15:\n> ")
        try:
            K_custom = parse_matrix(s)
        except Exception as e:
            print(f"  Could not parse that matrix: {e}\n  Please try again.")
            continue

        valid, det_mod = is_valid_key(K_custom)
        print(f"\nYour Key Matrix K =\n{K_custom}")
        print(f"det(K) mod 26 = {det_mod}  ->  gcd(det,26)=1 ? {valid}  "
              f"(key {'VALID' if valid else 'INVALID'})")

        if valid:
            return K_custom
        print("  This key is NOT invertible mod 26, so it cannot be used for "
              "decryption (gcd(det,26) must equal 1). Please enter a different matrix.")


def get_custom_message():
    msg = input("\nEnter the plaintext message you want to encrypt "
                "(letters only, anything else is ignored): \n> ")
    return msg


print("=" * 70)
print(" CUSTOM HILL CIPHER TEST -- your own key matrix and message")
print("=" * 70)

K_custom = get_custom_key()
n_custom = K_custom.shape[0]
plaintext_custom = get_custom_message()

ciphertext_custom = hill_encrypt(plaintext_custom, K_custom)
decrypted_custom, K_inv_custom = hill_decrypt(ciphertext_custom, K_custom)

# Show the padded plaintext too, since encryption pads to a multiple of n
padded_plain_custom = vector_to_text(pad_text(text_to_vector(plaintext_custom), n_custom))

print(f"\nBlock size (n)        : {n_custom}x{n_custom}")
print(f"Plaintext (padded)    : {padded_plain_custom}")
print(f"Ciphertext            : {ciphertext_custom}")
print(f"Decrypted             : {decrypted_custom}")
print(f"Lossless round-trip   : {decrypted_custom == padded_plain_custom}")
print(f"\nK inverse mod 26 =\n{K_inv_custom}")

identity_check_custom = (K_custom.dot(K_inv_custom)) % MOD
print(f"\nK * K_inv mod 26 (should be Identity):\n{identity_check_custom}")
print("=" * 70)
print(" END OF CUSTOM TEST -- continuing with the original report demo below")
print("=" * 70 + "\n")


# ---------------------------------------------------------------
# 3b. ORIGINAL DEMONSTRATION: ENCRYPT / DECRYPT WITH THE REPORT'S 3x3 KEY MATRIX
# ---------------------------------------------------------------
# (Kept exactly as in the original report so all the analysis below --
#  frequency, avalanche, key space, efficiency, known-plaintext attack --
#  still reproduces the same figures and numbers as your Word report.)

K = np.array([
    [6, 24, 1],
    [13, 16, 10],
    [20, 17, 15]
])

valid, det_mod = is_valid_key(K)
print("=== HILL CIPHER (3x3 KEY MATRIX) ===")
print("Key Matrix K =\n", K)
print(f"det(K) mod 26 = {det_mod}  -> gcd(det,26)=1 ? {valid}  (key {'VALID' if valid else 'INVALID'})")

plaintext = "LINEARALGEBRAISTHEFOUNDATIONOFMODERNCRYPTOGRAPHY"
ciphertext = hill_encrypt(plaintext, K)
decrypted, K_inv = hill_decrypt(ciphertext, K)

print("\nPlaintext :", plaintext)
print("Ciphertext:", ciphertext)
print("Decrypted :", decrypted)
print("Lossless round-trip:", decrypted.startswith(plaintext))
print("\nK inverse mod 26 =\n", K_inv)

# verify K * K_inv = I (mod 26)
identity_check = (K.dot(K_inv)) % MOD
print("\nK * K_inv mod 26 (should be Identity):\n", identity_check)

# ---------------------------------------------------------------
# 4. FREQUENCY ANALYSIS: PLAINTEXT vs CAESAR vs HILL CIPHERTEXT
# ---------------------------------------------------------------

LONG_PLAIN = ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGTHISPANGRAMISUSEDREPEATEDLY"
              "TOPRODUCEAREALISTICDISTRIBUTIONOFENGLISHLETTERSFORFREQUENCYANALYSIS"
              "INCRYPTOGRAPHYWESTUDYHOWCIPHERSDISGUISETHENATURALSTATISTICSOFLANGUAGE") * 3

caesar_cipher = caesar_encrypt(LONG_PLAIN, 7)
hill_cipher_text = hill_encrypt(LONG_PLAIN, K)


def letter_freq(text):
    vec = text_to_vector(text)
    counts = np.zeros(26)
    for v in vec:
        counts[v] += 1
    return counts / max(1, len(vec)) * 100


freq_plain = letter_freq(LONG_PLAIN)
freq_caesar = letter_freq(caesar_cipher)
freq_hill = letter_freq(hill_cipher_text)

# Chi-square-like "flatness" deviation from uniform distribution (1/26)
uniform = np.full(26, 100 / 26)
dev_plain = np.sum((freq_plain - uniform) ** 2)
dev_caesar = np.sum((freq_caesar - uniform) ** 2)
dev_hill = np.sum((freq_hill - uniform) ** 2)
print(f"\nFrequency deviation from uniform (lower = more secure):")
print(f"  Plaintext       : {dev_plain:.2f}")
print(f"  Caesar Cipher   : {dev_caesar:.2f}")
print(f"  Hill Cipher(3x3): {dev_hill:.2f}")

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
labels = list(ALPHABET)
colors = ["#4C72B0", "#DD8452", "#55A868"]
data = [freq_plain, freq_caesar, freq_hill]
titles = [f"Plaintext (deviation={dev_plain:.1f})",
          f"Caesar Cipher, shift=7 (deviation={dev_caesar:.1f})",
          f"Hill Cipher, 3\u00d73 key (deviation={dev_hill:.1f})"]
for ax, freqs, title, c in zip(axes, data, titles, colors):
    ax.bar(labels, freqs, color=c)
    ax.axhline(100 / 26, color="gray", linestyle="--", linewidth=1, label="Uniform (1/26)")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Frequency (%)")
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("Letter")
plt.tight_layout()
plt.savefig("/home/claude/ccp_report/assets/freq_analysis.png", dpi=160)
plt.close()
print("Saved freq_analysis.png")

# ---------------------------------------------------------------
# 5. AVALANCHE EFFECT: SENSITIVITY TO SINGLE-CHARACTER CHANGE
# ---------------------------------------------------------------

def hamming_pct(a, b):
    n = min(len(a), len(b))
    diff = sum(1 for i in range(n) if a[i] != b[i])
    return diff / n * 100


base_msg = "ATTACKATDAWNBRINGREINFORCEMENTSNOW"
trials = 30
caesar_changes = []
hill_changes = []
rng = np.random.default_rng(42)

for _ in range(trials):
    pos = rng.integers(0, len(base_msg))
    new_letter = ALPHABET[rng.integers(0, 26)]
    mutated = base_msg[:pos] + new_letter + base_msg[pos + 1:]

    c1 = caesar_encrypt(base_msg, 7)
    c2 = caesar_encrypt(mutated, 7)
    caesar_changes.append(hamming_pct(c1, c2))

    h1 = hill_encrypt(base_msg, K)
    h2 = hill_encrypt(mutated, K)
    hill_changes.append(hamming_pct(h1, h2))

avg_caesar_change = np.mean(caesar_changes)
avg_hill_change = np.mean(hill_changes)
print(f"\nAvalanche effect (avg % of ciphertext changed by a single-letter edit):")
print(f"  Caesar Cipher : {avg_caesar_change:.2f}%")
print(f"  Hill Cipher   : {avg_hill_change:.2f}%")

plt.figure(figsize=(7, 5))
plt.boxplot([caesar_changes, hill_changes], tick_labels=["Caesar Cipher", "Hill Cipher (3\u00d73)"],
            patch_artist=True,
            boxprops=dict(facecolor="#9ecae1"),
            medianprops=dict(color="black"))
plt.scatter([1] * trials, caesar_changes, alpha=0.4, color="#DD8452")
plt.scatter([2] * trials, hill_changes, alpha=0.4, color="#55A868")
plt.ylabel("Ciphertext characters changed (%)")
plt.title("Avalanche Effect: Impact of a Single-Letter Edit in Plaintext")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/home/claude/ccp_report/assets/avalanche.png", dpi=160)
plt.close()
print("Saved avalanche.png")

# ---------------------------------------------------------------
# 6. KEY SPACE COMPARISON (BRUTE-FORCE SECURITY)
# ---------------------------------------------------------------

def hill_keyspace_size(n, mod=26):
    """Number of invertible n x n matrices mod 26 = mod^(n^2) * Prod_p (product over prime
    factors p of mod of the fraction of singular matrices removed). We approximate using
    the formula for GL(n, Z_m) via CRT over prime power factors of m (26 = 2 * 13)."""
    def gl_count_prime_power(p, e, n):
        # |GL(n, Z_{p^e})| = p^{(e-1)n^2} * |GL(n, F_p)|
        gl_fp = 1
        for i in range(n):
            gl_fp *= (p ** n - p ** i)
        return (p ** ((e - 1) * n * n)) * gl_fp

    # 26 = 2^1 * 13^1
    return gl_count_prime_power(2, 1, n) * gl_count_prime_power(13, 1, n)


caesar_keyspace = 26
hill2_keyspace = hill_keyspace_size(2)
hill3_keyspace = hill_keyspace_size(3)
hill4_keyspace = hill_keyspace_size(4)

print(f"\nKey space sizes:")
print(f"  Caesar Cipher        : {caesar_keyspace:,}")
print(f"  Hill Cipher 2x2      : {hill2_keyspace:,}")
print(f"  Hill Cipher 3x3      : {hill3_keyspace:,}")
print(f"  Hill Cipher 4x4      : {hill4_keyspace:,}")

plt.figure(figsize=(7.5, 5))
names = ["Caesar\n(shift cipher)", "Hill 2\u00d72", "Hill 3\u00d73", "Hill 4\u00d74"]
sizes_exact = [caesar_keyspace, hill2_keyspace, hill3_keyspace, hill4_keyspace]
sizes = [float(s) for s in sizes_exact]
bars = plt.bar(names, sizes, color=["#DD8452", "#9ecae1", "#55A868", "#c44e52"])
plt.yscale("log")
plt.ylabel("Key space size (log scale)")
plt.title("Brute-Force Key Space: Caesar vs Hill Cipher Matrix Sizes")
for b, s in zip(bars, sizes):
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.3, f"{s:,.0f}",
              ha="center", fontsize=8, rotation=0)
plt.tight_layout()
plt.savefig("/home/claude/ccp_report/assets/keyspace.png", dpi=160)
plt.close()
print("Saved keyspace.png")

# ---------------------------------------------------------------
# 7. COMPUTATIONAL EFFICIENCY: ENCRYPTION TIME vs MESSAGE LENGTH
# ---------------------------------------------------------------

lengths = [100, 500, 1000, 5000, 10000, 50000, 100000]
hill_times = []
caesar_times = []
rng2 = np.random.default_rng(1)
for L in lengths:
    msg = "".join(rng2.choice(list(ALPHABET), size=L))
    t0 = time.perf_counter()
    hill_encrypt(msg, K)
    hill_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    caesar_encrypt(msg, 7)
    caesar_times.append(time.perf_counter() - t0)

print("\nEfficiency (encryption time in seconds):")
for L, ht, ct in zip(lengths, hill_times, caesar_times):
    print(f"  n={L:>7,}  Hill={ht:.5f}s   Caesar={ct:.5f}s")

plt.figure(figsize=(7.5, 5))
plt.plot(lengths, hill_times, marker="o", label="Hill Cipher (3\u00d73 matrix mult.)", color="#55A868")
plt.plot(lengths, caesar_times, marker="s", label="Caesar Cipher (scalar shift)", color="#DD8452")
plt.xlabel("Plaintext length (characters)")
plt.ylabel("Encryption time (seconds)")
plt.title("Computational Efficiency: Hill Cipher vs Caesar Cipher")
plt.legend()
plt.grid(alpha=0.4, linestyle="--")
plt.tight_layout()
plt.savefig("/home/claude/ccp_report/assets/efficiency.png", dpi=160)
plt.close()
print("Saved efficiency.png")

# ---------------------------------------------------------------
# 8. KNOWN-PLAINTEXT ATTACK DEMONSTRATION (STABILITY / WEAKNESS)
# ---------------------------------------------------------------
# If an attacker recovers n plaintext/ciphertext block pairs forming an
# invertible matrix P, the key can be recovered as K = C * P^-1 (mod 26).

print("\n=== KNOWN-PLAINTEXT ATTACK DEMONSTRATION ===")
# Split the original plaintext into 3-letter blocks (as Hill encryption does)
plain_padded = vector_to_text(pad_text(text_to_vector(plaintext), 3))
blocks = [plain_padded[i:i + 3] for i in range(0, len(plain_padded), 3)]

# An attacker needs 3 known plaintext/ciphertext block pairs whose plaintext
# vectors form an invertible matrix mod 26. Search the known blocks for such a set.
found = False
for i in range(len(blocks) - 2):
    candidate_blocks = blocks[i:i + 3]
    P_try = np.array([text_to_vector(b) for b in candidate_blocks]).T
    det_try = int(round(np.linalg.det(P_try))) % MOD
    if gcd(det_try, MOD) == 1:
        found = True
        break

print(f"Known plaintext blocks used: {candidate_blocks}")
P_blocks = P_try
C_blocks = np.array([text_to_vector(hill_encrypt(b, K)) for b in candidate_blocks]).T

P_inv = matrix_mod_inverse(P_blocks, MOD)
K_recovered = (C_blocks.dot(P_inv)) % MOD
print("Recovered Key (should equal original K):\n", K_recovered)
print("Attack successful:", np.array_equal(K_recovered % MOD, K % MOD))

print("\nAll computations completed successfully.")
