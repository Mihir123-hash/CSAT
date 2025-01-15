import secrets

def generate_secret_key(length=32):
    # Generate a random URL-safe text string of the specified length
    secret_key = secrets.token_urlsafe(length)
    return secret_key

# Example usage
if __name__ == "__main__":
    secret_key = generate_secret_key(12)
    print(f"Generated secret key: {secret_key}")
