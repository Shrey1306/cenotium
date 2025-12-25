"""Security protocol for encryption and message signing."""

import hashlib
import hmac
import json
from datetime import datetime
from typing import Dict

from cryptography.fernet import Fernet


class SecurityProtocol:
    """Handles encryption, decryption, and message signing."""

    def __init__(self, signing_key: bytes = b"default-signing-key"):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        self.trusted_keys: Dict[str, bytes] = {}
        self.signing_key = signing_key

    def encrypt_message(self, message: dict) -> bytes:
        """Encrypt a dictionary message using Fernet."""
        message_bytes = json.dumps(message).encode()
        return self.cipher_suite.encrypt(message_bytes)

    def decrypt_message(self, encrypted_message: bytes) -> dict:
        """Decrypt a Fernet-encrypted message."""
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_message)
        return json.loads(decrypted_bytes)

    def sign_message(self, message: dict) -> str:
        """Create HMAC-SHA256 signature for a message."""
        message_bytes = json.dumps(message, sort_keys=True).encode()
        signature = hmac.new(self.signing_key, message_bytes, hashlib.sha256)
        return signature.hexdigest()

    def verify_signature(self, message: dict, signature: str) -> bool:
        """Verify a message's digital signature."""
        expected_signature = self.sign_message(message)
        return hmac.compare_digest(signature, expected_signature)

    def rotate_keys(self):
        """Rotate encryption keys for enhanced security."""
        new_key = Fernet.generate_key()
        self.trusted_keys[datetime.now().isoformat()] = self.key
        self.key = new_key
        self.cipher_suite = Fernet(new_key)
