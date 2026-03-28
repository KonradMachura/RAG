import pytest
from src.core import security

def test_long_password_hashing():
    # Password longer than 72 bytes
    long_password = "a" * 100
    
    # This should not raise ValueError
    hashed = security.get_password_hash(long_password)
    assert hashed is not None
    
    # Should be verifiable
    assert security.verify_password(long_password, hashed) is True
    
    # Should treat first 72 chars as the same password
    assert security.verify_password(long_password[:72], hashed) is True
    
    # Should fail for different passwords
    assert security.verify_password("wrong_password", hashed) is False

def test_short_password_hashing():
    password = "short"
    hashed = security.get_password_hash(password)
    assert security.verify_password(password, hashed) is True
