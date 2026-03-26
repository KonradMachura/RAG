from src.core.security import get_password_hash, verify_password

def test_password_hashing():
    password = "test_password"
    hashed = get_password_hash(password)
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False

def test_create_access_token():
    from src.core.security import create_access_token
    import jwt
    from src.core.security import SECRET_KEY, ALGORITHM
    
    data = {"sub": "testuser"}
    token = create_access_token(data)
    assert isinstance(token, str)
    
    decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded["sub"] == "testuser"
    assert "exp" in decoded
