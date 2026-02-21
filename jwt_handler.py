import jwt
import datetime
from flask import current_app

JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_DAYS = 7

def create_token(user_id):
    """
    Create a JWT token for the given user_id with 7 day expiration.
    """
    exp = datetime.datetime.utcnow() + datetime.timedelta(days=JWT_EXP_DELTA_DAYS)
    payload = {
        'user_id': user_id,
        'exp': exp
    }
    secret = current_app.config['SECRET_KEY']
    token = jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)
    return token

def verify_token(token):
    """
    Verify a JWT token and return the payload if valid, else None.
    """
    secret = current_app.config['SECRET_KEY']
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
