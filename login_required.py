from functools import wraps
from flask import request, jsonify
import jwt_handler

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            return jsonify({'error': 'Authentication required'}), 401
        payload = jwt_handler.verify_token(token)
        if not payload or 'user_id' not in payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        request.user_id = payload['user_id']
        return f(*args, **kwargs)
    return decorated_function
