from fastapi import Depends, HTTPException
from jose import jwt
import datetime

SECRET_KEY = "supersecret"

def get_current_user(token: str = Depends(lambda: "token")):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user"]
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")

def create_token(username):
    return jwt.encode({"user": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=5)}, SECRET_KEY)
