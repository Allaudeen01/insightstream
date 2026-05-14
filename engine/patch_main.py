import re

with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add imports
imports = """
from auth import get_current_user
from models import User
from routers import auth
from fastapi import Depends
"""
content = imports + content

# Include router
app_decl = 'app = FastAPI(title="Virtual Data Scientist Engine", lifespan=lifespan)'
content = content.replace(app_decl, app_decl + '\napp.include_router(auth.router)')

# Find all @app.get/post/put/delete decorators and their functions
# Example: 
# @app.get("/health-check/{session_id}", response_model=HealthCheckResponse)
# def get_health_check(session_id: str):
#
# We want to change it to:
# def get_health_check(session_id: str, current_user: User = Depends(get_current_user)):

def replacer(match):
    decorator = match.group(1)
    func_def = match.group(2)
    # inject at the end of arguments
    # find the closing parenthesis of the function arguments
    # this is a simple string manipulation
    idx = func_def.rfind(')')
    if idx != -1:
        before = func_def[:idx]
        after = func_def[idx:]
        if before.strip().endswith('('):
            new_func_def = before + "current_user: User = Depends(get_current_user)" + after
        else:
            new_func_def = before + ", current_user: User = Depends(get_current_user)" + after
        return decorator + new_func_def
    return match.group(0)

# Regex to match decorator and function def
pattern = r'(@app\.(?:get|post|put|delete)[^\n]*\n(?:async )?def [^\(]+\([^)]*\)\s*(?:->\s*[^\n:]+)?\s*:)'
content = re.sub(pattern, replacer, content)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)
print("main.py patched")
