Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser

python -m venv antvenv
  source antvenv/bin/activate

python ./workspaces/vsc/app/app.py