# AgroGuard Backend Server

To run the server so mobile devices on your local network can connect:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
or
```bash
python -m app.main
```