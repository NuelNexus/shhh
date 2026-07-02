# app.py (corrected)
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)  # Must be named 'app'

ip_log = []

@app.route('/')
def home():
    return 'IP Logger is running. Use /log (POST) or /logs (GET)'

@app.route('/log', methods=['POST'])
def log_ip():
    data = request.get_json() or request.form.to_dict()
    data['received_at'] = datetime.utcnow().isoformat()
    data['source_ip'] = request.remote_addr
    ip_log.append(data)
    if len(ip_log) > 1000:
        ip_log.pop(0)
    return jsonify({'status': 'success'}), 200

@app.route('/logs', methods=['GET'])
def view_logs():
    return jsonify(ip_log)

@app.route('/latest', methods=['GET'])
def latest_ip():
    return jsonify(ip_log[-1] if ip_log else {'message': 'No logs'})

# For Vercel, no need for if __name__ == '__main__'