# app.py - Flask server to receive IP data
from flask import Flask, request, jsonify
import json
from datetime import datetime

app = Flask(__name__)

# Simple in-memory storage (for demo - use DB for production)
ip_log = []

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>IP Logger</h1>
    <p>Send POST requests to /log with IP data</p>
    <p>View logs at /logs</p>
    '''

@app.route('/log', methods=['POST'])
def log_ip():
    try:
        data = request.get_json()
        if not data:
            data = request.form.to_dict()
        
        # Add timestamp and source IP
        data['received_at'] = datetime.utcnow().isoformat()
        data['source_ip'] = request.remote_addr
        
        ip_log.append(data)
        
        # Keep only last 1000 entries (optional)
        if len(ip_log) > 1000:
            ip_log.pop(0)
        
        return jsonify({'status': 'success', 'message': 'IP logged'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/logs', methods=['GET'])
def view_logs():
    # Return logs as JSON
    return jsonify(ip_log)

@app.route('/latest', methods=['GET'])
def latest_ip():
    if ip_log:
        return jsonify(ip_log[-1])
    return jsonify({'message': 'No logs yet'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)