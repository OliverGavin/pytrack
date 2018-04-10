from flask import Flask
from flask_socketio import SocketIO, emit


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='gevent')


@socketio.on('publish', namespace='/camera_publish')
def publish(message):
    emit('subscribe', message, broadcast=True, namespace='/camera')


socketio.run(app)
