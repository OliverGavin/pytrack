from flask import Flask
from flask_socketio import SocketIO, emit

# from threading import Thread
# from gevent import monkey, sleep
# from gevent.threading import Thread
# monkey.patch_all()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='gevent')  # uwsgi --master --http 5000 --http-websockets --gevent 10 --wsgi-file app.py --enable-threads
# socketio = SocketIO(app, async_mode='threading')


@app.route('/')
def index():
    return """
    <html>

    <head>
    <script src='https://cdnjs.cloudflare.com/ajax/libs/socket.io/2.0.4/socket.io.js'></script>
    <script>

      var socket = io('http://localhost:5000/camera');
      socket.on('subscribe', function(data) {
         im = 'data:image/jpeg;charset=utf-8;base64, ' + data['count'];
         document.getElementById('img')
            .setAttribute(
                'src', im
            );
      });
    </script>
    </head>

    <body>
    <img id="img" src="" alt="..." />
    </body>

    </html>
    """


@socketio.on('publish', namespace='/camera')
def publish(message):
    emit('subscribe', message, broadcast=True, namespace='/camera')


# @socketio.on('connect', namespace='/test')
# def test_connect():
#     emit('my response', {'data': 'Connected'})
#
#
# @socketio.on('disconnect', namespace='/test')
# def test_disconnect():
#     print('Client disconnected')


socketio.run(app)
# TODO try instead of server: https://github.com/miguelgrinberg/Flask-SocketIO/blob/master/example/app.py#L17
