__author__ = 'User'
import daemon;

class Server(daemon.Daemon):
    def __init__(self):
        super(Server, self).__init__()