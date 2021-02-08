import logging, os

class Log():
    def __init__(self,LOG_FILE,clean = False):
        self.logger = logging.getLogger()
        self.LOG_FILE=LOG_FILE
        self.logger.setLevel(logging.DEBUG)
        self.LOG_FORMAT = "%(message)s"
        formatter = logging.Formatter(self.LOG_FORMAT)

        if clean:
            if os.path.isfile(self.LOG_FILE):
                with open(self.LOG_FILE, 'w') as f:
                    pass

        fh = logging.FileHandler(self.LOG_FILE)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def log(self, *args):
        s = ''
        for i in args:
            s += (str(i) + ' ')

        logging.debug(s)

