class Clients:
    def __init__(self, args=None, tdata=None, lmodel=None, title=None):
        self.train_data = tdata
        self.local_model = lmodel
        self.args = args
        self.batch_size = self.args.local_bs
        self.loss = 0
        self.accuracy = 0
        self.title = title
