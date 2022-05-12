# python 3.8
import asyncio
from opt import parse_opt
import time

from trainer.loss import Trainer, Validator

TIMEOUT = 3000


class Cluster:
    '''
    TEST Cluster
    '''
    def __init__(self, model=None, pack=None):
        self.model = model
        self.pack = pack
        if self.pack.args.mode == 'train':
            self.module = Trainer()
        elif self.pack.args.mode == 'val':
            self.module = Validator()


    async def __aenter__(self, *args):

        return self.module(self.net, self.pack)

    async def __aexit__(self, exc_type, exc_val, exc_tb):

        pass


class LocalAPI:

    def __init__(self, args, base_steam=None, base_reader=None,
                 base_net=None, base_cluster=None, verbose=False):

        super(LocalAPI).__init__()
        self.args = args
        self.mode = args.mode
        self.net = base_net
        self.cluster = base_cluster
        self.stream = base_steam(reader=base_reader(model=self.net, cluster=self.cluster),
                                 writer=None)
        self.verbose = verbose

    def execute(self):
        result = []
        a_start = time.time()

        for l in range(self.args.num_clinets):

            self.stream.scheduler()

        tasks = self.stream.executor()

        loop = asyncio.get_event_loop()

        finished, unfinished = loop.run_until_complete(
            asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        )

        for task in finished:
            result.append(task.result())

            if self.verbose:
                print(task.result())

        print("unfinished:", len(unfinished))

        loop.close()

        a_end = time.time()

        print(f'total time {a_end - a_start:.5f}times')
        print(f"{'+' * 20}")

        return result



if __name__ == '__main__':

    from fed_platform.stream import FedStream, FedReader
    args = parse_opt()

    model = None

    running = LocalAPI(args, base_steam=FedStream, base_reader=FedReader, base_net=model, base_cluster=Cluster,
                       verbose=True)



    pass