# python 3.8
import asyncio
from opt import parse_opt
import time

from trainer.loss import Trainer, Validator, BaseLearner
from utils.pack import LoaderPack

from torch import nn

TIMEOUT = 3000


class Cluster:
    '''
    TEST Cluster
    '''
    def __init__(self, model_map_location=None, pack=None):
        self.pack = pack   # pre set

        if self.pack.args.mode == 'train':
            self.module = Trainer()
        elif self.pack.args.mode == 'val':
            self.module = Validator()
        else:
            print(f'mode {self.pack.args.mode}::')
            self.module = BaseLearner()

        self.model = self._set_model(model_map_location)

    def _set_model(self, model_map_location):
        model = model_map_location(pretrained=self.pack.args.pretrained)


        return model


    def _init_weight(self, model):
        if isinstance(model, nn.Module):
            pass
        state_dict = None


    async def __aenter__(self, *args):

        return self.module(self.model, self.pack)

    async def __aexit__(self, exc_type, exc_val, exc_tb):

        pass


class SegmentationCluster(Cluster):

    def __init__(self, *args, **kwargs):
        super(SegmentationCluster, self).__init__(*args, **kwargs)


class LocalAPI:

    def __init__(self, args, base_steam =None, base_reader=None,
                 base_net=None, base_cluster=None, verbose=False):

        super(LocalAPI).__init__()
        self.args = args
        self.mode = args.mode
        self.net = base_net
        self.cluster: Cluster = base_cluster
        self.stream: FedStream = base_steam(reader=base_reader(model_map_locate=self.net, cluster=self.cluster),
                                 writer=None)

        self.client_map = {0:'client_almost', 1: 'client_animal', 2: 'client_vehicle',
                  3: 'client_obj', 4: 'client_all'}
        self.root = args.root
        self.client_state_dict_path = {}

        self.verbose = verbose

    def dispense_task(self):

        for l in range(self.args.num_clinets):
            pack = LoaderPack(args, self.client_map[l])
            self.stream.scheduler(pack=pack)

        tasks = self.stream.executor()

        return tasks


    def execute(self):
        result = []
        a_start = time.time()

        chk_pth_path = {}
        tasks = self.dispense_task()


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
    from model.hailnet import hail_mobilenet_v3_large
    args = parse_opt()

    running = LocalAPI(args, base_steam=FedStream, base_reader=FedReader, base_net=hail_mobilenet_v3_large, base_cluster=Cluster,
                       verbose=True)



