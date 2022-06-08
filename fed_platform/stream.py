
import asyncio
import requests
from queue import PriorityQueue
import time
from fed_platform.cluster import Cluster, SegmentationCluster, ClassificationCluster

from collections import deque

TIMEOUT = 3000
'''
timeout not used in presence
'''


class BaseStream:

    def __init__(self, timeout=TIMEOUT, reader=None, writer=None):
        self.reader: BaseReader = reader
        self.writer: BaseWriter = writer
        self._tasks = None
        self.timeout = timeout
        self._schedule = []   # task tuple set
        print(f"{'-'*10}\nStream Info:\n{self}\n{'-'*10}")

    def __repr__(self):
        return f"STREAM :: {self.__class__} \nSET :: READER: {self.reader} \nSET :: WRITER: {self.writer}"

    @property
    def queue(self):
        return PriorityQueue()

    @property
    def tasks(self):
        return self._tasks

    @tasks.setter
    def tasks(self, cls):

        self._tasks = cls

    def scheduler(self, url=None):

        if self.check_status(self.reader):
            self._schedule.append(self.reader.request(url))

        else:
            print(f'Check Reader Type: {type(self.reader)}')

        print(f'reserved : {len(self._schedule)} ', end=' ')

    def executor(self):
        print('>> execute')
        return self._schedule

    @staticmethod
    def check_status(reader):

        if isinstance(reader, BaseReader):
            return True
        else:
            return False


class FedStream(BaseStream):
    '''
    :parameter: reader: FedReader, writer: FedWriter
    '''
    def __init__(self, *args, **kwargs):
        super(FedStream, self).__init__(*args, **kwargs)
        self.reader: FedReader

        if self.check_status(self.reader):
            pass

    def scheduler(self, pack=None):
        pack.tb_writer = self.writer
        self.reader.packs.append(pack)

        self._schedule.append(self.reader.run())
        print(f"\t{'|'}", end=' ')
        print(f'Reserved : {len(self._schedule)} ', end=' ')

    @staticmethod
    def check_status(reader):

        if isinstance(reader, FedReader):
            return True
        else:
            return False


class BaseReader:

    def __init__(self, base=requests, session=None, timeout=TIMEOUT):
        self.session = session
        self._urls = None
        self.timeout = timeout
        self.base_engine = base

    def __repr__(self):
        return f"{self.__class__} :: BASE SESSION: {self.session} BASE ENGINE: {self.base_engine.__title__}"

    async def __aenter__(self, *args):

        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):

        return self

    @property
    def urls(self):
        return self._urls

    @urls.setter
    def urls(self, query):
        self._urls = query

    async def request(self, url=None):

        if url is not None:
            self.urls = url

        if issubclass(self.session, Cluster):

            async with self.session(self.base_engine, url) as response:

                result = response

                return result


class FedReader(BaseReader):

    def __init__(self, model_map_locate=None, cluster=None):
        super(FedReader).__init__()
        self.map_model = model_map_locate   # func before allocate
        self.cluster : Cluster = cluster
        self.packs = deque()

    def __repr__(self):
        return f"{self.__class__} :: BASE Cluster: {self.cluster} BASE NET: {self.map_model}"

    async def run(self):
        '''
        :param loader:
        :return: fed model params
        '''
        async with self.cluster(self.map_model, self.packs.popleft()) as response:
            result = response
            return result


class BaseWriter:

    pass
