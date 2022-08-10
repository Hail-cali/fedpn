
import torch
import numpy as np

OVERFLOW  = 100000003


def fed_aggregate_avg(agg, results):
    '''
    :param agg reseted model
    :param results: defaultdict ( model,
    :return:
    '''
    # result['state']
    # result['data_len']
    # result['data_info']
    size = 1
    # agg.to('cpu')
    lst = ['4','11','12','13']
    for client in results:
        # client.to('cpu')
        for k, v in client['params'].items():
            if k[:2] in lst:
                agg[k] += (v * client['data_len'])
                size += client['data_len']

    print('debug', size, len(results))


    for k, v in agg.items():
        if torch.is_tensor(v):
            agg[k] = torch.div(v, size)

        elif isinstance(v, np.ndarray):
            agg[k] = np.divide(v, size)

        elif isinstance(v, list):
            agg[k] = [val / size for val in agg[k]]

        elif isinstance(v, int):
            agg[k] = v / size




def fed_aggregate_(result):

    return
