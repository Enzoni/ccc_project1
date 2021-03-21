import json

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



class Region:
    def __init__(self, info):
        self.region_id = info['properties']['id']
        self.xmin = info['properties']['xmin']
        self.xmax = info['properties']['xmax']
        self.ymin = info['properties']['ymin']
        self.ymax = info['properties']['ymax']
        self.count = 0
        self.score = 0

    def print_all(self):
        print(self.region_id)
        print(self.xmin)
        print(self.xmax)
        print(self.ymin)
        print(self.ymax)
        print(self.count)
        print(self.score)
        return


def read_grid_info(grid_file_name):
    with open(grid_file_name, encoding='utf-8') as f:
        grid_info = json.load(f)['features']
    regions = []
    for i in grid_info:
        regions.append(Region(i))
    return regions



class Message:
    def __init__(self, info):
        self.text = str.lower(info['value']['properties']['text'])
        self.coordinates = info['value']['geometry']['coordinates']
        self.region_id = self.find_region_id(regions)
        self.sentiment_score = 0

    #
    def find_region_id(self, regions):
        for region in regions:
            if region.xmin <= self.coordinates[0] < region.xmax and region.ymin <= self.coordinates[1] < region.ymax:
                self.region_id = region.region_id
                return

    # calculate sentiment_score
    def cal_sentiment_score(self, sentiment_dic):
        words = self.text.split()
        for word in words:
            # check if the word is with a punctuation
            if word[-1] in ['!', ',', '?', '.', "'", '"']:
                word = word[:-1]
            if word in sentiment_dic.keys():
                self.sentiment_score += sentiment_dic[word]
        return

    #
    def print_all(self):
        print(self.text)
        print(self.coordinates)
        print(self.region_id)





if __name__ == '__main__':
    sentiment_file_name = "data/AFINN.txt"
    grid_file_name = 'data/melbGrid.json'
    message_file_nmae = 'data/tinyTwitter.json'

    # read sentiment file
    with open(sentiment_file_name, "r") as f:
        data = f.read()

    sentiment_data = [i.split('\t') for i in data.split('\n')]
    # create sentiment dictionary
    sentiment_dic = {}
    for i, j in sentiment_data:
        sentiment_dic[i] = j


    # read grid data
    regions = read_grid_info(grid_file_name)
    # TODO 检查 regions
    # for i in regions:
    #     i.print_all()


    # read tweet message
    with open(message_file_nmae, encoding='utf-8') as f:
        message_list = json.load(f)

        total_rows = message_list['total_rows']
        offset = message_list['offset']
        data = message_list['rows']

    message = []

    for i in data:
        message.append(Message(i))

        # TODO 查看 message 内容
        Message(i).print_all()
