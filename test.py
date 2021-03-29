import json
from itertools import islice, product
from collections import defaultdict

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

BATCH_SIZE = 2

class Region:
    """
    region_id：(str) Cell, A1 or A2 or ...
    xmin,xmax,ymin,ymax: (float) latitude and longitude range.
    count: (int) number of of Tweets
    score: (int) Sentiment Score
    """
    def __init__(self, info):

        self.region_id = info['properties']['id']
        self.xmin = info['properties']['xmin']
        self.xmax = info['properties']['xmax']
        self.ymin = info['properties']['ymin']
        self.ymax = info['properties']['ymax']
        self.count = 0
        self.score = 0

class Message:
    """
    text: (str) text need to analysis
    coordinates: list(str) [x,y]
    sentiment_score: int sentiment score for current Tweet
    """
    def __init__(self, info):
        self.text = str.lower(info['value']['properties']['text'])
        self.coordinates = info['value']['geometry']['coordinates']
        self.sentiment_score = 0

    def cal_sentiment_score(self, sentiment_dic):
        """
        calculate sentiment_score
        :param sentiment_dic: dictionary, key: word, value: sentiment score
        """
        words = self.text.split()
        for word in words:
            # check if the word is with a punctuation
            if word[-1] in ['!', ',', '?', '.', "'", '"']:
                word = word[:-1]
            if word in sentiment_dic.keys():
                self.sentiment_score += sentiment_dic[word]

    def update_score(self, regions):
        """
        update sentiment_score to corresponding Cell
        :param regions: list(Region)
        """
        for region in regions:
            if region.xmin <= self.coordinates[0] < region.xmax and region.ymin <= self.coordinates[1] < region.ymax:
                region.count +=1
                region.score += self.sentiment_score


    #
    def print_all(self):
        print(self.text)
        print(self.coordinates)


def read_grid_info(grid_file_name):
    """

    :param grid_file_name: (str) jason file
    :return:
    """
    with open(grid_file_name, encoding='utf-8') as f:
        grid_info = json.load(f)['features']
    regions = []
    for i in grid_info:
        regions.append(Region(i))
    return regions



def read_twitters_data(s):
    s = s.strip().rstrip(',')
    try:
        out = json.loads(s)
        return out
    except:
        try:
            out = json.loads(s[:-2])
            return out
        except:
            return


def read_sentiment_data(sentiment_file_name):
    # read sentiment file
    with open(sentiment_file_name, "r") as f:
        data = f.read()

    sentiment_data = [i.split('\t') for i in data.split('\n')]
    # create sentiment dictionary
    sentiment_dic = {}
    for i, j in sentiment_data:
        sentiment_dic[i] = int(j)
    return sentiment_dic


def sum_regions(regions_list):
    cnt_tweets = defaultdict(int)
    cnt_score = defaultdict(int)
    for regions in regions_list:
        for region in regions:
            cnt_tweets[region.region_id] += region.count
            cnt_score[region.region_id] +=region.score
    print(cnt_tweets, cnt_score)






if __name__ == '__main__':
    tweets_file = r'C:\Users\enzon\Desktop\ccc_1\data\smallTwitter.json'
    sentiment_dic_file_name = "data/AFINN.txt"
    grid_file_name = 'data/melbGrid.json'
    message_file_nmae = 'data/tinyTwitter.json'

    if rank == 0:
        # read grid data
        regions = read_grid_info(grid_file_name)
        sentiment_dic = read_sentiment_data(sentiment_dic_file_name)
    else:
        regions = None
        sentiment_dic = None
    # broadcast grid data
    regions = comm.bcast(regions, root=0)
    sentiment_dic = comm.bcast(sentiment_dic, root=0)


    if not regions or not sentiment_dic:
        raise FileNotFoundError()

    # TODO 检查 regions sentiment_dic 是否正确读入
    # print('rank:', rank)
    # for i in regions:
    #     i.print_all()


    # read sentiment file
    if rank == 0:
        with open(tweets_file, 'r', encoding='utf-8') as twitter_file:
            # read total_rows and offset info
            twitters_data = next(twitter_file)
            i = 0
            while twitters_data:
                twitters_data = list(islice(twitter_file, BATCH_SIZE))
                comm.send(twitters_data, dest=i % (size - 1) + 1, tag=4)
                i += 1
        for i in range(1, size):
            comm.send(None, dest=i, tag=4)

    else:
        while True:
            twitters_data = comm.recv(source=0, tag=4)

            if twitters_data is None:
                break

            new = list(map(read_twitters_data, twitters_data))
            for i in new:
                try:
                    curr_message = Message(i)
                    curr_message.cal_sentiment_score(sentiment_dic)
                    curr_message.update_score(regions)
                except:
                    continue

    regions = comm.gather(regions, root=0)
    if rank == 0:
        sum_regions(regions)
