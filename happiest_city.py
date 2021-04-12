import json
import sys
from itertools import islice, product
from collections import defaultdict

from mpi4py import MPI

BATCH_SIZE = 32


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

    def cal_sentiment_score(self, sentiment_dic, max_sentiment_word_len):
        """
        calculate sentiment_score
        :param sentiment_dic: dictionary, key: word, value: sentiment score
        """
        words = self.text.split()
        for word_len in range(max_sentiment_word_len):
            for i in range(len(words) - word_len):
                curr = ' '.join(words[i:i + word_len + 1])
                if curr[-1] in ['!', ',', '?', '.', "'", '"']:
                    curr = curr[:-1]
                if curr in sentiment_dic.keys():
                    self.sentiment_score += sentiment_dic[curr]

    def update_score(self, regions):
        """
        update sentiment_score to corresponding Cell
        :param regions: list(Region)
        """
        for region in regions:
            if region.xmin < self.coordinates[0] <= region.xmax and region.ymin < self.coordinates[1] <= region.ymax:
                region.count += 1
                region.score += self.sentiment_score


def read_grid_info(grid_file_name):
    """
    :param grid_file_name: (str) jason file
    :return: list(Region) list of Cells
    """
    with open(grid_file_name, encoding='utf-8') as f:
        grid_info = json.load(f)['features']
    regions = []
    for i in grid_info:
        regions.append(Region(i))
    return regions


def read_twitters_data(s):
    """
    :param s: str all information for current Tweets
    :return: dictionary: convert str into dictionary
    """
    s = s.strip().rstrip(',]}') + '}}'
    try:
        out = json.loads(s)
        return out
    except:
        return


def read_sentiment_data(sentiment_file_name):
    """
    convert sentiment file into sentiment score dictionary
    :param sentiment_file_name: str
    :return: dictionary key:word, value: sentiment score
    """
    with open(sentiment_file_name, "r") as f:
        data = f.read()

    sentiment_data = [i.split('\t') for i in data.split('\n')]
    sentiment_dic = {}
    for i, j in sentiment_data:
        sentiment_dic[i] = int(j)
    return sentiment_dic


def sum_regions(regions_list):
    """
    since parallel programming is implemented, there are more than one list of Region(Cell), this function is used to
    sum up number of tweets and sentiment score.
    :param regions_list: list(Regions)
    """
    cnt_tweets = defaultdict(int)
    cnt_score = defaultdict(int)
    for regions in regions_list:
        for region in regions:
            cnt_tweets[region.region_id] += region.count
            cnt_score[region.region_id] += region.score
    cells = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4', 'C5', 'D3', 'D4', 'D5']
    print(f'{"Cell":<6}' f'{"#Total Tweets":^18}'  f'{"#Overal Sentiment Score":^25}')
    for i in cells:
        print(f'{i:<8}' f'{cnt_tweets[i]:^15}' f'{cnt_score[i]:^25}')


def main(rank, size, tweets_file, grid_file_name, sentiment_dic_file_name):
    # if only have one process
    if size == 1:
        # read grid data
        regions = read_grid_info(grid_file_name)
        sentiment_dic = read_sentiment_data(sentiment_dic_file_name)
        max_sentiment_word_len = max([len(i.split()) for i in sentiment_dic.keys()])

        with open(tweets_file, 'r', encoding='utf-8') as twitter_file:
            # read total_rows and offset info
            twitters_data = next(twitter_file)
            i = 0
            while twitters_data:
                twitters_data = list(islice(twitter_file, BATCH_SIZE))

                twitters_data = list(map(read_twitters_data, twitters_data))
                for j in twitters_data:
                    try:
                        curr_message = Message(j)
                        curr_message.cal_sentiment_score(sentiment_dic, max_sentiment_word_len)
                        curr_message.update_score(regions)
                    except:
                        continue

        # collect and sum up information
        sum_regions([regions])
    # have more than one process
    else:
        if rank == 0:
            # read grid data
            regions = read_grid_info(grid_file_name)
            sentiment_dic = read_sentiment_data(sentiment_dic_file_name)
            max_sentiment_word_len = max([len(i.split()) for i in sentiment_dic.keys()])
        else:
            regions, sentiment_dic, max_sentiment_word_len = None, None, None
        # broadcast grid data
        regions = comm.bcast(regions, root=0)
        sentiment_dic = comm.bcast(sentiment_dic, root=0)
        max_sentiment_word_len = comm.bcast(max_sentiment_word_len, root=0)

        if not regions or not sentiment_dic:
            raise FileNotFoundError()

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
                # process twitters data
                twitters_data = comm.recv(source=0, tag=4)

                if twitters_data is None:
                    break

                twitters_data = list(map(read_twitters_data, twitters_data))
                for i in twitters_data:
                    try:
                        curr_message = Message(i)
                        curr_message.cal_sentiment_score(sentiment_dic, max_sentiment_word_len)
                        curr_message.update_score(regions)
                    except:
                        continue

        # collect and sum up information
        regions = comm.gather(regions, root=0)
        if rank == 0:
            sum_regions(regions)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    try:
        grid_file_name = sys.argv[1]
        sentiment_dic_file_name = sys.argv[2]
        tweets_file = sys.argv[3]
    except:
        tweets_file = 'data/smallTwitter.json'
        grid_file_name = 'data/melbGrid.json'
        sentiment_dic_file_name = "data/AFINN.txt"

    main(rank, size, tweets_file, grid_file_name, sentiment_dic_file_name)
