import collections
class MovingAverageFilter:
    def __init__(self, len=10):
        if len <= 0:
            raise ValueError(f'Average length must be greater than zero: {len}')
        self.__buff = collections.deque([0] * len, maxlen = len)
        self.__N = len
        self.__cumsum = 0

    def filt(self, newval):
        self.__cumsum += (newval - self.__buff.popleft())
        self.__buff.append(newval)
        return self.__cumsum/self.__N
