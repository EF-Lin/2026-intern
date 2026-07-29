from obspy import read, Stream


def load_data(files: list) -> Stream:
    data = Stream()
    for f in files:
        data += read(f)
    return data
